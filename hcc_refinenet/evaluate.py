import os
import math
from typing import Iterable, List, Dict, Tuple
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import json
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism

from config import CFG
from dataset import make_folds, make_loaders, df_to_items
from transforms import make_val_transforms
from model import build_model

from monai.data import CacheDataset
from monai.data.utils import list_data_collate
from torch.utils.data import DataLoader

try:
    from monai.metrics import HausdorffDistanceMetric, SurfaceDistanceMetric
    HAS_HD = True
except Exception:
    HAS_HD = False


def _to_numpy_bool(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy()
    x = np.asarray(x)
    # squeeze any leading 1s
    while x.ndim > 3 and x.shape[0] == 1:
        x = x[1 - 1]  # drop leading axis
    assert x.ndim == 3, f"expected 3D, got {x.shape}"
    return x.astype(bool)


def _binary_stats(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    tp = np.logical_and(pred, gt).sum(dtype=np.int64)
    fp = np.logical_and(pred, ~gt).sum(dtype=np.int64)
    fn = np.logical_and(~pred, gt).sum(dtype=np.int64)
    tn = np.logical_and(~pred, ~gt).sum(dtype=np.int64)

    eps = 1e-8
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    spec = tn / (tn + fp + eps)
    return dict(dice=float(dice), iou=float(iou), precision=float(prec), recall=float(rec), specificity=float(spec))


def _hd_metrics(pred: torch.Tensor, gt: torch.Tensor, spacing: Tuple[float, float, float]):
    if not HAS_HD:
        return math.nan, math.nan

    h = HausdorffDistanceMetric(
        include_background=False, percentile=95.0, reduction="mean", directed=False
    )
    s = SurfaceDistanceMetric(
        include_background=False, symmetric=True, reduction="mean"
    )

    hd = h(y_pred=pred, y=gt, spacing=spacing)
    msd = s(y_pred=pred, y=gt, spacing=spacing)

    try:
        hdv = float(hd.item())
    except Exception:
        hdv = float(hd.mean().item())
    try:
        msdv = float(msd.item())
    except Exception:
        msdv = float(msd.mean().item())
    return hdv, msdv


def _load_model_from_ckpt(ckpt_path: str, in_channels: int, device: torch.device):
    m = build_model(in_channels=in_channels).to(device)
    state = torch.load(ckpt_path, map_location="cpu")
    m.load_state_dict(state["state_dict"])
    m.eval()
    return m


def _infer_logits(models: List[torch.nn.Module],
                  x: torch.Tensor,
                  roi_size: Tuple[int, int, int],
                  overlap: float) -> np.ndarray:
    def _sigmoid(z): return 1.0 / (1.0 + np.exp(-z))
    def _prob_to_logit(p, eps=1e-6): p = np.clip(p, eps, 1 - eps); return np.log(p / (1 - p))

    logits_sum = None
    use_amp = torch.cuda.is_available()
    amp_ctx = torch.autocast(device_type="cuda") if use_amp else torch.cpu.amp.autocast(enabled=False)
    with torch.no_grad(), amp_ctx:
        for m in models:
            preds = sliding_window_inference(x, roi_size=roi_size, sw_batch_size=1,
                                             predictor=m, overlap=overlap)  # [1,1,Z,Y,X]
            prob = torch.sigmoid(preds)[0, 0].float().cpu().numpy()
            logit = _prob_to_logit(prob)
            logits_sum = logit if logits_sum is None else (logits_sum + logit)
    mean_logit = logits_sum / float(len(models))
    return _sigmoid(mean_logit)


def evaluate(
    fold: int = None,
    ensemble_folds: Iterable[int] = None,
    thresh: float = 0.97,
    save_csv_name: str = "eval_metrics.csv",
    out_dir: str = None
):
    cfg = CFG()
    if out_dir is None:
        out_dir = cfg.OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    set_determinism(seed=cfg.SEED)

    df = pd.read_csv(cfg.CSV_PATH)

    if ensemble_folds is not None:
        split_fold = list(ensemble_folds)[0]
    elif fold is not None:
        split_fold = int(fold)
    else:
        split_fold = cfg.FOLD_IDX

    folds = make_folds(df, n_folds=cfg.N_FOLDS, seed=cfg.SEED)
    tr_idx, va_idx = folds[split_fold]
    val_df = df.iloc[va_idx].reset_index(drop=True)

    val_tf = make_val_transforms(cfg)

    _, val_loader = make_loaders(cfg, train_df=df.iloc[tr_idx], val_df=val_df,
                                 train_tf=val_tf, val_tf=val_tf, cache_rate=0.2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models: List[torch.nn.Module] = []
    if ensemble_folds is not None:
        for f in ensemble_folds:
            ckpt = os.path.join(cfg.fold_dir(int(f)), "best.pt")
            if os.path.exists(ckpt):
                models.append(_load_model_from_ckpt(ckpt, cfg.IN_CHANNELS, device))
            else:
                print(f"[warn] missing ckpt for fold {f}: {ckpt}")
        if not models:
            raise FileNotFoundError("No checkpoints found for ensemble_folds.")
        print(f"[info] evaluating ensemble over folds: {list(ensemble_folds)}")
    else:
        ckpt = os.path.join(cfg.fold_dir(split_fold), "best.pt")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
        models.append(_load_model_from_ckpt(ckpt, cfg.IN_CHANNELS, device))
        print(f"[info] evaluating single fold {split_fold} from {ckpt}")

    rows = []
    for batch in tqdm(val_loader, desc="Evaluating"):
        pid = batch.get("patient_id", [None])[0] if isinstance(batch.get("patient_id", None), list) else None

        x = batch["img"].to(device)
        y = batch["lab"]
        roi = batch.get("roi", None)

        prob = _infer_logits(models, x, cfg.ROI_SIZE, cfg.SW_OVERLAP)

        if roi is not None:
            roi_np = _to_numpy_bool(roi[0])
            prob = prob * roi_np.astype(np.float32)

        seg = (prob > float(thresh)).astype(np.uint8)

        gt = _to_numpy_bool(y[0])

        stats = _binary_stats(seg.astype(bool), gt.astype(bool))

        if HAS_HD:
            pred_t = torch.from_numpy(seg[None, None].astype(np.float32))
            gt_t   = torch.from_numpy(gt[None, None].astype(np.float32))
            hd95, msd = _hd_metrics(pred_t, gt_t, spacing=cfg.SPACING)
        else:
            hd95, msd = math.nan, math.nan

        rows.append({
            "patient_id": pid,
            "dice": stats["dice"],
            "iou": stats["iou"],
            "precision": stats["precision"],
            "recall": stats["recall"],
            "specificity": stats["specificity"],
            "hd95": hd95,
            "msd": msd,
            "voxels_pred": int(seg.sum()),
            "voxels_gt": int(gt.sum()),
        })

    out_df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, save_csv_name)
    out_df.to_csv(save_path, index=False)

    def agg(col):
        v = out_df[col].dropna().values.astype(float)
        return float(np.mean(v)), float(np.std(v))

    md, sd = agg("dice")
    mi, si = agg("iou")
    mp, sp = agg("precision")
    mr, sr = agg("recall")
    ms, ss = agg("specificity")
    if HAS_HD:
        mh, sh = agg("hd95")
        mm, sm = agg("msd")
    else:
        mh = sh = mm = sm = float("nan")

    metrics = {
        "dice":        {"mean": md, "std": sd},
        "iou":         {"mean": mi, "std": si},
        "precision":   {"mean": mp, "std": sp},
        "recall":      {"mean": mr, "std": sr},
        "specificity": {"mean": ms, "std": ss},
        "hd95":        {"mean": mh, "std": sh},
        "msd":         {"mean": mm, "std": sm},
    }
    os.makedirs(CFG().OUT_DIR, exist_ok=True)
    save_path = os.path.join(CFG().OUT_DIR, save_csv_name)
    out_df.to_csv(save_path, index=False)

    json_path = save_path.replace(".csv", ".json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Validation metrics ===")
    print(f"Dice                 : {md:.4f} ± {sd:.4f}")
    print(f"IoU                  : {mi:.4f} ± {si:.4f}")
    print(f"Precision            : {mp:.4f} ± {sp:.4f}")
    print(f"Recall               : {mr:.4f} ± {sr:.4f}")
    print(f"Specificity          : {ms:.4f} ± {ss:.4f}")
    if HAS_HD:
        print(f"HD95 (mm)            : {mh:.2f} ± {sh:.2f}")
        print(f"Mean Surf Dist. (mm) : {mm:.2f} ± {sm:.2f}")
    print(f"\nSaved per-case metrics to: {save_path}")
    return metrics

def evaluate_full_dataset_ensemble(
    ensemble_folds: Iterable[int] = (0, 1, 2, 3, 4),
    thresh: float = 0.97,
    save_csv_name: str = "eval_ens_ALL.csv",
):
    cfg = CFG()
    set_determinism(seed=cfg.SEED)
    df = pd.read_csv(cfg.CSV_PATH)

    val_tf = make_val_transforms(cfg)
    val_items = df_to_items(df.reset_index(drop=True))
    val_ds = CacheDataset(val_items, transform=val_tf, cache_rate=0.2, num_workers=0)
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=list_data_collate,
        pin_memory=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models: List[torch.nn.Module] = []
    missing = []
    for f in ensemble_folds:
        ckpt = os.path.join(cfg.fold_dir(int(f)), "best.pt")
        if os.path.exists(ckpt):
            models.append(_load_model_from_ckpt(ckpt, cfg.IN_CHANNELS, device))
        else:
            missing.append((f, ckpt))
    if not models:
        raise FileNotFoundError("No checkpoints found for requested ensemble_folds.")
    if missing:
        print("[warn] missing checkpoints (skipped): " + ", ".join(f"f{f}:{p}" for f, p in missing))
    print(f"[info] evaluating FULL DATASET with ensemble over folds: {list(ensemble_folds)}")

    rows = []
    for batch in tqdm(val_loader, desc="Evaluating (ALL patients)"):
        pid = batch.get("patient_id", [None])[0] if isinstance(batch.get("patient_id", None), list) else None

        x = batch["img"].to(device)
        y = batch.get("lab", None)
        roi = batch.get("roi", None)

        prob = _infer_logits(models, x, cfg.ROI_SIZE, cfg.SW_OVERLAP)

        if roi is not None:
            roi_np = _to_numpy_bool(roi[0])
            prob = prob * roi_np.astype(np.float32)

        seg = (prob > float(thresh)).astype(np.uint8)

        if y is None:
            rows.append({
                "patient_id": pid, "dice": np.nan, "iou": np.nan,
                "precision": np.nan, "recall": np.nan, "specificity": np.nan,
                "hd95": np.nan, "msd": np.nan,
                "voxels_pred": int(seg.sum()), "voxels_gt": np.nan,
            })
            continue

        gt = _to_numpy_bool(y[0])

        stats = _binary_stats(seg.astype(bool), gt.astype(bool))

        if HAS_HD:
            pred_t = torch.from_numpy(seg[None, None].astype(np.float32))
            gt_t   = torch.from_numpy(gt[None, None].astype(np.float32))
            hd95, msd = _hd_metrics(pred_t, gt_t, spacing=cfg.SPACING)
        else:
            hd95, msd = math.nan, math.nan

        rows.append({
            "patient_id": pid,
            "dice": stats["dice"],
            "iou": stats["iou"],
            "precision": stats["precision"],
            "recall": stats["recall"],
            "specificity": stats["specificity"],
            "hd95": hd95,
            "msd": msd,
            "voxels_pred": int(seg.sum()),
            "voxels_gt": int(gt.sum()),
        })

    out_df = pd.DataFrame(rows)
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    save_path = os.path.join(cfg.OUT_DIR, save_csv_name)
    out_df.to_csv(save_path, index=False)

    def agg(col):
        v = out_df[col].dropna().values.astype(float)
        if len(v) == 0:
            return float("nan"), float("nan")
        return float(v.mean()), float(v.std())

    md, sd = agg("dice"); mi, si = agg("iou"); mp, sp = agg("precision"); mr, sr = agg("recall"); ms, ss = agg("specificity")
    if HAS_HD:
        mh, sh = agg("hd95"); mm, sm = agg("msd")
    else:
        mh = sh = mm = sm = float("nan")

    print("\n=== Full-dataset Ensemble metrics (ALL patients) ===")
    print(f"Dice                 : {md:.4f} ± {sd:.4f}")
    print(f"IoU                  : {mi:.4f} ± {si:.4f}")
    print(f"Precision            : {mp:.4f} ± {sp:.4f}")
    print(f"Recall               : {mr:.4f} ± {sr:.4f}")
    print(f"Specificity          : {ms:.4f} ± {ss:.4f}")
    if HAS_HD:
        print(f"HD95 (mm)            : {mh:.2f} ± {sh:.2f}")
        print(f"Mean Surf Dist. (mm) : {mm:.2f} ± {sm:.2f}")
    print(f"Saved per-case metrics to: {save_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("Evaluate tumor segmentation")

    ap.add_argument("--fold", type=int, default=None,
                    help="single fold index to evaluate (0..4). Ignored if --ensemble_folds is given.")
    ap.add_argument("--ensemble_folds", type=int, nargs="+", default=None,
                    help="e.g., --ensemble_folds 0 1 2 3 4. Uses the first fold to define the val split.")
    ap.add_argument("--thresh", type=float, default=0.5, help="binarization threshold for probs.")
    ap.add_argument("--save_csv_name", type=str, default="eval_metrics.csv")

    ap.add_argument("--ens_all_patients", action="store_true",
                    help="Evaluate ALL patients in CFG.CSV_PATH using an ensemble of folds (predict.py style).")
    ap.add_argument("--ens_folds", type=int, nargs="+", default=[0,1,2,3,4],
                    help="Which fold indices to ensemble with --ens_all_patients.")

    args = ap.parse_args()

    if args.ens_all_patients:
        evaluate_full_dataset_ensemble(
            ensemble_folds=args.ens_folds,
            thresh=args.thresh,
            save_csv_name=args.save_csv_name,
        )
    else:
        evaluate(
            fold=args.fold,
            ensemble_folds=args.ensemble_folds,
            thresh=args.thresh,
            save_csv_name=args.save_csv_name,
        )