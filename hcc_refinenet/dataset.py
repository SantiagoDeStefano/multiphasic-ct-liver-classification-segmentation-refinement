import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from monai.data import CacheDataset
from monai.data.utils import list_data_collate
from config import CFG


def _resolve_path(p):
    if p is None:
        return None
    if isinstance(p, float) and pd.isna(p):
        return None
    if isinstance(p, str) and p.strip() in ("", "nan", "None"):
        return None

    p = str(p)
    if os.path.isabs(p):
        return os.path.normpath(p)

    base = getattr(CFG, "IMG_ROOT", "") or os.path.dirname(CFG().CSV_PATH)
    base_norm = os.path.normpath(base)

    rel = p.replace("\\", "/").lstrip("/")
    parts = rel.split("/")
    if parts and parts[0].lower() == "cect" and os.path.basename(base_norm).lower() == "cect":
        parts = parts[1:]
    rel = os.path.join(*parts) if parts else ""

    return os.path.normpath(os.path.join(base_norm, rel))

def make_folds(df, n_folds, seed=2025):
    strat = (df[["path_mask_C1", "path_mask_C2", "path_mask_C3"]].notna().sum(axis=1) > 0).astype(int)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(skf.split(df.index, strat))

def df_to_items(df: pd.DataFrame):
    items = []
    for _, r in df.iterrows():
        pC1 = _resolve_path(r.get("path_C1", None))
        if pC1 is None:
            continue
        item = {
            "patient_id": r["patient_id"],
            # images: C1 only
            "path_C1": _resolve_path(r.get("path_C1", None)),
            "path_C2": _resolve_path(r.get("path_C2", None)),
            "path_C3": _resolve_path(r.get("path_C3", None)),
            # masks: any/all phases (optional)
            "path_mask_C1": _resolve_path(r.get("path_mask_C1", None)),
            "path_mask_C2": _resolve_path(r.get("path_mask_C2", None)),
            "path_mask_C3": _resolve_path(r.get("path_mask_C3", None)),
        }
        items.append(item)
    return items

def make_loaders(cfg: CFG, train_df, val_df, train_tf, val_tf, cache_rate=0.2):
    train_items = df_to_items(train_df)
    val_items = df_to_items(val_df)

    train_ds = CacheDataset(train_items, transform=train_tf, cache_rate=cache_rate, num_workers=0)
    val_ds   = CacheDataset(val_items,   transform=val_tf,   cache_rate=cache_rate, num_workers=0)

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.NUM_WORKERS, collate_fn=list_data_collate, pin_memory=False)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False,
                              num_workers=cfg.NUM_WORKERS, collate_fn=list_data_collate, pin_memory=False)
    return train_loader, val_loader