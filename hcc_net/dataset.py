import cv2, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset
from typing import Dict, Any, List
from config import CFG
from transforms import get_train_tfms, get_valid_tfms
import os

NAME2IDX = {n:i for i,n in enumerate(CFG.CLASSES)}

try:
    import nibabel as nib
    _HAVE_NIB = True
except Exception:
    _HAVE_NIB = False

def _resolve_path(p: str) -> str:
    if os.path.isabs(p):
        return p
    base = getattr(CFG, "IMG_ROOT", "") or os.path.dirname(CFG.CSV_PATH)
    if not (p.startswith("CECT/") or p.startswith("CECT\\")):
        p = os.path.join("CECT", p)
    return os.path.normpath(os.path.join(base, p))

def _read_gray(p: str) -> np.ndarray:
    fp = _resolve_path(p)
    if not os.path.exists(fp):
        print(f"[WARN] Missing file: {fp}, using blank phase.")
        return np.zeros((CFG.IMG_SIZE, CFG.IMG_SIZE), dtype=np.float32)

    if fp.lower().endswith((".nii", ".nii.gz")):
        if not _HAVE_NIB:
            raise RuntimeError("Install nibabel: pip install nibabel")
        try:
            arr = nib.load(fp).get_fdata()
        except Exception as e:
            print(f"[WARN] Failed to load NIfTI: {fp} ({e}), using blank phase.")
            return np.zeros((CFG.IMG_SIZE, CFG.IMG_SIZE), dtype=np.float32)
        if arr.size == 0:
            print(f"[WARN] Empty NIfTI: {fp}, using blank phase.")
            return np.zeros((CFG.IMG_SIZE, CFG.IMG_SIZE), dtype=np.float32)
        arr = _to_2d(arr).astype(np.float32)
        return arr

    im = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
    if im is None:
        print(f"[WARN] OpenCV failed to read: {fp}, using blank phase.")
        return np.zeros((CFG.IMG_SIZE, CFG.IMG_SIZE), dtype=np.float32)
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return im.astype(np.float32)

def _load_slices(p: str) -> np.ndarray:
    fp = _resolve_path(p)
    k = CFG.SLICES_PER_PHASE // 2
    n_slices = CFG.SLICES_PER_PHASE

    if not os.path.exists(fp):
        print(f"[WARN] Missing file  {fp}, using blank phase.")
        blank = np.zeros((CFG.IMG_SIZE, CFG.IMG_SIZE), dtype=np.float32)
        return np.stack([blank] * n_slices, axis=-1)

    if fp.lower().endswith((".nii", ".nii.gz")):
        if not _HAVE_NIB:
            raise RuntimeError("Install nibabel: pip install nibabel")
        try:
            vol = nib.load(fp).get_fdata()
        except Exception as e:
            print(f"[WARN] Failed to load NIfTI  {fp} ({e}), using blank.")
            blank = np.zeros((CFG.IMG_SIZE, CFG.IMG_SIZE), dtype=np.float32)
            return np.stack([blank] * n_slices, axis=-1)

        vol = np.squeeze(vol)
        while vol.ndim > 3:
            vol = vol[..., 0]
        if vol.ndim == 2:
            vol = vol[..., None]

        depth = vol.shape[-1]
        if depth <= 0:
            print(f"[WARN] Empty depth in NIfTI  {fp}, using blank.")
            blank = np.zeros((CFG.IMG_SIZE, CFG.IMG_SIZE), dtype=np.float32)
            return np.stack([blank] * n_slices, axis=-1)

        mid = depth // 2
        idxs = np.arange(mid - k, mid + k + 1)
        idxs = np.clip(idxs, 0, depth - 1)

        chans = []
        for idx in idxs:
            sl = vol[..., idx].astype(np.float32)
            sl = _hu_window(sl, CFG.WINDOW_CENTER, CFG.WINDOW_WIDTH)
            sl = _resize_to_match(sl, target_shape=(CFG.IMG_SIZE, CFG.IMG_SIZE))
            chans.append(sl)
        return np.stack(chans, axis=-1)

    im = _read_gray(p)  # 2D float32
    im = _hu_window(im, CFG.WINDOW_CENTER, CFG.WINDOW_WIDTH)
    im = _resize_to_match(im, target_shape=(CFG.IMG_SIZE, CFG.IMG_SIZE))
    return np.stack([im] * n_slices, axis=-1)

def _to_2d(arr: np.ndarray) -> np.ndarray:
    if arr is None:
        raise ValueError("None passed to _to_2d")

    arr = np.asarray(arr)
    arr = np.squeeze(arr)  # drop size-1 dims


    if arr.ndim == 2:
        return arr

    while arr.ndim > 3:
        arr = arr[..., 0]

    if arr.ndim == 3:
        depth_axis = -1
        mid = arr.shape[depth_axis] // 2
        arr = np.take(arr, indices=mid, axis=depth_axis)
        return arr

    return arr

def _resize_to_match(img: np.ndarray, target_shape=(512, 512)) -> np.ndarray:
    if img is None or not isinstance(img, np.ndarray):
        raise ValueError("Empty image passed to _resize_to_match()")
    img = _to_2d(img)
    h, w = img.shape[:2]
    th, tw = target_shape
    if h == 0 or w == 0:
        raise ValueError(f"Invalid image with shape {img.shape}")
    out = cv2.resize(img.astype(np.float32), (tw, th), interpolation=cv2.INTER_AREA)
    return out.astype(np.float32)

def _hu_window(img: np.ndarray, center: int, width: int) -> np.ndarray:
    if img is None:
        return None
    if img.dtype not in (np.uint16, np.int16, np.int32, np.float32):
        return (img.astype(np.float32) / 255.0)
    
    lo, hi = center - width//2, center + width//2
    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo + 1e-6)
    return img.astype(np.float32)

def _tight_bbox(mask: np.ndarray, pad: int = 8):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return max(y1-pad,0), max(x1-pad,0), y2+pad, x2+pad

def _crop(img: np.ndarray, bbox):
    if bbox is None: return img
    y1, x1, y2, x2 = bbox
    return img[y1:y2, x1:x2]

class HCCDataset(Dataset):
    def __init__(self, df: pd.DataFrame, training: bool):
        self.df = df.reset_index(drop=True)
        self.training = training
        self.tfms = get_train_tfms() if training else get_valid_tfms()

    def __len__(self): return len(self.df)

    def _load_phases(self, row) -> np.ndarray:
        vols = []
        fallback = None

        for ph in CFG.PHASES:
            col = f"path_{ph}"
            p = row[col] if col in row and isinstance(row[col], str) and row[col] else ""
            if p:
                v = _load_slices(p)
                if fallback is None:
                    fallback = v
            else:
                v = None
            vols.append(v)

        base = fallback if fallback is not None else vols[0]
        vols = [base if v is None else v for v in vols]

        arr = np.concatenate(vols, axis=-1)
        return arr

    def _maybe_crop_with_liver(self, arr: np.ndarray, mask_path: str) -> np.ndarray:
        if not CFG.USE_LIVER_MASK or not mask_path or not isinstance(mask_path, str):
            return arr
        m = _read_gray(mask_path)

        if m.dtype != np.uint8: m = (m > 0).astype(np.uint8) * 255
        bbox = _tight_bbox(m)
        if bbox is None:
            return arr
        cropped = np.stack([_crop(arr[...,c], bbox) for c in range(arr.shape[-1])], axis=-1)
        return cropped

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        arr = self._load_phases(row)

        arr = self._maybe_crop_with_liver(arr, row.get("liver_mask_path",""))

        if CFG.ADD_MASK_AS_CHANNEL and row.get("liver_mask_path",""):
            m = _read_gray(row["liver_mask_path"])
            if m is not None:
                if arr.shape[:2] != m.shape:
                    # simple resize to match arr size before augmentation stage
                    m = cv2.resize(m, (arr.shape[1], arr.shape[0]), interpolation=cv2.INTER_NEAREST)
                m = (m > 0).astype(np.float32)
                arr = np.concatenate([arr, m[...,None]], axis=-1)

        out = self.tfms(image=arr)
        x = out["image"]            # CHW
        y = torch.tensor(NAME2IDX[row["label"]], dtype=torch.long)
        return {"image": x, "label": y, "pid": row["patient_id"]}

