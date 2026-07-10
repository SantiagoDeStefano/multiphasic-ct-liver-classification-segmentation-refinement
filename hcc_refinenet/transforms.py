# transforms.py
import numpy as np
from typing import Sequence, List, Tuple
from monai.transforms import (
    Compose, LoadImaged, CopyItemsd, DeleteItemsd, EnsureChannelFirstd, EnsureTyped,
    Orientationd, Spacingd, ResampleToMatchd, SqueezeDimd, RandFlipd, RandRotate90d,
    RandScaleIntensityd, RandShiftIntensityd, RandGaussianNoised, RandCropByPosNegLabeld,
    MapTransform, Lambdad, GaussianSmoothd, ConcatItemsd, SpatialPadd, SelectItemsd, CropForegroundd
)
from monai.data import MetaTensor
from monai.config import KeysCollection
from monai.utils import InterpolateMode
import torch
import torch.nn.functional as F
from config import CFG


def make_predict_transforms(cfg: CFG):
    return Compose([
        _common_io_and_geometry(cfg),
        _to_six_channels_and_types(cfg),
        SelectItemsd(keys=["img"]),
        EnsureTyped(keys=["img"], dtype=[np.float32], track_meta=True),
    ])

class Ensure3DROId(MapTransform):
    def __init__(self, keys=("roi",)):
        super().__init__(keys)
    def __call__(self, data):
        import numpy as np
        d = dict(data)
        for k in self.keys:
            r = d.get(k, None)
            if r is None: 
                continue
            r = np.asarray(r)
            if r.ndim == 4 and r.shape[0] == 1:
                r = r[0]
            if r.ndim == 2:
                r = r[None, ...]
            if r.ndim != 3:
                raise ValueError(f"[Ensure3DROId] unexpected shape {r.shape}")
            d[k] = r
        return d


class MakeROIdFromLabd(MapTransform):
    def __init__(self, lab_key="lab", roi_key="roi", dilate_k=(5,5,5)):
        super().__init__([lab_key])
        self.lab_key = lab_key
        self.roi_key = roi_key
        self.k = tuple(int(x) for x in dilate_k) if dilate_k is not None else None

    def __call__(self, data):
        d = dict(data)
        lab = d.get(self.lab_key, None)
        if lab is None:
            return d
        import torch
        x = torch.as_tensor(lab).float()
        if x.ndim == 3: x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 4: x = x.unsqueeze(0)
        x = (x > 0).float()
        if self.k is not None:
            pz, py, px = (k//2 for k in self.k)
            pad = (px, px, py, py, pz, pz)
            x = torch.nn.functional.pad(x, pad, mode="replicate")
            x = torch.nn.functional.max_pool3d(x, kernel_size=self.k, stride=1)
        roi = (x[0,0] > 0).to(torch.uint8)
        d[self.roi_key] = roi
        return d

class ApplyROIMaskd(MapTransform):
    def __init__(self, img_keys=("img_c1","img_c2","img_c3"), lab_key="lab", roi_key="roi"):
        super().__init__(list(img_keys) + [lab_key, roi_key])
        self.img_keys = list(img_keys)
        self.lab_key = lab_key
        self.roi_key = roi_key

    @staticmethod
    def _match_shape(roi: torch.Tensor, target_shape_zyx: Tuple[int, int, int]) -> torch.Tensor:
        if roi.ndim == 3:
            roi = roi.unsqueeze(0)
        roi = roi.to(torch.float32)

        _, z, y, x = roi.shape
        tz, ty, tx = target_shape_zyx

        def center_crop(start_len, target_len):
            if start_len <= target_len:
                return slice(0, start_len)
            off = (start_len - target_len) // 2
            return slice(off, off + target_len)

        cz = center_crop(z, tz)
        cy = center_crop(y, ty)
        cx = center_crop(x, tx)
        roi = roi[:, cz, cy, cx]

        _, z, y, x = roi.shape
        pz = max(0, tz - z)
        py = max(0, ty - y)
        px = max(0, tx - x)
        if pz or py or px:
            pad = (px // 2, px - px // 2, py // 2, py - py // 2, pz // 2, pz - pz // 2)
            roi = F.pad(roi, pad, value=0.0)

        return roi

    def __call__(self, data):
        d = dict(data)

        if self.roi_key not in d or d[self.roi_key] is None:
            return d

        roi = torch.as_tensor(d[self.roi_key])

        for k in self.img_keys:
            if k in d and d[k] is not None:
                x = torch.as_tensor(d[k]).to(torch.float32)
                target_zyx = tuple(int(v) for v in x.shape[-3:])
                roi_matched = self._match_shape(roi, target_zyx)
                # Broadcast over channels
                d[k] = (x * roi_matched).type_as(x)

        if self.lab_key in d and d[self.lab_key] is not None:
            y = torch.as_tensor(d[self.lab_key])
            if y.ndim == 3:
                y = y.unsqueeze(0)
            target_zyx = tuple(int(v) for v in y.shape[-3:])
            roi_matched = self._match_shape(roi, target_zyx)
            y = (y.to(torch.float32) * roi_matched).round().to(torch.uint8)
            d[self.lab_key] = y.squeeze(0)

        return d


class EnsurePhaseKeysEarlyd(MapTransform):
    def __init__(self, keys=("img_c2","img_c3"), ref_key="img_c1"):
        super().__init__(keys)
        self.ref_key = ref_key
    def __call__(self, data):
        d = dict(data)
        ref = d.get(self.ref_key, None)
        if ref is None:
            return d
        import numpy as np
        ref_np = np.asarray(ref)
        for k in self.keys:
            if k not in d or d[k] is None:
                d[k] = np.zeros_like(ref_np, dtype=ref_np.dtype)
        return d


class EnsurePhaseKeysd(MapTransform):
    def __init__(self, keys, ref_key="img_c1"):
        super().__init__(keys)
        self.ref_key = ref_key
    def __call__(self, data):
        d = dict(data)
        ref = d.get(self.ref_key, None)
        if ref is None:
            return d
        for k in self.keys:
            if k not in d or d[k] is None:
                z = torch.zeros_like(ref)
                if isinstance(ref, MetaTensor):
                    z = MetaTensor(z, affine=ref.affine, meta=ref.meta.copy())
                d[k] = z
        return d

class FixSize3Dd(MapTransform):
    def __init__(self, keys, spatial_size):
        super().__init__(keys)
        self.sz = tuple(int(v) for v in spatial_size)

    def __call__(self, data):
        d = dict(data)
        Zt, Yt, Xt = self.sz
        for k in self.keys:
            if k not in d: 
                continue
            x = d[k]
            s = x.shape
            has_c = (x.ndim == 4)
            z, y, xw = (s[-3], s[-2], s[-1]) if has_c else (s[-3], s[-2], s[-1])

            def cc(start, size, target):
                if size <= target: return slice(0, size)
                off = (size - target) // 2
                return slice(off, off + target)
            cz, cy, cx = cc(0, z, Zt), cc(0, y, Yt), cc(0, xw, Xt)
            x = x[(slice(None), cz, cy, cx) if has_c else (cz, cy, cx)]

            z, y, xw = x.shape[-3], x.shape[-2], x.shape[-1]
            pz = max(0, Zt - z); py = max(0, Yt - y); px = max(0, Xt - xw)
            if pz or py or px:
                pad = (px//2, px - px//2, py//2, py - py//2, pz//2, pz - pz//2)
                x = F.pad(x, pad)
            d[k] = x
        return d

class EnsureSameShaped(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        shapes = [d[k].shape[-3:] for k in self.keys if k in d]
        if not shapes:
            return d
        target = tuple(min(s[i] for s in shapes) for i in range(3))
        for k in self.keys:
            if k not in d:
                continue
            x = d[k]
            diff = [x.shape[-3+i] - target[i] for i in range(3)]
            sl = tuple(slice(max(0, d // 2), max(0, d // 2) + target[i])
                       if d > 0 else slice(None)
                       for i, d in enumerate(diff))
            d[k] = x[(...,) + sl]
        return d

class WindowTo01d(MapTransform):
    def __init__(self, keys, lo: float, hi: float):
        super().__init__(keys)
        self.lo = float(lo)
        self.hi = float(hi)
        self.den = max(self.hi - self.lo, 1e-8)

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            x = d[k]
            x = (x - self.lo) / self.den
            d[k] = x.clamp(0.0, 1.0)
        return d


class Clamp01d(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            d[k] = d[k].clamp(0.0, 1.0)
        return d

class FixImageShaped(MapTransform):
    def __init__(self, keys: KeysCollection):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            a = d.get(k, None)
            if a is None:
                continue
            a = np.asarray(a)
            a = np.squeeze(a)
            if a.ndim > 3:
                if a.ndim == 4 and a.shape[-1] == 3:
                    a = a.mean(axis=-1)
                else:
                    raise ValueError(f"[FixImageShaped] Unexpected image shape {a.shape}")
            if a.ndim != 3:
                raise ValueError(f"[FixImageShaped] Image must be 3-D after fix, got {a.shape}")
            d[k] = a.astype(np.float32)
        return d


class FixMaskShaped(MapTransform):
    def __init__(self, keys: KeysCollection):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            a = d.get(k, None)
            if a is None:
                continue
            a = np.asarray(a)
            a = np.squeeze(a)
            if a.ndim != 3:
                raise ValueError(f"[FixMaskShaped] Mask '{k}' not 3-D after squeeze: {a.shape}")
            d[k] = (a > 0).astype(np.uint8)
        return d


class UnionMasksd(MapTransform):
    def __init__(self, img_key="img", mask_keys=("mask_C1", "mask_C2", "mask_C3")):
        super().__init__(keys=[img_key] + list(mask_keys))
        self.img_key = img_key
        self.mask_keys = list(mask_keys)

    def __call__(self, data):
        d = dict(data)
        if self.img_key not in d:
            raise ValueError("image key missing before UnionMasksd")
        H, W, D = d[self.img_key].shape[-3:]
        acc = None
        for k in self.mask_keys:
            m = d.get(k, None)
            if m is None:
                continue
            m = np.asarray(m)
            if m.shape != (H, W, D):
                raise ValueError(f"[UnionMasksd] Mask '{k}' not aligned: {m.shape} vs {(H,W,D)}")
            acc = m if acc is None else np.clip(acc + m, 0, 1)
        if acc is None:
            acc = np.zeros((H, W, D), dtype=np.uint8)
        d["lab"] = acc.astype(np.uint8)
        return d


def _common_io_and_geometry(cfg: CFG):
    mask_keys = ["mask_C1", "mask_C2", "mask_C3"]

    return Compose([
        LoadImaged(
            keys=["path_C1", "path_C2", "path_C3", "path_mask_C1", "path_mask_C2", "path_mask_C3"],
            allow_missing_keys=True, image_only=False
        ),

        CopyItemsd(keys=["path_C1"], names=["img_c1"]),
        CopyItemsd(keys=["path_C2"], names=["img_c2"], allow_missing_keys=True),
        CopyItemsd(keys=["path_C3"], names=["img_c3"], allow_missing_keys=True),
        CopyItemsd(keys=["path_mask_C1", "path_mask_C2", "path_mask_C3"], names=mask_keys, allow_missing_keys=True),
        DeleteItemsd(keys=["path_C1", "path_C2", "path_C3", "path_mask_C1", "path_mask_C2", "path_mask_C3"]),

        FixImageShaped(keys=["img_c1", "img_c2", "img_c3"]),
        FixMaskShaped(keys=mask_keys),

        EnsureChannelFirstd(
            keys=["img_c1", "img_c2", "img_c3"] + mask_keys,
            channel_dim="no_channel",
            allow_missing_keys=True
        ),

        EnsureTyped(
            keys=["img_c1", "img_c2", "img_c3"] + mask_keys,
            dtype=[np.float32, np.float32, np.float32] + [np.uint8]*len(mask_keys),
            track_meta=True,
            allow_missing_keys=True,
        ),

        Orientationd(keys=["img_c1", "img_c2", "img_c3"] + mask_keys, axcodes="RAS", allow_missing_keys=True),

        Spacingd(keys=["img_c1", "img_c2", "img_c3"], pixdim=cfg.SPACING, mode=InterpolateMode.BILINEAR, padding_mode="border", allow_missing_keys=True),
        ResampleToMatchd(keys=["img_c2", "img_c3"], key_dst="img_c1", mode=InterpolateMode.BILINEAR, padding_mode="border", allow_missing_keys=True),
        ResampleToMatchd(keys=mask_keys, key_dst="img_c1", mode=InterpolateMode.NEAREST, padding_mode="border", allow_missing_keys=True),

        SqueezeDimd(keys=mask_keys, dim=0, allow_missing_keys=True),

        EnsurePhaseKeysd(keys=["img_c2", "img_c3"], ref_key="img_c1"),

        UnionMasksd(img_key="img_c1", mask_keys=mask_keys),

        MakeROIdFromLabd(lab_key="lab", roi_key="roi", dilate_k=(5,5,5)),

        EnsureSameShaped(keys=["img_c1","lab","roi"]),

        ApplyROIMaskd(img_keys=("img_c1",), lab_key="lab", roi_key="roi"),

        Ensure3DROId(keys=("roi",)),
        EnsureChannelFirstd(keys=["lab", "roi"], channel_dim="no_channel"),

        CropForegroundd(
            keys=["img_c1","img_c2","img_c3","lab","roi"],
            source_key="roi",
            margin=8,
            allow_missing_keys=True
        ),

        SqueezeDimd(keys=["lab", "roi"], dim=0),
    ])


def _to_six_channels_and_types(cfg: CFG):
    def per_phase(phase_name, src_key):
        p = phase_name.lower()
        return [
            CopyItemsd(keys=[src_key], names=[f"{p}_w1", f"{p}_w2", f"{p}_w3"], times=3),
            WindowTo01d(keys=[f"{p}_w1"], lo=cfg.HU_MIN, hi=cfg.HU_MAX),
            WindowTo01d(keys=[f"{p}_w2"], lo=-160.0, hi=240.0),
            WindowTo01d(keys=[f"{p}_w3"], lo=0.0,   hi=200.0),
            CopyItemsd(
                keys=[f"{p}_w1", f"{p}_w2", f"{p}_w3"],
                names=[f"{p}_w1s", f"{p}_w2s", f"{p}_w3s"]
            ),
            GaussianSmoothd(
                keys=[f"{p}_w1s", f"{p}_w2s", f"{p}_w3s"],
                sigma=0.75
            ),
        ]

    phase_to_imgkey = {
        "C1": "img_c1",
        "C2": "img_c2",
        "C3": "img_c3",
    }

    per_phase_blocks = []
    concat_keys = []
    for phase in cfg.PHASES:
        src = phase_to_imgkey[phase]
        p = phase.lower()
        per_phase_blocks += per_phase(phase, src)
        concat_keys += [
            f"{p}_w1", f"{p}_w1s",
            f"{p}_w2", f"{p}_w2s",
            f"{p}_w3", f"{p}_w3s",
        ]

    return Compose(
        per_phase_blocks + [
            ConcatItemsd(keys=concat_keys, name="img", dim=0),
            SelectItemsd(keys=["img", "lab", "roi"]),
            EnsureTyped(keys=["img","lab","roi"],
                        dtype=[np.float32, np.uint8, np.uint8],
                        track_meta=True),
        ]
    )

def make_train_transforms(cfg: CFG):
    return Compose([
        _common_io_and_geometry(cfg),
        _to_six_channels_and_types(cfg),

        Ensure3DROId(keys=("roi",)),
        EnsureChannelFirstd(keys=["lab", "roi"], channel_dim="no_channel"),

        RandCropByPosNegLabeld(
            keys=["img", "lab", "roi"],
            label_key="lab",
            spatial_size=cfg.ROI_SIZE,
            pos=2,
            neg=1,
            num_samples=6,
            image_threshold=0.0,
            allow_smaller=True,
        ),

        SpatialPadd(keys=["img", "lab", "roi"], spatial_size=cfg.ROI_SIZE),
        SqueezeDimd(keys=["lab", "roi"], dim=0),

        EnsureTyped(keys=["img", "lab", "roi"],
                    dtype=[torch.float32, torch.float32, torch.uint8],
                    track_meta=True),

        RandFlipd(keys=["img", "lab", "roi"], prob=0.5, spatial_axis=None),
        RandRotate90d(keys=["img", "lab", "roi"], prob=0.20, max_k=3),
        RandScaleIntensityd(keys=["img"], factors=0.10, prob=0.30),
        RandShiftIntensityd(keys=["img"], offsets=0.10, prob=0.30),
        RandGaussianNoised(keys=["img"], prob=0.10, std=0.01),

        ApplyROIMaskd(img_keys=("img",), lab_key="lab", roi_key="roi"),

        Clamp01d(keys=["img"]),

        FixSize3Dd(keys=["img", "lab", "roi"], spatial_size=cfg.ROI_SIZE),

        EnsureTyped(keys=["img", "lab", "roi"],
                    dtype=[torch.float32, torch.float32, torch.uint8],
                    track_meta=False),
        EnsureChannelFirstd(keys=["img"], channel_dim=0),
        EnsureChannelFirstd(keys=["lab"], channel_dim=0),
        EnsureSameShaped(keys=["img", "lab", "roi"]),

        SelectItemsd(keys=["img", "lab", "roi"]),
    ])

def make_val_transforms(cfg: CFG):
    return Compose([
        _common_io_and_geometry(cfg),
        _to_six_channels_and_types(cfg),

        Ensure3DROId(keys=("roi",)),

        # no cropping / no augments
        ApplyROIMaskd(img_keys=("img",), lab_key="lab", roi_key="roi"),
        SelectItemsd(keys=["img","lab","roi"]),
        EnsureTyped(keys=["img","lab","roi"],
            dtype=[np.float32, np.uint8, np.uint8],
            track_meta=True),
    ])

