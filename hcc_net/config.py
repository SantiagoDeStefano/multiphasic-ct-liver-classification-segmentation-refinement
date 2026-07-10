from dataclasses import dataclass
import os

@dataclass
class CFG:
    IMG_ROOT: str = r"D:\HCC"

    SEED: int = 2025

    CLASSES = ["HCC", "ICC", "CHCC"]
    NUM_CLASSES: int = 3

    PHASES = ["C1", "C2", "C3", "P"]
    N_PHASES: int = len(PHASES)

    SLICES_PER_PHASE: int = 1

    USE_LIVER_MASK: bool = False
    ADD_MASK_AS_CHANNEL: bool = False
    CROP_TO_LIVER: bool = False

    IMG_SIZE: int = 384
    BATCH_SIZE: int = 12
    EPOCHS: int = 30
    LR: float = 2e-4
    WEIGHT_DECAY: float = 1e-4
    LABEL_SMOOTHING: float = 0.05
    USE_FOCAL: bool = True
    FOCAL_GAMMA: float = 2.0
    MIXED_PRECISION: bool = True
    EARLY_STOP_PATIENCE: int = 6
    MODEL_NAME: str = "efficientnetv2_rw_s"
    FREEZE_BACKBONE_EPOCHS: int = 1
    COSINE_TMAX: int = 10
    NUM_WORKERS: int = 4

    N_FOLDS: int = 5
    FOLD_IDX: int = 3 # 0 1 2 3 4

    PHASE_CSV: str = r"D:\Downloads\patient_data.csv"
    CSV_PATH: str = r"D:\HCC\patient_rows.csv"
    CKPT_DIR: str = r"D:\HCC\checkpoints"
    BEST_CKPT: str = os.path.join(rf"D:\HCC\Naive_Model\{MODEL_NAME}", f"best_fold{FOLD_IDX}.pt")

    WINDOW_CENTER: int = 150
    WINDOW_WIDTH: int = 300

    HFLIP_P: float = 0.5
    VFLIP_P: float = 0.0
    ROT_P: float = 0.35
    ROT_DEG: int = 10
    SHIFT_SCALE_ROT_P: float = 0.35
    GRID_DISTORT_P: float = 0.15

