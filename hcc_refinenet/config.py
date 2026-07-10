from dataclasses import dataclass
import os

@dataclass
class CFG:
    
    SEED: int = 2025

    IMG_ROOT: str = r"D:\HCC\CECT"
    PHASE_CSV: str = r"D:\Downloads\patient_data_unzipped_nii.csv"
    CSV_PATH: str = r"D:\HCC\TumorDetection\patient_tumor.csv"
    OUT_DIR: str = r"D:\HCC\TumorDetection\patient_tumor_out"

    PHASES = ["C1", "C2", "C3"]
    CLASSES = ["HCC", "ICC", "CHCC"]
    
    @property
    def IN_CHANNELS(self):
        return 6 * len(self.PHASES)
    
    N_FOLDS: int = 5
    FOLD_IDX: int = 2
    MAX_EPOCHS: int = 150
    BATCH_SIZE: int = 1
    NUM_WORKERS: int = 4 #4
    LR: float = 1e-4
    WEIGHT_DECAY: float = 1e-5

    SPACING = (1.5, 1.5, 1.5)
    ROI_SIZE = (96, 96, 64)
    SW_OVERLAP: float = 0.5

    HU_MIN: int = -100
    HU_MAX: int = 400

    USE_FOCAL: bool = True
    FOCAL_ALPHA: float = 0.5
    FOCAL_GAMMA: float = 2.0
    DICE_WEIGHT: float = 1.0
    FOCAL_WEIGHT: float = 1.0

    BRIGHT_GAIN: float = 0.5

    def ckpt_dir(self):
        return os.path.join(self.OUT_DIR, "checkpoints")

    def fold_dir(self, k):
        return os.path.join(self.OUT_DIR, f"fold{k}")

os.makedirs(CFG().OUT_DIR, exist_ok=True)
os.makedirs(CFG().ckpt_dir(), exist_ok=True)

