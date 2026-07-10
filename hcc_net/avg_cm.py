import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = r"D:\HCC\metrics\effb1_cv"
CLASSES = ["HCC", "ICC", "CHCC"]
N_FOLDS = 5
SAVE_PATH = "avg_confusion_matrix.png"

cms = []

for fold_idx in range(N_FOLDS):
    mpath = os.path.join(BASE_DIR, f"fold{fold_idx}", "metrics.json")

    if not os.path.exists(mpath):
        raise FileNotFoundError(f"Missing: {mpath}")

    with open(mpath, "r", encoding="utf-8") as f:
        data = json.load(f)

    cm = np.array(data["confusion_matrix"], dtype=float)
    cms.append(cm)

cms = np.stack(cms, axis=0)
cm_mean = cms.mean(axis=0)
cm_rounded = np.rint(cm_mean).astype(int)

plt.figure(figsize=(7, 6))

sns.heatmap(
    cm_rounded,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=True,
    square=True,
    linewidths=0.5,
    linecolor="white"
)

plt.title("Averaged Confusion Matrix (5-fold CV)", fontsize=16)
plt.ylabel("True Label", fontsize=14)
plt.xlabel("Predicted Label", fontsize=14)
plt.xticks(np.arange(len(CLASSES)) + 0.5, CLASSES, rotation=45, ha='right')
plt.yticks(np.arange(len(CLASSES)) + 0.5, CLASSES, rotation=0)

plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved averaged confusion matrix to: {SAVE_PATH}")

