# Deep Learning for Multiphasic CT-Based Liver Cancer Subtype Classification and Segmentation Refinement

This repository contains the implementation of two complementary deep learning models for multiphasic contrast-enhanced CT (CECT) analysis of primary liver cancer:

- HCC-Net: a phase-stacked EfficientNet-V2-S classifier for three-way subtype classification (HCC, ICC, cHCC-iCCA)
- HCC-RefineNet: a 3D U-Net for post-hoc tumor boundary refinement using tumor-centered regions of interest

The two models are applied independently to the same study and are organized as two subfolders in this repository.

## Repository Structure

```
project_root/
  hcc_net/                 HCC-Net classification pipeline
  hcc_refinenet/           HCC-RefineNet segmentation pipeline
  hcc_net_result/          HCC-Net classification results
  hcc_refinenet_result/    HCC-RefineNet segmentation results
  README.md                This file
```

## Requirements

Python 3.9 or later is recommended. Each subfolder has its own `requirements.txt`.

```
pip install -r hcc_net/requirements.txt
pip install -r hcc_refinenet/requirements.txt
```

## HCC-Net (Classification)

### Configuration

All paths and hyperparameters are set in `hcc_net/config.py`. Update the following fields to match your environment:

```
IMG_ROOT      Root directory containing the CECT data
PHASE_CSV     Original per-phase CSV exported from the raw dataset
CSV_PATH      Patient-level CSV built by prepare_df.py
CKPT_DIR      Directory where fold checkpoints are saved
BEST_CKPT     Path to the checkpoint used for evaluation and prediction
FOLD_IDX      Which fold (0 to 4) the current run trains or evaluates
```

### Data Preparation

```
cd hcc_net
python prepare_df.py
python check_leakage_phases.py
python check_path_leakage.py
python sanity_checks.py
```

`prepare_df.py` produces the CSV pointed to by `CFG.CSV_PATH`, with one row per patient and one column per phase path (`path_C1`, `path_C2`, `path_C3`, `path_P`). The remaining scripts are optional integrity checks.

### Training

Set `CFG.FOLD_IDX` in `config.py` to the fold you want to train (0 to 4), then run:

```
python train.py
```

Repeat for all five folds to reproduce the full cross-validation result.

### Evaluation

```
python eval_metrics.py --ckpt path/to/checkpoint.pt --csv path/to/eval.csv --outdir results/
python eval_metrics_cv.py
python avg_cm.py
```

`eval_metrics_cv.py` aggregates out-of-fold predictions and metrics across all five folds, producing the confusion matrix, ROC curves, precision-recall curves, and metrics reported in the paper.

### External Validation

```
python prepare_df_external.py
python predict_external_ensemble.py
python analyze_external_predictions.py
```

### Prediction on New Data

```
python predict.py
```

Update the input CSV path and checkpoint path in `config.py` or directly in the script before running.

## HCC-RefineNet (Segmentation Refinement)

### Configuration

All paths and hyperparameters are set in `hcc_refinenet/config.py`. Update the following fields to match your environment:

```
IMG_ROOT      Root directory containing the CECT data
PHASE_CSV     Original per-phase CSV exported from the raw dataset
CSV_PATH      Patient-level tumor CSV built by prepare_df.py
OUT_DIR       Output directory for exported predictions and masks
PHASES        Contrast phases used as input (default C1, C2, C3)
FOLD_IDX      Which fold (0 to 4) the current run trains or evaluates
```

### Data Preparation

```
cd hcc_refinenet
python unzip_nii_gz.py
python prepare_df.py
python quick_check_paths.py
```

Skip `unzip_nii_gz.py` if the raw dataset is not distributed as compressed archives.

### Training

Set `CFG.FOLD_IDX` in `config.py` to the fold you want to train (0 to 4), then run:

```
python train.py
```

The best checkpoint for that fold, selected by validation Dice, is saved to the directory configured in `config.py`.

### Evaluation

```
python evaluate.py --fold 0 --thresh 0.65
python evaluate.py --ensemble_folds 0 1 2 3 4 --thresh 0.65
python evaluate.py --ens_all_patients --ens_folds 0 1 2 3 4 --thresh 0.65
python oof_eval.py
python merge_oof.py
```

`evaluate.py --fold` evaluates a single fold. `--ensemble_folds` evaluates an ensemble of folds on the validation split defined by the first fold listed. `--ens_all_patients` evaluates an ensemble across the full patient set. `oof_eval.py` and `merge_oof.py` aggregate out-of-fold results across all five folds.

### Threshold Selection

The segmentation threshold used in the paper (T = 0.65) was selected from a sweep over the range 0.5 to 0.95 on the validation set. To reproduce this sweep and plot:

```
python plot_threshold.py
```

### Prediction

```
python predict.py
```

By default this uses all five folds. Edit the `folds` argument in the `__main__` block to use a subset.

## Notes

`config.py` is the single source of truth for experiment settings in both pipelines; command-line arguments are only used in `eval_metrics.py` (HCC-Net) and `evaluate.py` (HCC-RefineNet). The middle-slice assumption, phase-stacking strategy, ROI construction, and HU windowing described in the paper are implemented in the respective `dataset.py` and `transforms.py` files.

## How to Cite

If you found our work useful, please cite us. For the HCC-Net classification method and the HCC-RefineNet segmentation method, please cite:

Pham Khoi Nguyen, Tran Ngoc Thao Vy, Nguyen Thi Kim Phung. "Deep Learning for Multiphasic CT-Based Liver Cancer Subtype Classification and Segmentation Refinement." MAPR, 2026. doi: xxx.

```bibtex
@inproceedings{pham2026liverct,
  title     = {Deep Learning for Multiphasic CT-Based Liver Cancer Subtype Classification and Segmentation Refinement},
  author    = {Pham, Khoi Nguyen and Tran, Ngoc Thao Vy and Nguyen, Thi Kim Phung},
  booktitle = {Proceedings of the International Conference on Multimedia Analysis and Pattern Recognition (MAPR)},
  year      = {2026},
  note      = {to appear}
}
```