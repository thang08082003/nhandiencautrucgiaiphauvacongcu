# STATUS (Claude Code, 2026-10-07)

All experiments are finished. Numbers: `results/REPORT_DATA.md` (regenerated 2026-10-07 15:49 UTC by `report_tables.py`). Code on Drive `src/` (GitHub will be updated when the user says all work is done).

## nnU-Net v2 3d_fullres (CT, 6 test cases, evaluated on the original grid)
- Trainer `nnUNetTrainer_20epochs` (20 epochs instead of 1000, because of the GPU quota). Epochs 0-8 on Colab T4 (~374 s/epoch), epochs 9-19 on Kaggle T4 (resumed with `--c`, ~234 s/epoch). Inference uses `checkpoint_best.pth`. Mean validation Dice (nnU-Net) 0.860.
- Test: Dice 0.795, HD95 60.8 mm (`results/ct/nnunet_metrics.csv`). Per label: spleen 0.933, kidney_right 0.840, kidney_left 0.830, gallbladder 0.536, liver 0.948, pancreas 0.683.
- Dataset built from `cache/ct/*.npy` (same 1.5 mm input as the custom 3D U-Net, no AMOS re-download). Splits are identical to data_ct.py (splits_final.json).

## Post-processing (largest connected component per label, `src/postprocess.py`)
- Custom 3D U-Net: Dice 0.770 -> 0.769, HD95 33.8 -> 12.7 mm (`results/ct/pred_custom_pp_metrics.csv`).
- nnU-Net: Dice 0.795 -> 0.794, HD95 60.8 -> 11.7 mm (`results/ct/pred_nnunet_pp_metrics.csv`); spleen 0.939, kidney_r 0.848, kidney_l 0.837, gallbladder 0.554, liver 0.957, pancreas 0.634 (pancreas drops: fragmented organ).
- CAMUS skipped: 2D predictions are not saved to disk; 2D models already output one blob per structure.

## CholecSeg8k: 4 extra runs (30-min budget each, Kaggle T4, ~50-57 s/epoch, comparable to Colab's 55-62 s/epoch)
Test mean Dice / NSD (6 classes present in test):
- aug_focal (strong aug + Dice+Focal, seed 42): 0.856 / 0.735, 32 epochs
- base_s1 (baseline config, seed 1): 0.846 / 0.703, 37 epochs
- aug_strong_s1 (seed 1): 0.854 / 0.723, 37 epochs
- aug_focal_s1 (seed 1): 0.852 / 0.713, 32 epochs
Results in `results/cholecseg8k/ablation/<tag>/`; `analyze_2d.py ablation` was rerun for cholec and camus.
Note: Kaggle T4 vs Colab T4 for these 4 runs; seed variation (42 vs 1) is about 0.01 Dice, so ablation differences below that are within noise.

## Not done / caveats
- Only 20 nnU-Net epochs: nnU-Net is under-trained, its HD95 before post-processing is dominated by small false-positive islands.
- Single seed for everything except the CholecSeg8k seed-1 replicates above.
