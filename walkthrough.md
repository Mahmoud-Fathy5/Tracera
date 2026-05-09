# Walkthrough — Retrain Notebook Enhancement & Deployment Readiness

## What Changed

### 1. Rebuilt `retrain_notebook.ipynb` (31 cells)

The notebook was completely rebuilt with these new sections:

#### Image Scraping (Cells 2-3)
- **22 real-world categories** scraped via Bing Image Downloader (300 images each ≈ 6,600 target):
  - Phone selfies, portrait mode, mirror selfies
  - Social media: WhatsApp, Instagram, Snapchat, Facebook compressed photos
  - Brand-specific: iPhone, Samsung Galaxy, Pixel camera photos
  - Life events: birthday, wedding, graduation, baby photos
  - Everyday: food, pets, group photos, travel, street photography
  - Edge cases: screenshots, low-quality/blurry, memes
- **8 fake image categories** + 200 from thispersondoesnotexist.com

#### Custom Image Interface (Cells 4-5)
- `add_custom_images(source_paths, count=None, label='real'|'fake')` — clean function to add your own images
- Accepts single path or list of paths, with optional count limit
- Heavy phone-simulation augmentation (`augment_phone_heavy`):
  - JPEG compression (quality 15-95, simulates WhatsApp forwarding)
  - Resize down→up (phone scaling artifacts)
  - Color jitter (brightness, contrast, saturation)
  - Gaussian blur (cheap lens) + sharpening (phone post-processing)
  - Vignetting (Instagram-style filter)
  - Portrait-mode simulation (edge blur, center sharp)
  - Horizontal flip + sensor noise injection
- 3 extraction passes per image set: heavy augmented, light augmented, clean

#### k=8 Eigenvalues (Cells 7-9)
- Changed `TOP_K_EIGENVALUES` from 16 → **8**
- Built a remapping function that converts the old k=16 cached features to k=8 layout:
  - Slices out eigenvalue positions 9-16 from each layer
  - Recomputes inter-layer cosine similarities on 8-dim profiles
- New VGG extractor uses k=8 directly for all new images

#### Bug Fixes
| Bug | Fix |
|---|---|
| Double-counting in train/val split | Clean 80/20 split done once, no duplicate data |
| `early_stopping_rounds` in constructor | Moved to `.fit()` call |
| Class imbalance ignored | Added `scale_pos_weight` computed from real/fake ratio |
| Placeholder `NEW_REAL_DIR` crashes | Scraping cells auto-populate the directories |

#### New Analysis Cells (Cells 14-17)
- Confusion matrix (side-by-side: default 0.5 vs optimal threshold)
- Threshold vs Precision/Recall/F1 curve — visual guide for choosing deployment threshold
- Training/Validation AUC curves
- Top 30 feature importance bar chart

---

### 2. New `threshold_config.json`

```json
{
  "threshold": 0.50,
  "threshold_mode": "auto",
  "notes": "Set threshold_mode to 'manual' to override..."
}
```

To adjust detection sensitivity without retraining:
1. Edit `threshold_config.json`
2. Set `"threshold_mode": "manual"`
3. Set `"threshold": 0.45` (or whatever value you want)
4. Restart the app — done

---

### 3. Updated `inference.py`

render_diffs(file:///d:/web%20v2%20(1)/web%20v2/inference.py)

Added threshold_config.json reader in `GramNetDetector.__init__()`. When `threshold_mode` is `"manual"`, the JSON value overrides the model's trained threshold. Gracefully handles missing/malformed config.

---

## Files Modified/Created

| File | Action |
|---|---|
| [retrain_notebook.ipynb](file:///d:/web%20v2%20(1)/web%20v2/retrain_notebook.ipynb) | Rebuilt (31 cells) |
| [threshold_config.json](file:///d:/web%20v2%20(1)/web%20v2/threshold_config.json) | New |
| [inference.py](file:///d:/web%20v2%20(1)/web%20v2/inference.py) | Modified (threshold override) |
| [model/retrain_notebook.ipynb](file:///d:/web%20v2%20(1)/model/retrain_notebook.ipynb) | Copy |

## Deployment Workflow

1. Upload notebook to Kaggle, attach your cached features zip as input
2. Optionally add your own image datasets and uncomment `add_custom_images()` calls in Cell 5
3. Run all cells — scraping + extraction + training takes ~30-60 min on T4
4. Download output files: `xgb_detector_k8_retrained.json`, `norm_stats_v3_retrained.pt`, `config_detector_k8_retrained.json`, `threshold_config.json`
5. Copy to `web v2/model/` directory
6. Update inference.py model file paths (or rename files to match existing names)
7. Tune threshold via `threshold_config.json` as needed
