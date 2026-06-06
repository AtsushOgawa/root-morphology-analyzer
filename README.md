# root_analysis.py

**Automatic measurement of root morphological parameters from flatbed scanner images**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)

---

## Overview

`root_analysis.py` is a free, open-source Python script that automatically measures five root morphological parameters from flatbed scanner images:

| Parameter | Unit |
|---|---|
| Total root length | mm |
| Root tip count | — |
| Root surface area | cm² |
| Mean root diameter | mm |
| Root volume | cm³ |

The script automatically detects whether roots are **stained** (dark roots on a light background) or **unstained** (light roots on a dark background) from mean image brightness, and applies the appropriate processing parameters without any manual configuration.

This script was developed as a free alternative to the commercial software WinRhizo (Regent Instruments) and was validated against WinRhizo measurements on 99 images (50 unstained lettuce, 49 stained rice). Pearson correlation coefficients exceeded 0.96 for all five parameters. See the [Citation](#citation) section for the reference paper.

---

## Requirements

- Python 3.8 or higher
- The following Python packages:

```
numpy
Pillow
scikit-image
scipy
```

Install all dependencies with:

```bash
pip install numpy Pillow scikit-image scipy
```

---

## Supported image formats

`.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`, `.bmp`

- 8-bit or 16-bit grayscale/color images
- Resolution: automatically read from image metadata (default: 1200 dpi if not available)
- Recommended scan resolution: 600–1200 dpi

---

## Usage

### 1. Prepare your images

Place all root images in a folder named **根画像** (or edit `IMAGE_DIR` in the script) on your Desktop:

```
Desktop/
└── 根画像/
    ├── sample_01.tif
    ├── sample_02.tif
    └── ...
```

Subfolders are also scanned automatically.

### 2. Run the script

```bash
python root_analysis.py
```

### 3. Check the output

Results are saved to `Desktop/根画像/result.csv` with the following columns:

| Column | Description |
|---|---|
| `ファイル名` | Image filename |
| `根の種類` | Detected type (stained / unstained) |
| `解像度(DPI)` | Image resolution |
| `Otsu閾値` | Otsu binarization threshold |
| `総根長 (mm)` | Total root length |
| `根端数` | Root tip count |
| `根表面積 (cm²)` | Root surface area |
| `根平均直径 (mm)` | Mean root diameter |
| `根の体積 (cm³)` | Root volume |

---

## How it works

```
Input image
    │
    ▼
Grayscale conversion (16-bit → 8-bit normalization if needed)
    │
    ▼
Auto-detection of staining status (mean brightness threshold: 128)
    ├─ Bright < 128 → Unstained (light roots, dark background)
    └─ Bright ≥ 128 → Stained   (dark roots, light background)
    │
    ▼
Binarization (Otsu's method)
  + Gaussian smoothing σ=1 for stained images only
    │
    ▼
Noise removal (< 500 px connected components removed)
    │
    ▼
Skeletonization → Pruning
  (pruning threshold: 0.127 mm unstained / 0.05 mm stained)
    │
    ▼
Distance transform → Local diameter estimation
    │
    ▼
Parameter calculation with correction coefficients
    │
    ▼
Output: result.csv
```

---

## Parameters and calibration

The correction coefficients were calibrated using 8 unstained (lettuce) and 8 stained (rice) images against WinRhizo (version 2009b, Automatic setting).

| Parameter | Unstained | Stained |
|---|---|---|
| Length correction coefficient | 1.0775 | 1.1491 |
| Diameter correction coefficient | 1.0287 | 1.1845 |
| Pruning threshold (mm) | 0.127 | 0.05 |

> **Note:** If you use a different scanner, resolution, or plant species, recalibration against your reference measurements is recommended. Edit the correction coefficients in the `# ★ Parameter settings` section of the script.

---

## Validation results

Validated on 99 images (50 unstained lettuce + 49 stained rice):

| Parameter | Pearson *r* | Mean relative error | Within ±20% | *t*-test |
|---|---|---|---|---|
| Total root length | 0.9998 | +0.0% ± 1.2% | 100% | n.s. |
| Root tip count | 0.9930 | +4.4% ± 9.8% | 89% | ** |
| Root surface area | 0.9944 | −0.0% ± 6.3% | 98% | n.s. |
| Mean root diameter | 0.9587 | +0.0% ± 6.6% | 98% | n.s. |
| Root volume | 0.9816 | +0.3% ± 12.3% | 92% | n.s. |

\*\* p < 0.01; n.s. not significant (paired t-test, α = 0.05)

---

## Citation

If you use this script in your research, please cite the following paper:

> Ogawa, A. (2025). Development and validation of a Python script for automatic measurement of root morphological parameters using flatbed scanner images in comparison with WinRhizo. *Plant Root*, XX: XX–XX. https://doi.org/10.5281/zenodo.20571517

BibTeX:
```bibtex
@article{ogawa2025root,
  author  = {Ogawa, Atsushi},
  title   = {Development and validation of a {Python} script for automatic 
             measurement of root morphological parameters using flatbed 
             scanner images in comparison with {WinRhizo}},
  journal = {Plant Root},
  year    = {2025},
  volume  = {XX},
  pages   = {XX--XX},
  doi     = {10.5281/zenodo.20571517}
}
```

> **Note:** The DOI and volume/page numbers will be updated upon publication.

---

## Development note

This script was developed by a researcher (Atsushi Ogawa, Akita Prefectural University) without programming expertise, through iterative dialogue with a generative AI (Anthropic Claude). The agricultural domain knowledge for calibration, validation, and parameter tuning was provided by the author; the AI assisted with code generation, debugging, and statistical analysis.

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

You are free to use, modify, and distribute this script for any purpose, including commercial use, with attribution.

---

## Author

**Atsushi Ogawa**  
Faculty of Bioresource Sciences, Akita Prefectural University, Japan  

---

## Acknowledgements

WinRhizo is a registered trademark of Regent Instruments Inc. This script is an independent implementation and is not affiliated with or endorsed by Regent Instruments.
