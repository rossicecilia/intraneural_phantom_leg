# Decoding phantom limb movements from intraneural recordings ✅

Code associated with the paper: **"Decoding phantom limb movements from intraneural recordings"**  
Authors: Rossi, C., Bumbasirevic, M., Čvančara, P., Stieglitz, T., Raspopovic, S., Donati, E., & Valle, G. (2025). Decoding phantom
limb movements from intraneural recordings. Preprint medRxiv, 2025-08. (submitted to Nature Communication,
currently under review)

> Note: If you plan to cite this work, please cite the paper directly (see the published manuscript for the full citation).
https://www.medrxiv.org/content/10.1101/2025.08.21.25333903v1.full.pdf

---

## Repository structure 🔧

- `intraneural_phantom_leg/`
  - `plots/` — plotting scripts and Jupyter notebooks (`decoding_results.py`, `eng_analysis.py`, corresponding `.ipynb` files)
  - `training/` — (placeholder) training scripts (`train_ml.py`, `train_snn.py`)
  - `utils/` — small helper utilities (`utils_functions.py`, etc.)
- `Source Data File.xlsx` — **required** source data (not included in this repo)

---

## Requirements & setup ⚙️

Recommended: Python 3.8+ and a virtual environment.

PowerShell (Windows) quick setup:
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

---

## Usage — reproduce figures 🖼️
1. Place the Source Data File.xlsx at the repository root (or change the file_path variables in the plotting scripts).

    - intraneural_phantom_leg/plots/decoding_results.py expects ../../Source Data File.xlsx when run from intraneural_phantom_leg/plots
    intraneural_phantom_leg/plots/eng_analysis.py is written assuming it is run from intraneural_phantom_leg (it uses ../Source Data File.xlsx), so either run it from there or update the path accordingly.

2. Run the plotting scripts:

    # From repository root:
    python [decoding_results.py](http://_vscodecontentref_/9)
    # Or run notebooks using:
    jupyter notebook [decoding_results.ipynb](http://_vscodecontentref_/10)

This will save the figure files (SVG and image) into your current working directory.

---
