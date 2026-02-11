# Decoding phantom limb movements from intraneural recordings 

Code associated with the paper: **"Decoding phantom limb movements from intraneural recordings"**  
Authors: Rossi, C., Bumbasirevic, M., Čvančara, P., Stieglitz, T., Raspopovic, S., Donati, E., & Valle, G. Decoding phantom limb movements from intraneural recordings. Nat Commun (2026). 
https://doi.org/10.1038/s41467-026-69297-0

> Note: If you plan to cite this work, please cite the paper directly (see the published manuscript for the full citation).
> (https://rdcu.be/e3q1S)

## Paper Abstract

Limb loss causes severe sensorimotor deficits and often necessitates prosthetic devices, particularly in lower-limb amputees. Although direct neural recording from residual nerves offersa biomimetic route for prosthetic control, low signal amplitudes and challenges in nerve interfacinghave limited adoption. Intraneural multichannel electrodes provide a potential solution by enablingaccess to motor signals from muscles lost after amputation. Here, we report intraneural recordingsfrom two transfemoral amputees using transversal intrafascicular multichannel electrodesimplanted in distal branches of the sciatic nerve. We identified multiunit activity associated withvolitional phantom movements of the knee, ankle, and toes, exhibiting joint- and direction-specificmodulation distributed across electrodes. A Spiking Neural Network–based decoder outperformedconventional methods in predicting attempted movements, with further gains achieved byintegrating intraneural and intermuscular signals. Motor and sensory maps showed minimaloverlap, indicating early segregation within the sciatic nerve. These findings pave the way forbidirectional, neurally-controlled prosthetic systems.

---

## Repository structure 🔧

- `intraneural_phantom_leg/`
  - `plots/` — plotting scripts and Jupyter notebooks (`decoding_results.py`, `eng_analysis.py`, corresponding `.ipynb` files)
  - `training/` — (placeholder) training scripts (`train_ml.py`, `train_snn.py`)
  - `utils/` — small helper utilities (`utils_functions.py`, etc.)
- `Source Data File.xlsx` — **required** source data (included in this repo)
- `requirements.txt` — requirement file cointaining the project dependencies

**training** and **utils** directories are currently work in progress. They will contain functions useful not only to process, encoding in form of events and decode electroneurographic signals, but also to implement the machine learning classifiers (such as SVM and MLP) and spiking neural network (SNNs) decoders. 

---
## Requirements & setup ⚙️

Recommended: Python 3.8+ and a virtual environment.

PowerShell (Windows) quick setup:
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---
## Usage — reproduce figures 🖼️
1. Place the Source Data File.xlsx at the repository root (or change the file_path variables in the plotting scripts).

    - intraneural_phantom_leg/plots/decoding_results.py expects ../../Source Data File.xlsx when run from intraneural_phantom_leg/plots
    - intraneural_phantom_leg/plots/eng_analysis.py is written assuming it is run from intraneural_phantom_leg (it uses ../Source Data File.xlsx), so either run it from there or update the path accordingly.

2. Run the plotting scripts:
    ```powershell
    # From repository root:
    python ./intraneural_phantom_leg/plots/eng_analysis.py
    python ./intraneural_phantom_leg/plots/decoding_results.py
    # Or run notebooks using:
    jupyter notebook ./intraneural_phantom_leg/plots/eng_analysis.ipynb
    jupyter notebook ./intraneural_phantom_leg/plots/decoding_results.ipynb

This will save the figure files (SVG and image) into the repository root.

---
## 🧑‍💻 Author
Cecilia Rossi
📧 cecilia.m.rossi@gmail.com

🔗 LinkedIn : https://www.linkedin.com/in/cecilia-rossi-2930b8291/
 | GitHub: https://github.com/rossicecilia
