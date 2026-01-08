# Decoding phantom limb movements from intraneural recordings 

Code associated with the paper: **"Decoding phantom limb movements from intraneural recordings"**  
Authors: Rossi, C., Bumbasirevic, M., Čvančara, P., Stieglitz, T., Raspopovic, S., Donati, E., & Valle, G. (2025). Decoding phantom
limb movements from intraneural recordings. Preprint medRxiv, 2025-08. (submitted to Nature Communication,
currently under review)

> Note: If you plan to cite this work, please cite the paper directly (see the published manuscript for the full citation).
https://www.medrxiv.org/content/10.1101/2025.08.21.25333903v1.full.pdf

## Paper Abstract

Limb loss leads to severe sensorimotor deficits and requires the use of a prosthetic device, especially in lower limb amputees. While direct recording from residual nerves offers a biomimetic route for an effective prosthetic control, the low amplitude and noisy nature of these neural signals together with the challenge of establishing a reliable nerve interfacing , have hindered its adoption. Intraneural multichannel electrodes could potentially establish an effective interface with the nerve fibers, enabling access to motor signals even from muscles lost after the amputation. In this study, we report the direct neural recordings of two transfemoral amputees using transversal intrafascicular multichannel electrodes (TIME) implanted in the distal branch of the sciatic nerves.
We observed multiunit activity associated with volitional phantom movements of the knee, ankle and toes flexion and extension, with joint and direction specific neural modulation in both participants. The motor signals were distributed across all the electrodes, showing both single-joint and multi-joint selectivity, as well as direction selectivity for limb flexion and extension. After characterizing the neural evoked activity, we developed a Spiking Neural Network (SNN)-based decoder that outperform conventional motor decoders in predicting attempted phantom leg movements. Decoding accuracy improved further by including a broader signal bandwidth that captured both intraneural (ENG) and inter-muscular (imEMG) activity. Finally, comparing motor maps (recording) with sensory maps (stimulation) revealed a minimal overlap, suggesting early segregation of motor and sensory fibers within the sciatic nerve before the knee bifurcation. Our findings demonstrate the feasibility to record motor signal and decode lower-limb movements directly from the nerves in amputees using intraneural interfaces. This provides preliminary validation of motor decoding feasibility for bidirectional, neurally-controlled prosthetic limbs combining natural control with somatosensory feedback through a single implanted interface.

---

## Repository structure 🔧

- `intraneural_phantom_leg/`
  - `plots/` — plotting scripts and Jupyter notebooks (`decoding_results.py`, `eng_analysis.py`, corresponding `.ipynb` files)
  - `training/` — (placeholder) training scripts (`train_ml.py`, `train_snn.py`)
  - `utils/` — small helper utilities (`utils_functions.py`, etc.)
- `Source Data File.xlsx` — **required** source data (included in this repo)
- `requirements.txt` — requirement file cointaining the project dependencies

training and utils directories are currently work in progress. They will contain functions useful not only to process, encoding in form of events and decode electroneurographic signals, but also to implement the machine learning classifiers (such as SVM and MLP) and spiking neural network (SNNs) decoders. 
---
## Requirements & setup ⚙️

Recommended: Python 3.8+ and a virtual environment.

PowerShell (Windows) quick setup:
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

---
## Usage — reproduce figures 🖼️
1. Place the Source Data File.xlsx at the repository root (or change the file_path variables in the plotting scripts).

    - intraneural_phantom_leg/plots/decoding_results.py expects ../../Source Data File.xlsx when run from intraneural_phantom_leg/plots
    intraneural_phantom_leg/plots/eng_analysis.py is written assuming it is run from intraneural_phantom_leg (it uses ../Source Data File.xlsx), so either run it from there or update the path accordingly.

2. Run the plotting scripts:
    ```powershell
    # From repository root:
    python [decoding_results.py](http://_vscodecontentref_/9)
    # Or run notebooks using:
    jupyter notebook [decoding_results.ipynb](http://_vscodecontentref_/10)

This will save the figure files (SVG and image) into your current working directory.

---
## 🧑‍💻 Author
Cecilia Rossi
📧 cecilia.m.rossi@gmail.com

🔗 LinkedIn : https://www.linkedin.com/in/cecilia-rossi-2930b8291/
 | GitHub: https://github.com/rossicecilia
