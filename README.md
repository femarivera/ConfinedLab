<p align="center">
  <img src="assets/logo_confinedlab.png" height="100">
</p>

# ConfinedLab

Model files for the research work entitled *"Revisiting hydraulic response times in regional groundwater systems: the role of system connectivity and stress magnitude"*

As part of the [PEPR One Water DEESAC project](https://www.onewater.fr/fr/actualite/actualite/lancement-du-projet-deesac-durabilite-exploitabilite-des-eaux-souterraines-des "Go to onewater.fr"), its primary goal is to investigate the transient response of multilayer aquifer systems to external climatic and anthropogenic forcings using synthetic numerical models, with a focus on confined aquifers within regional multilayer groundwater systems. We aim to assess the implications of response times on past and future system behaviour to inform sustainability assessments.

> 🛠️ This project uses [ConfinedLab-mlibs](https://github.com/femarivera/ConfinedLab-mlibs) as its utility library.

---

## Repository structure

```
ConfinedLab/
├── templates/
│   ├── 2D/          ← MODFLOW 6 model file templates for 2D experiments
│   └── 3D/          ← MODFLOW 6 model file templates for 3D experiments
├── docs/            ← documentation for the model files
├── gis/             ← GIS files (shapefiles, rasters, QGIS projects)
├── assets/          ← logo and figure files used in the README
├── mf/              ← simulation outputs
├── gwmodelling.yaml           ← full conda environment specification
├── gwmodelling-portable.yaml  ← portable conda environment specification
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## Installation

### 1. Create and activate the conda environment

```bash
conda env create -f gwmodelling.yaml
conda activate gwmodelling
```

This includes necessary dependencies and software, such as MODFLOW 6.

### 2. Clone the repository and install

```bash
git clone https://github.com/femarivera/ConfinedLab.git
cd ConfinedLab
pip install -e .
```

This will automatically install all Python dependencies, including [ConfinedLab-mlibs](https://github.com/femarivera/ConfinedLab-mlibs).

---

## License

This project is licensed under the BSD 3-Clause License — see the [LICENSE](LICENSE) file for details.

---

# Response Times Estimation Framework

A framework for building, running, and post-processing steady-state and transient MODFLOW 6 groundwater models to estimate hydraulic response times, capture rates, and sustainable yields.

## Structure

Inside each of the `runs/Response time experiments` subfolders you will find:

| File | Description |
|---|---|
| `Model.py` | Runs a MODFLOW 6 simulation |
| `Run_parallel.py` | Launches parallel runs for different parameter sets |
| `setup.xlsx` | Contains the variables and parameters of the model |
| `Postprocessing response time.py` | Postprocessing and visualization of the results |

`Model.py` is tied to `setup.xlsx`, which defines all inputs needed to build and run the steady-state and transient MODFLOW 6 model.

## `setup.xlsx` sheet reference

| Sheet | Description |
|---|---|
| `Grid` | Definition of the structured grid parameters. |
| `Geometry` | Parameters for constructing the synthetic model geometry, using the `modgeom6` module functions. |
| `Tdis` | Time discretization. |
| `Parameters` | Model parameters per zone. |
| `Transient_recharge` | Time series of recharge rates per zone. |
| `Observations` | Cell IDs of the observation points. |
| `Well_st` | Well locations and pumping rates for the steady-state simulation. |
| `Wells` | Well locations and pumping rate time series for the transient simulation. |
| `Response_times` | Parameters for response time estimation via the `modtransient6` function. **Note:** the steady-state recharge and pumping rates here must match those defined for the last stress period of the transient series. |
| `Q_values_st` | Pumping rates per well for sequential steady-state runs, used to investigate capture rates and sustainable yields via the `modpump6` module functions. |
| `Q_values_tr` | Pumping rates per well for sequential transient runs, used to investigate capture rates and sustainable yields via the `modpump6` module functions. |
| `Parameter_analysis` | Parameter sets for sensitivity analysis, used to launch parallel runs. |

## Usage

From a subfolder within `Runs/Response time experiments`

### 1. Run a single model

```bash
python Model.py
```

This builds and runs the model for the current parameter set and estimates response times.

### 2. Run a parameter senstivity analysis in parallel

```bash
python Run_parallel.py
```

This launches parallel runs across all parameter sets defined in `setup.xlsx - Parameter_analysis`. A folder is created per parameter set, containing the simulation results and the response time estimation for that run.

### 3. Post-process and visualize results

Then run:

```bash
python "Postprocessing response time.py"
```

This generates summary plots and a `tr_analysis.csv` file summarizing the results across all parameter sets.
Within each experiment folder, all setup files and scripts have their respective parameters and variables set to reproduce the results presented in the manuscript.


## Contact

For questions, suggestions, or contributions, please contact:

**Carlos Felipe Marin Rivera**  
Bordeaux INP, UMR 5805 Lab EPOC, Université de Bordeaux  
cmarinriver@bordeaux-inp.fr

<p float="left">
  <img src="assets/logo_ensegid.jpg" height="50" style="margin-right:10px;" />
  <img src="assets/logo_epoc.png" height="50" style="margin-right:10px;" />
  <img src="assets/logo_ubordeaux.png" height="50" />
</p>