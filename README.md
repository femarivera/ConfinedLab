<p align="center">
  <img src="assets/logo_confinedlab.png" height="100">
</p>

# ConfinedLab

Model files for the research work entitled *"Revisiting hydraulic response times in regional groundwater systems: the role of system connectivity and stress magnitude"*

As part of the [PEPR One Water DEESAC project](https://www.onewater.fr/fr/actualite/actualite/lancement-du-projet-deesac-durabilite-exploitabilite-des-eaux-souterraines-des "Go to onewater.fr"), its primary goal is to investigate the transient response of multilayer aquifer systems to external climatic and anthropogenic forcings using synthetic numerical models, with a focus on confined aquifers within regional multilayer groundwater systems. We aim to assess the implications of response times on past and future system behaviour to inform sustainability assessments.

> 🛠️ This project uses [mlibs](https://github.com/femarivera/mlibs) (v0.1.1) as its utility library.

---

## Repository structure

```
ConfinedLab/
├── runs/
│   └── Response time experiments/   ← one folder per experiment
├── templates/
│   ├── 2D/          ← model script templates for 2D experiments
│   └── 3D/          ← model script templates for 3D experiments
├── gis/             ← GIS files (shapefiles, QGIS project)
├── assets/          ← logo and figure files used in the README
├── gwmodelling.yml  ← conda environment specification
├── LICENSE
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/femarivera/ConfinedLab.git
cd ConfinedLab
```

### 2. Create and activate the conda environment

```bash
conda env create -f gwmodelling.yml
conda activate gwmodelling
```

This installs Python and all dependencies at the exact versions used for the manuscript, including the appropriate version of [mlibs](https://github.com/femarivera/mlibs).

### 3. Install MODFLOW 6

The results were produced with **MODFLOW 6 >= 6.5.0**. Needs prior installation.

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

## Experiments

| Folder | Description |
|---|---|
| `01 Recharge decrease 50percent - Base case` | Base case: 50% recharge decrease |
| `02 Recharge decrease 100percent` | 100% recharge decrease |
| `03 Recharge decrease 50percent Random` | Base case with random heterogeneous parameter fields |
| `04 Recharge decrease 50percent B 3 layers` | Base case: 3 layer model |
| `05 Recharge decrease 50percent B 450m` | Base case: Reduced basin lateral extent |
| `06 Recharge decrease 50percent 7 layers no outcrops` | Base case: 7 layer model with no outcrops on deepest confined aquifers |
| `07 Recharge decrease 50percent 7 layers` | Base case: 7 layer model with outcrops for all confined aquifers |
| `08 Recharge decrease 50percent Mixed aquitards` | Base case: Aquitards with different thicknesses |

## Templates

`templates/2D` and `templates/3D` contain starting scripts to build new experiments:

| File | Description |
|---|---|
| `* model template full script.py` | Complete model build, run and post-processing workflow |
| `* model template simplified script.py` | Minimal version of the workflow |
| `Parameter_analysis.py` | Parameter sensitivity analysis |
| `Sustainable_yield.py` | Sustainable yield estimation using `modpump6` |
| `Run_parallel.py` | Parallel launcher |
| `setup.xlsx` | File with input settings as described below |

## `setup.xlsx` sheet reference

| Sheet | Description |
|---|---|
| `grid` | Definition of the structured grid parameters. |
| `geometry` | Parameters for constructing the synthetic model geometry, using the `modgeom6` module functions. |
| `tdis` | Time discretization. |
| `parameters` | Model parameters per zone. |
| `transient_recharge` | Time series of recharge rates per zone. |
| `observations` | Cell IDs of the observation points. |
| `wells_st` | Well locations and pumping rates for the steady-state simulation. |
| `wells` | Well locations and pumping rate time series for the transient simulation. |
| `response_times` | Parameters for response time estimation via the `modtransient6` module functions. **Note:** the steady-state recharge and pumping rates here must match those defined for the last stress period of the transient series. |
| `q_values_st` | Pumping rates per well for sequential steady-state runs, used to investigate capture rates and sustainable yields via the `modpump6` module functions. |
| `q_values_tr` | Pumping rates per well for sequential transient runs, used to investigate capture rates and sustainable yields via the `modpump6` module functions. |
| `parameter_analysis` | Parameter sets for sensitivity analysis, used to launch parallel runs. |

## Usage

From a subfolder within runs/Response time experiments:

### 1. Run a single model

```bash
python Model.py
```

This builds and runs the model for the current parameter set and estimates response times.

### 2. Run a parameter sensitivity analysis in parallel

```bash
python Run_parallel.py
```

This launches parallel runs across all parameter sets defined in `setup.xlsx - parameter_analysis`. A folder is created per parameter set, containing the simulation results and the response time estimation for that run. Check the number of available cores to define the active workers before launching.

### 3. Post-process and visualize results

Then run:

```bash
python "Postprocessing response time.py"
```

This generates summary plots and a `tr_analysis.csv` file summarizing the results across all parameter sets. Within each experiment folder, all setup files and scripts have their respective parameters and variables set to reproduce the results presented in the manuscript.

---

## License

This project is licensed under the BSD 3-Clause License — see the [LICENSE](LICENSE) file for details.

---

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