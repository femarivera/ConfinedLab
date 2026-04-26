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
├── docs/            ← documentation, manuals, and guides for the model files
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