# ConfinedLab

**ConfinedLab** is a modular Python toolkit for generating, simulating, and analyzing synthetic multilayer groundwater flow systems using MODFLOW 6. 
It is designed for research, teaching, and rapid prototyping of conceptual hydrogeological models of confined aquifer systems.

Its primary goal is to investigate the transient response of multilayer aquifer systems to external climatic and anthropogenic forcings by using 
numerical flow models. Furthermore, it aims to assess the implications of response times into past and future behaviour of the system to inform 
approaches for the estimation of sustainable yields.

---

## Features

- **Synthetic Geometry Generation:**  
  Easily create multilayer geometries of synthetic stratigraphic configurations of a sedimentary multilayer system.

- **Flexible Model Setup:**  
  Quickly define model grids, hydraulic properties, boundary conditions, and recharge scenarios.

- **Optimize a pumping scenario**  
  Estimate the sustainable yield based on a constrained optimization approach with user defined constrains and goals.

- **Visualization & Analysis:**  
  Built-in plotting and post-processing tools for heads, flows, budgets, and more.

- **Analyse parameter influence on results:**  
  Investigate the effect of a model parameter on the sustainable yield estimations.


## Example: Creating a Synthetic Model Geometry

```python
from mlibs import modgeom6

# Define grid and geometry parameters
nlay, nrow, ncol = 5, 1, 600
outcrop_cells = [200, 150, 100, 50, 0]
base_thicknesses = [300, 150, 200, 150, 200]

# Create idomain and geometry arrays
idomain = modgeom6.create_idomain(nlay, nrow, ncol, outcrop_cells, direction="right")
top = modgeom6.compute_top_all(idomain, outcrop_z=[100, 150, 200, 250, 350], transition=True, slope=True,
                               transition_cells=50, transition_type="contain",
                               outcrop_zmin=[0, 200, 300, 400, 500], outcrop_zmax=[200, 300, 400, 500, 500])
thickness = modgeom6.compute_thickness_all(idomain, base_thicknesses, transition=True, transition_cells=50)
bottom = modgeom6.compute_bottom(top, thickness)
```

## Contact

For questions, suggestions, or contributions, please contact:  
Carlos Felipe Marin Rivera  
Bordeaux INP, Lab EPOC, Université de Bordeaux  

---
