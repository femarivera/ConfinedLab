# ConfinedLab

**ConfinedLab** is a modular Python toolkit for generating, simulating, and analyzing synthetic multilayer groundwater flow systems using MODFLOW 6. 
It is designed for research, teaching, and rapid prototyping of conceptual and numerical hydrogeological models of confined aquifer systems.

As part of the [PEPR One Water DEESAC project](https://www.onewater.fr/fr/actualite/actualite/lancement-du-projet-deesac-durabilite-exploitabilite-des-eaux-souterraines-des "Go to onewater.fr"), its primary goal is to investigate the transient response of multilayer aquifer systems to external 
climatic and anthropogenic forcings by using numerical flow models, with a focus on confined aquifers. Furthermore, it aims to assess the implications 
of response times into past and future behaviour of the system to inform sustainability assessments.

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

# Define synthetic geometry generation parameters
nlay, nrow, ncol = 5, 1, 600
epsilon = 0 # Minimum allowed cell thickness in meters
outcrop_z = np.array([100, 150, 200, 250, 350]) # Elevation (Just used when SLOPE is set to False)
outcrop_zmax = np.array([200, 300, 400, 500, 500]) # Elevation (Just used when SLOPE are set to True)
outcrop_zmin = np.array([0, 200, 300, 400, 500]) # Elevation (Just used when SLOPE are set to True)
base_thicknesses = np.array([300, 150, 200, 150, 200]) # Layer thickness in meters
outcrop_cells = np.array([300, 250, 150, 100, 0]) 
transition = 60 # Number of transitions cells

# Create idomain and geometry arrays
idomain = modgeom6.compute_idomain(nlay, nrow, ncol, outcrop_cells)
ztop = modgeom6.compute_top(idomain, outcrop_z, transition=True, slope=True,
                            transition_cells=transition, transition_type="contain", 
                            outcrop_zmin=outcrop_zmin, outcrop_zmax=outcrop_zmax)
thickness_array = modgeom6.compute_thickness(idomain, base_thicknesses, 
                                             transition=True, transition_type="extend", 
                                             transition_cells=transition)
zbot = modgeom6.compute_bottom(ztop, thickness_array)
idomain = modgeom6.idomain_from_thickness(thickness_array, epsilon)

# --- flopy simulation building section --- #

modplot6.plot_cross_section_array(gwf, 
                      zone_array, 
                      nrow//2, 
                      figsize=(19, 5),
                      fontsize=14,
                      label="Model layes") 
```
![Example geometry output](assets/example_output_geometry.png)


```python
from mlibs import modgeom6, modplot6

# Define synthetic geometry generation parameters
nlay, nrow, ncol = 5, 1, 600
epsilon = 0 # Minimum allowed cell thickness in meters
outcrop_z = np.array([100, 150, 200, 250, 350]) # Elevation (Just used when SLOPE is set to False)
outcrop_zmax = np.array([200, 300, 400, 500, 500]) # Elevation (Just used when SLOPE are set to True)
outcrop_zmin = np.array([0, 200, 300, 400, 500]) # Elevation (Just used when SLOPE are set to True)
base_thicknesses = np.array([300, 150, 200, 150, 200]) # Layer thickness in meters
outcrop_cells = np.array([300, 250, 150, 100, 0]) 
transition = 60 # Number of transitions cells

# Create idomain and geometry arrays
idomain = modgeom6.compute_idomain(nlay, nrow, ncol, outcrop_cells)
ztop = modgeom6.compute_top(idomain, outcrop_z, transition=True, slope=True,
                            transition_cells=transition, transition_type="contain", 
                            outcrop_zmin=outcrop_zmin, outcrop_zmax=outcrop_zmax)
thickness_array = modgeom6.compute_thickness(idomain, base_thicknesses, 
                                             transition=True, transition_type="extend", 
                                             transition_cells=transition)
zbot = modgeom6.compute_bottom(ztop, thickness_array)
idomain = modgeom6.idomain_from_thickness(thickness_array, epsilon)

# --- flopy simulation building section --- #

modplot6.plot_cross_section_array(gwf, 
                      zone_array, 
                      nrow//2, 
                      figsize=(19, 5),
                      fontsize=14,
                      label="Model layes") 
```
![Example geometry output](assets/example_output_geometry_2.png)


## Contact

For questions, suggestions, or contributions, please contact:  
Carlos Felipe Marin Rivera  
Bordeaux INP, Lab EPOC, Université de Bordeaux
cmarinriver@bordeaux-inp.fr  

<p float="left">
  <img src="assets/logo_ensegid.jpg" height="50" style="margin-right:10px;" />
  <img src="assets/logo_epoc.png" height="50" style="margin-right:10px;" />
  <img src="assets/logo_ubordeaux.png" height="50" />
</p>


---
