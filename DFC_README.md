# DFC force model and particle loader

`DFCModel.cu` implements the discrete fresh concrete sphere/sphere,
sphere/mesh, and sphere/analytical-boundary contact law. `dfc_model.py`
contains the material constants used by the reference L-box example, registers
the CUDA model and its contact-history fields, and reads `X,Y,Z,r` input. The
CSV radius is the aggregate radius; the loader adds the 4 mm mortar layer and
uses the resulting full radius for both geometry and mass.

Run the fast unit tests with:

```bash
conda run -n tmp_workshop python -m unittest discover -s tests -v
```

Run the GPU two-sphere characterization with:

```bash
conda run -n tmp_workshop python test_dfc_two_spheres.py \
  --output-dir test_output
```

This writes `two_sphere_force_distance.csv` and
`two_sphere_force_distance.png`. The plot shows the CUDA and CPU-oracle force
curves, normal strain, effective contact area, and stress-strain response. The
first CUDA JIT compilation can take several minutes with CUDA 13; subsequent
runs use DEM-Engine's cache.

Application scripts can use the shared helpers as follows:

```python
from pathlib import Path
from dfc_model import DFCParameters, add_particles, configure_force_model, read_particles_csv

params = DFCParameters()
material = solver.LoadMaterial(params.material_properties())
configure_force_model(solver, Path("DFCModel.cu"))
particles = read_particles_csv("input.csv", params)
particle_tracker = add_particles(solver, material, particles)
```
