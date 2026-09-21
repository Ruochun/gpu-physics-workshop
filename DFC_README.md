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

## L-box simulation

Run the complete settling and flow experiment with:

```bash
conda run -n tmp_workshop python lbox_dfc.py --output-dir lbox_output
```

The default container uses `LBox.obj` as the complete confining boundary.
DEME splits it into natural convex patches using a 45-degree hard-angle
threshold before initialization. The temporary gate at `x=56 mm` is the only
analytical boundary. It remains active until total kinetic energy is below the
configured threshold for several consecutive checks; its particle contacts are
then disabled and the concrete flows into the horizontal section.
The output directory contains particle VTK frames, the L-box mesh, gate VTK
frames for the settling phase, `particles.pvd` and `scene.pvd` ParaView time
series, per-frame diagnostics, and a JSON record of all run parameters.

For a short installation check rather than a physical run:

```bash
conda run -n tmp_workshop python lbox_dfc.py \
  --particle-limit 10 --initial-timestep 1e-6 --max-timestep 1e-6 \
  --min-settle-time 0 --max-settle-time 1e-6 --flow-time 1e-6 \
  --ke-check-interval 1e-6 --output-interval 1e-6 \
  --output-dir /tmp/lbox-smoke
```
