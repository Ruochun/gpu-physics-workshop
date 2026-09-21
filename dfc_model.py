"""Shared configuration and input helpers for the discrete fresh concrete model."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class DFCParameters:
    """DFC parameters in the millimetre-second unit system used by the L-box."""

    mortar_layer: float = 4.0
    density: float = 1.47919e-9
    ENm: float = 2.0
    ENa: float = 100.0
    alpha: float = 0.25
    beta: float = 0.5
    np: float = 1.0
    sgmTmax: float = 2.0e-4
    sgmTau0: float = 1.0e-7
    kappa0: float = 100.0
    eta_inf: float = 2.0e-5
    flocbeta: float = 0.01
    flocm: float = 1.0
    flocTcr: float = 100.0
    lambda_init: float = 2.0

    def material_properties(self) -> dict[str, float]:
        """Return names expected by ``DFCModel.cu``."""
        return {
            "mortar_layer_mat": self.mortar_layer,
            "ENm_mat": self.ENm,
            "ENa_mat": self.ENa,
            "alpha_mat": self.alpha,
            "beta_mat": self.beta,
            "np_mat": self.np,
            "sgmTmax_mat": self.sgmTmax,
            "sgmTau0_mat": self.sgmTau0,
            "kappa0_mat": self.kappa0,
            "eta_inf_mat": self.eta_inf,
            "flocbeta_mat": self.flocbeta,
            "flocm_mat": self.flocm,
            "flocTcr_mat": self.flocTcr,
            "lambda_init_mat": self.lambda_init,
        }


@dataclass(frozen=True)
class Particle:
    x: float
    y: float
    z: float
    aggregate_radius: float
    radius: float
    mass: float


CONTACT_WILDCARDS = {
    "contact_info_step_time",
    "contact_info_lambda",
    "contact_info_strain_x",
    "contact_info_strain_y",
    "contact_info_strain_z",
}


def read_particles_csv(
    filename: str | Path, parameters: DFCParameters = DFCParameters()
) -> list[Particle]:
    """Read ``X,Y,Z,r`` aggregate data and add the mortar layer to ``r``.

    Raises ``ValueError`` with a row number for malformed, non-finite, or
    non-positive input.  The mass is based on the full DFC sphere radius.
    """
    path = Path(filename)
    particles: list[Particle] = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"X", "Y", "Z", "r"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing CSV column(s): {', '.join(sorted(missing))}")
        for row_number, row in enumerate(reader, start=2):
            try:
                x, y, z, aggregate_radius = (float(row[key]) for key in ("X", "Y", "Z", "r"))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{row_number}: X, Y, Z, and r must be numbers") from exc
            values = (x, y, z, aggregate_radius)
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{path}:{row_number}: values must be finite")
            if aggregate_radius <= 0.0:
                raise ValueError(f"{path}:{row_number}: aggregate radius must be positive")
            radius = aggregate_radius + parameters.mortar_layer
            if radius <= 0.0:
                raise ValueError(f"{path}:{row_number}: full radius must be positive")
            volume = (4.0 / 3.0) * math.pi * radius**3
            particles.append(Particle(x, y, z, aggregate_radius, radius, parameters.density * volume))
    if not particles:
        raise ValueError(f"{path}: no particle rows found")
    return particles


def add_particles(solver, material, particles: Sequence[Particle], family: int = 0):
    """Create sphere templates, add the particles to DEME, and return a tracker."""
    templates = [solver.LoadSphereType(p.mass, p.radius, material) for p in particles]
    positions = [[p.x, p.y, p.z] for p in particles]
    batch = solver.AddClumps(templates, positions)
    batch.SetFamily(family)
    return solver.Track(batch)


def configure_force_model(solver, model_path: str | Path):
    """Install the CUDA DFC model and declare all material/history inputs."""
    model = solver.ReadContactForceModel(str(Path(model_path).resolve()))
    model.SetMustHaveMatProp(set(DFCParameters().material_properties()))
    model.SetMustPairwiseMatProp(set())
    model.SetPerContactWildcards(CONTACT_WILDCARDS)
    return model
