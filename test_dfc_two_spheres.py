#!/usr/bin/env python3
"""Characterize and regression-test DFC contact between two equal spheres.

The script moves two fixed DEME spheres through a loading/unloading cycle,
reads the force produced by the compiled CUDA model, compares it with a CPU
oracle for the zero-rate normal law, and writes CSV and PNG diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from dfc_model import DFCParameters, configure_force_model


def normal_force_oracle(distance: float, radius: float, p: DFCParameters) -> dict[str, float | str]:
    """Reference zero-rate sphere/sphere response; compression is positive."""
    overlap = max(2.0 * radius - distance, 0.0)
    if overlap <= 0.0:
        return dict(overlap=0.0, strain=0.0, area=0.0, stress=0.0, force=0.0, branch="open")
    l0 = 2.0 * radius - p.mortar_layer
    lij = 2.0 * radius - overlap
    delta_rmin = overlap * 0.5 * (2.0 * radius - overlap) / lij
    area = math.pi * max(radius**2 - abs(radius - delta_rmin) ** 2, 0.0)
    eps_a = math.log(1.0 - p.mortar_layer / l0)
    strain = math.log(lij / l0)
    if strain >= 0.0:
        stress = min(p.ENm * strain, p.sgmTmax * (1.0 + p.lambda_init))
        branch = "tension"
    elif strain >= eps_a:
        stress = p.ENm * strain
        branch = "mortar_compression"
    else:
        stress = p.ENm * eps_a + p.ENa * (strain - eps_a)
        branch = "aggregate_compression"
    return dict(overlap=overlap, strain=strain, area=area, stress=stress,
                force=-area * stress, branch=branch)


def run_gpu_sweep(radius: float, samples: int, timestep: float, p: DFCParameters):
    import deme

    solver = deme.DEMSolver()
    solver.SetVerbosity("WARNING")
    solver.InstructBoxDomainDimension(100.0, 100.0, 100.0)
    solver.SetGravitationalAcceleration([0.0, 0.0, 0.0])
    solver.SetInitTimeStep(timestep)
    solver.SetCDUpdateFreq(1)
    solver.SetMaxVelocity(1.0e5)
    solver.SetErrorOutVelocity(2.0e5)
    solver.SetContactOutputContent(["OWNER", "FORCE", "CNT_WILDCARD"])

    material = solver.LoadMaterial(p.material_properties())
    configure_force_model(solver, Path(__file__).with_name("DFCModel.cu"))
    mass = p.density * (4.0 / 3.0) * math.pi * radius**3
    sphere = solver.LoadSphereType(mass, radius, material)
    left_batch = solver.AddClumps(sphere, [[-radius, 0.0, 0.0]])
    right_batch = solver.AddClumps(sphere, [[radius, 0.0, 0.0]])
    left_batch.SetFamily(1)
    right_batch.SetFamily(2)
    solver.SetFamilyFixed(1)
    solver.SetFamilyFixed(2)
    left = solver.Track(left_batch)
    right = solver.Track(right_batch)
    solver.Initialize()

    # Cover open, tensile-film, mortar compression, and shallow aggregate contact.
    min_distance = 2.0 * radius - 2.25 * p.mortar_layer
    max_distance = 2.0 * radius + 0.25 * p.mortar_layer
    loading = [max_distance - i * (max_distance - min_distance) / (samples - 1)
               for i in range(samples)]
    path = [("loading", d) for d in loading] + [("unloading", d) for d in reversed(loading[:-1])]
    rows = []
    for phase, distance in path:
        left.SetPos([-0.5 * distance, 0.0, 0.0])
        right.SetPos([0.5 * distance, 0.0, 0.0])
        left.SetVel([0.0, 0.0, 0.0])
        right.SetVel([0.0, 0.0, 0.0])
        solver.DoDynamicsThenSync(timestep)
        # ContactWrenches is zero for an owner with no current contacts, unlike
        # the acceleration scratch value of a fixed body, which may be stale.
        gpu_force = (-float(left.ContactWrenches()[0][0][0])
                     if distance < 2.0 * radius and solver.GetNumContacts() else 0.0)
        ref = normal_force_oracle(distance, radius, p)
        rows.append({"phase": phase, "distance": distance, "gpu_force": gpu_force, **ref})
    return rows


def write_results(rows, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "two_sphere_force_distance.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for phase, style in (("loading", "-"), ("unloading", "--")):
        subset = [row for row in rows if row["phase"] == phase]
        axes[0, 0].plot([r["distance"] for r in subset], [r["gpu_force"] for r in subset],
                        style, label=f"DEME {phase}")
    loading = [row for row in rows if row["phase"] == "loading"]
    axes[0, 0].plot([r["distance"] for r in loading], [r["force"] for r in loading],
                    ":", color="black", label="CPU oracle")
    axes[0, 0].axhline(0, color="0.6", linewidth=.8)
    axes[0, 0].set_yscale("symlog", linthresh=0.1)
    axes[0, 0].set(xlabel="center distance [mm]", ylabel="normal force [model units]",
                   title="Force–distance response")
    axes[0, 0].legend()

    axes[0, 1].plot([r["overlap"] for r in loading], [r["strain"] for r in loading])
    axes[0, 1].axhline(0, color="0.6", linewidth=.8)
    axes[0, 1].set(xlabel="geometric overlap [mm]", ylabel="normal logarithmic strain",
                   title="DFC normal strain")
    axes[1, 0].plot([r["overlap"] for r in loading], [r["area"] for r in loading])
    axes[1, 0].set(xlabel="geometric overlap [mm]", ylabel="contact area [mm²]",
                   title="Effective contact area")
    axes[1, 1].plot([r["strain"] for r in loading], [r["stress"] for r in loading])
    axes[1, 1].axhline(0, color="0.6", linewidth=.8)
    axes[1, 1].set_yscale("symlog", linthresh=1.0e-3)
    axes[1, 1].set(xlabel="normal logarithmic strain", ylabel="normal stress [model units]",
                   title="Constitutive response")
    fig.suptitle("Two-sphere DFC contact characterization")
    png_path = output_dir / "two_sphere_force_distance.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    return csv_path, png_path


def validate(rows, relative_tolerance: float = 2.0e-2) -> None:
    compared = [r for r in rows if abs(float(r["force"])) > 1.0e-10]
    errors = [abs(float(r["gpu_force"]) - float(r["force"])) / abs(float(r["force"]))
              for r in compared]
    worst = max(errors, default=0.0)
    if not all(math.isfinite(float(r["gpu_force"])) for r in rows):
        raise AssertionError("CUDA model produced a non-finite force")
    if any(abs(float(r["gpu_force"])) > 1.0e-12 for r in rows if r["branch"] == "open"):
        raise AssertionError("CUDA sweep reported force after the spheres separated")
    if worst > relative_tolerance:
        raise AssertionError(f"GPU/CPU maximum relative error {worst:.3g} exceeds {relative_tolerance:.3g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radius", type=float, default=9.0, help="full DFC sphere radius [mm]")
    parser.add_argument("--samples", type=int, default=61)
    parser.add_argument("--timestep", type=float, default=1.0e-8)
    parser.add_argument("--output-dir", type=Path, default=Path("test_output"))
    args = parser.parse_args()
    if args.radius <= DFCParameters().mortar_layer or args.samples < 3 or args.timestep <= 0:
        parser.error("radius must exceed the mortar layer, samples >= 3, and timestep > 0")
    rows = run_gpu_sweep(args.radius, args.samples, args.timestep, DFCParameters())
    validate(rows)
    csv_path, png_path = write_results(rows, args.output_dir)
    print(f"Validated {len(rows)} two-sphere states")
    print(f"Data: {csv_path}")
    print(f"Plot: {png_path}")


if __name__ == "__main__":
    main()
