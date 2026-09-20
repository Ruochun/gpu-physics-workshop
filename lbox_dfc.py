#!/usr/bin/env python3
"""Run the two-phase DFC simulation in the supplied L-box geometry."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from xml.sax.saxutils import quoteattr

from dfc_model import DFCParameters, add_particles, configure_force_model, read_particles_csv

PARTICLE_FAMILY = 0
LBOX_FAMILY = 10
GATE_FAMILY = 11


def positive(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return result


def nonnegative(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise argparse.ArgumentTypeError("must be a non-negative finite number")
    return result


def prepare_output_directory(path: Path, overwrite: bool) -> None:
    """Create an output directory, optionally removing only known products."""
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is not empty: {path} (use --overwrite)")
        known_names = {"frames.csv", "particles.pvd", "scene.pvd", "run_config.json", "lbox_mesh.vtk"}
        known_prefixes = ("particles_", "boundaries_")
        for item in path.iterdir():
            if item.is_file() and (item.name in known_names or item.name.startswith(known_prefixes)):
                item.unlink()
    path.mkdir(parents=True, exist_ok=True)


def write_pvd(path: Path, datasets: list[tuple[float, int, str]]) -> None:
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        "  <Collection>",
    ]
    for time, part, filename in datasets:
        lines.append(
            f"    <DataSet timestep={quoteattr(f'{time:.12g}')} part={quoteattr(str(part))} "
            f"file={quoteattr(filename)}/>"
        )
    lines.extend(("  </Collection>", "</VTKFile>", ""))
    path.write_text("\n".join(lines), encoding="utf-8")


class LBoxRun:
    def __init__(self, args: argparse.Namespace):
        import deme

        self.args = args
        self.deme = deme
        self.params = DFCParameters(mortar_layer=args.mortar_thickness)
        particles = read_particles_csv(args.particles, self.params)
        if args.particle_limit:
            particles = particles[: args.particle_limit]
        self.particles = particles
        self.solver = deme.DEMSolver()
        self.time = 0.0
        self.timestep = args.initial_timestep
        self.frame = 0
        self.frame_rows: list[dict[str, object]] = []
        self.particle_datasets: list[tuple[float, int, str]] = []
        self.scene_datasets: list[tuple[float, int, str]] = []

    def configure(self) -> None:
        s, a = self.solver, self.args
        s.SetVerbosity(a.verbosity)
        s.SetOutputFormat("VTK")
        s.SetOutputContent(["XYZ", "VEL", "ABSV", "FAMILY"])
        s.SetMeshOutputFormat("VTK")
        s.SetContactOutputContent(["OWNER", "FORCE", "CNT_WILDCARD"])
        s.SetJitifyClumpTemplates(False)
        s.SetJitifyMassProperties(False)
        s.InstructBoxDomainDimension(2000.0, 2000.0, 2000.0)
        s.SetErrorOutAvgContacts(250)
        s.SetGravitationalAcceleration([0.0, -9810.0, 0.0])

        material = s.LoadMaterial(self.params.material_properties())
        configure_force_model(s, Path(__file__).with_name("DFCModel.cu"))

        mesh = s.AddWavefrontMeshObject(str(a.mesh.resolve()), material)
        mesh.SetMass(1.0)
        mesh.SetFamily(LBOX_FAMILY)
        s.SetFamilyFixed(LBOX_FAMILY)

        # The supplied particle cloud reaches x ~= 52 mm. A plane at x=56 mm
        # closes the vertical compartment while leaving the mortar film intact.
        gate = s.AddBCPlane([a.gate_x, 0.0, 0.0], [-1.0, 0.0, 0.0], material)
        gate.SetFamily(GATE_FAMILY)
        s.SetFamilyFixed(GATE_FAMILY)

        self.particle_tracker = add_particles(s, material, self.particles, PARTICLE_FAMILY)
        self.ke_inspector = s.CreateInspector("clump_kinetic_energy")
        self.max_speed_inspector = s.CreateInspector("clump_max_absv")
        s.SetInitTimeStep(self.timestep)
        s.SetMaxVelocity(a.max_velocity)
        s.SetErrorOutVelocity(a.error_velocity)
        s.SetInitBinNumTarget(500_000)
        s.DisableAdaptiveBinSize()
        s.Initialize()

        mesh_path = a.output_dir / "lbox_mesh.vtk"
        s.WriteMeshFile(str(mesh_path))
        self.mesh_filename = mesh_path.name

    def advance_to(self, target: float) -> None:
        """Advance while smoothly increasing the initial conservative timestep."""
        while self.time + 1.0e-14 < target:
            if self.timestep < self.args.max_timestep:
                duration = min(target - self.time, 20.0 * self.timestep)
            else:
                duration = target - self.time
            self.solver.DoDynamicsThenSync(duration)
            self.time += duration
            if self.timestep < self.args.max_timestep:
                self.timestep = min(
                    self.args.max_timestep, self.timestep * self.args.timestep_growth
                )
                self.solver.UpdateStepSize(self.timestep)

    def write_frame(self, phase: str, gate_active: bool) -> None:
        particle_name = f"particles_{self.frame:06d}.vtk"
        self.solver.WriteSphereFile(str(self.args.output_dir / particle_name))
        self.particle_datasets.append((self.time, 0, particle_name))
        self.scene_datasets.extend(
            ((self.time, 0, particle_name), (self.time, 1, self.mesh_filename))
        )
        if gate_active:
            boundary_name = f"boundaries_{self.frame:06d}.vtk"
            self.solver.WriteAnalyticalFile(str(self.args.output_dir / boundary_name))
            self.scene_datasets.append((self.time, 2, boundary_name))
        ke = float(self.ke_inspector.GetValue())
        max_speed = float(self.max_speed_inspector.GetValue())
        contacts = int(self.solver.GetNumContacts())
        self.frame_rows.append(
            {
                "frame": self.frame,
                "time": self.time,
                "phase": phase,
                "gate_active": int(gate_active),
                "kinetic_energy": ke,
                "max_speed": max_speed,
                "contacts": contacts,
                "timestep": self.timestep,
            }
        )
        print(
            f"frame={self.frame:06d} time={self.time:.6g} phase={phase} "
            f"KE={ke:.6g} vmax={max_speed:.6g} contacts={contacts} dt={self.timestep:.3g}"
        )
        self.frame += 1

    def settle(self) -> str:
        a = self.args
        low_ke_count = 0
        next_output = 0.0
        next_check = 0.0
        self.write_frame("settling", True)
        next_output += a.output_interval
        next_check += a.ke_check_interval
        reason = "maximum settling time reached"
        while self.time < a.max_settle_time - 1.0e-14:
            event = min(next_output, next_check, a.max_settle_time)
            self.advance_to(event)
            if self.time + 1.0e-12 >= next_check:
                ke = float(self.ke_inspector.GetValue())
                if self.time >= a.min_settle_time and ke <= a.settle_ke:
                    low_ke_count += 1
                else:
                    low_ke_count = 0
                next_check += a.ke_check_interval
                if low_ke_count >= a.settle_hold:
                    reason = f"kinetic energy held below threshold for {low_ke_count} checks"
                    break
            if self.time + 1.0e-12 >= next_output:
                self.write_frame("settling", True)
                next_output += a.output_interval
        if not self.frame_rows or abs(float(self.frame_rows[-1]["time"]) - self.time) > 1.0e-12:
            self.write_frame("settling", True)
        return reason

    def flow(self) -> None:
        self.solver.DoDynamicsThenSync(0.0)
        self.solver.DisableContactBetweenFamilies(GATE_FAMILY, PARTICLE_FAMILY)
        flow_end = self.time + self.args.flow_time
        self.write_frame("flow", False)
        next_output = min(self.time + self.args.output_interval, flow_end)
        while self.time < flow_end - 1.0e-14:
            self.advance_to(next_output)
            self.write_frame("flow", False)
            next_output = min(next_output + self.args.output_interval, flow_end)

    def finalize(self, settling_reason: str) -> None:
        out = self.args.output_dir
        with (out / "frames.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.frame_rows[0]))
            writer.writeheader()
            writer.writerows(self.frame_rows)
        write_pvd(out / "particles.pvd", self.particle_datasets)
        write_pvd(out / "scene.pvd", self.scene_datasets)
        config = {
            "arguments": {key: str(value) if isinstance(value, Path) else value
                          for key, value in vars(self.args).items()},
            "material": asdict(self.params),
            "particle_count": len(self.particles),
            "settling_end_time": next(
                float(row["time"]) for row in reversed(self.frame_rows) if row["phase"] == "settling"
            ),
            "settling_reason": settling_reason,
            "final_time": self.time,
        }
        (out / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    def run(self) -> None:
        self.configure()
        reason = self.settle()
        print(f"Opening gate: {reason}")
        self.flow()
        self.finalize(reason)


def build_parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=Path, default=here / "input.csv")
    parser.add_argument("--mesh", type=Path, default=here / "LBox_Mesh.obj")
    parser.add_argument("--output-dir", type=Path, default=here / "lbox_output")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--mortar-thickness", type=nonnegative, default=4.0)
    parser.add_argument("--gate-x", type=float, default=56.0)
    parser.add_argument("--initial-timestep", type=positive, default=1.0e-8)
    parser.add_argument("--max-timestep", type=positive, default=1.0e-5)
    parser.add_argument("--timestep-growth", type=positive, default=1.01)
    parser.add_argument("--settle-ke", type=nonnegative, default=0.5)
    parser.add_argument("--settle-hold", type=int, default=5)
    parser.add_argument("--min-settle-time", type=nonnegative, default=0.1)
    parser.add_argument("--max-settle-time", type=positive, default=5.0)
    parser.add_argument("--ke-check-interval", type=positive, default=0.05)
    parser.add_argument("--flow-time", type=nonnegative, default=10.0)
    parser.add_argument("--output-interval", type=positive, default=0.1)
    parser.add_argument("--max-velocity", type=positive, default=40000.0)
    parser.add_argument("--error-velocity", type=positive, default=50000.0)
    parser.add_argument("--particle-limit", type=int, default=0,
                        help="use only the first N particles for smoke testing")
    parser.add_argument("--verbosity", choices=("QUIET", "ERROR", "WARNING", "INFO", "METRIC", "DEBUG"),
                        default="INFO")
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.particles.is_file():
        parser.error(f"particle CSV does not exist: {args.particles}")
    if not args.mesh.is_file():
        parser.error(f"mesh does not exist: {args.mesh}")
    if args.initial_timestep > args.max_timestep:
        parser.error("--initial-timestep cannot exceed --max-timestep")
    if args.timestep_growth < 1.0:
        parser.error("--timestep-growth must be at least 1")
    if args.settle_hold < 1:
        parser.error("--settle-hold must be at least 1")
    if args.particle_limit < 0:
        parser.error("--particle-limit cannot be negative")
    if args.error_velocity <= args.max_velocity:
        parser.error("--error-velocity must exceed --max-velocity")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    prepare_output_directory(args.output_dir, args.overwrite)
    LBoxRun(args).run()


if __name__ == "__main__":
    main()
