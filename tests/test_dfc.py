import math
import tempfile
import unittest
from pathlib import Path

from dfc_model import DFCParameters, read_particles_csv
from lbox_dfc import prepare_output_directory, write_pvd
from test_dfc_two_spheres import normal_force_oracle


class TestDFC(unittest.TestCase):
    def test_csv_adds_mortar_and_computes_full_sphere_mass(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "particles.csv"
            source.write_text("X,Y,Z,r\n1,2,3,5\n", encoding="utf-8")
            p = DFCParameters()
            particle = read_particles_csv(source, p)[0]
            self.assertEqual(particle.aggregate_radius, 5.0)
            self.assertEqual(particle.radius, 9.0)
            self.assertAlmostEqual(particle.mass, p.density * 4.0 / 3.0 * math.pi * 9.0**3)

    def test_csv_rejects_invalid_input(self):
        invalid = ("X,Y,Z\n0,0,0\n", "X,Y,Z,r\n0,0,0,-1\n", "X,Y,Z,r\n0,nan,0,1\n")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "bad.csv"
            for content in invalid:
                source.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_particles_csv(source)

    def test_reference_law_covers_all_normal_branches(self):
        p = DFCParameters()
        states = [normal_force_oracle(d, 9.0, p) for d in (18.1, 17.0, 13.0, 9.0)]
        self.assertEqual([s["branch"] for s in states], [
            "open", "tension", "mortar_compression", "aggregate_compression"
        ])
        self.assertLess(states[1]["force"], 0.0)
        self.assertGreater(states[2]["force"], 0.0)
        self.assertGreater(states[3]["force"], states[2]["force"])

    def test_pvd_writer_escapes_filenames(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "series.pvd"
            write_pvd(target, [(0.25, 0, "a&b.vtk")])
            text = target.read_text(encoding="utf-8")
            self.assertIn('timestep="0.25"', text)
            self.assertIn('file="a&amp;b.vtk"', text)

    def test_output_directory_requires_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "output"
            target.mkdir()
            generated = target / "particles_000000.vtk"
            generated.write_text("generated", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                prepare_output_directory(target, overwrite=False)
            prepare_output_directory(target, overwrite=True)
            self.assertFalse(generated.exists())


if __name__ == "__main__":
    unittest.main()
