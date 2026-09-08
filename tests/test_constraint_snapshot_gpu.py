"""Explicit GPU check of first-failure capture; no robot or training dependencies."""
import json
import tempfile
import unittest
from pathlib import Path

import genesis as gs
import numpy as np
from genesis.utils.misc import qd_to_numpy


class ConstraintSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        gs.init(backend=gs.gpu, logging_level="warning")

    def test_first_invalid_solve_is_immutable_and_json_safe(self):
        scene = gs.Scene(show_viewer=False)
        scene.add_entity(gs.morphs.Plane())
        scene.add_entity(gs.morphs.Box(size=(.2,.2,.2),pos=(0,0,.12)))
        scene.build(n_envs=4)
        self.addCleanup(scene.destroy)
        scene.step(update_visualizer=False)
        solver=scene.sim.rigid_solver
        with self.assertRaises(Exception):
            solver.constraint_failure_diagnostic()
        qacc=solver.constraint_solver.constraint_state.qacc
        values=qd_to_numpy(qacc)
        values[1,2]=np.nan
        values[2,3]=np.inf
        qacc.from_numpy(values)
        solver.capture_constraint_failure_snapshot(17)
        first=solver.constraint_failure_diagnostic()
        self.assertEqual(first["environment_index"],2)
        self.assertEqual(first["invalid_environment_indices"],[2,3])
        self.assertEqual(first["invalid_dof_indices"],[1])
        self.assertEqual(first["simulation"]["substep_global"],17)
        json.dumps(first,allow_nan=False)
        values[:]=np.nan
        qacc.from_numpy(values)
        solver.capture_constraint_failure_snapshot(18)
        self.assertEqual(first,solver.constraint_failure_diagnostic())
        with tempfile.TemporaryDirectory(dir=Path.cwd()/"build") as directory:
            path=Path(directory)/"failure.json"
            solver.set_constraint_failure_log_path(path)
            solver._write_constraint_failure()
            self.assertEqual(json.loads(path.read_text()),first)


if __name__ == "__main__":
    unittest.main()
