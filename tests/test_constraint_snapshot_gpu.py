"""Explicit GPU check of first-failure capture; no robot or training dependencies."""
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

import genesis as gs
from genesis.engine.solvers.rigid.constraint_failure import capture_pre_solve
from genesis.utils.misc import qd_to_numpy
from tests.utils.assertions import assert_equal


class ConstraintSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        gs.init(backend=gs.gpu, logging_level="warning")

    def test_normal_step_captures_failure_before_integration(self):
        from genesis.engine.solvers.rigid import rigid_solver as rigid_module

        scene = gs.Scene(show_viewer=False)
        scene.add_entity(gs.morphs.Plane())
        scene.add_entity(gs.morphs.Box(size=(.2, .2, .2), pos=(0, 0, .12)))
        scene.add_entity(gs.morphs.Box(size=(.2, .2, .2), pos=(2, 0, .12)))
        scene.build(n_envs=2)
        self.addCleanup(scene.destroy)
        scene.step(update_visualizer=False)
        solver = scene.sim.rigid_solver
        before = qd_to_numpy(solver.rigid_info.qpos).copy()
        substep = solver._sim.cur_substep_global
        solve = rigid_module.func_solve_body

        def inject_failure(*args):
            solve(*args)
            state = solver.constraint_solver.constraint_state
            values = qd_to_numpy(state.qacc).copy()
            values[0, 1] = np.nan
            state.qacc.from_numpy(values)

        with patch.object(rigid_module, "func_solve_body", side_effect=inject_failure):
            scene.step(update_visualizer=False)
        with self.assertRaisesRegex(Exception, "Invalid constraint forces"):
            solver.check_errno()
        result = solver.constraint_failure_diagnostic()
        self.assertEqual(result["schema_version"], 4)
        self.assertEqual(result["environment_index"], 1)
        self.assertEqual(result["simulation"]["substep_global"], substep)
        assert_equal(result["replay_state"]["qpos"], before[:, 1])
        self.assertEqual(len(result["constraint"]["islands"]), 2)
        self.assertEqual(set(result["constraint"]["dofs_island_idx"]), {0, 1})
        for island in result["constraint"]["islands"]:
            self.assertIn("ls_alpha", island)
            self.assertIn("line_search_evaluated", island)
        json.dumps(result, allow_nan=False)

    def test_first_invalid_solve_is_immutable_and_json_safe(self):
        scene = gs.Scene(show_viewer=False)
        scene.add_entity(gs.morphs.Plane())
        scene.add_entity(gs.morphs.Box(size=(.2,.2,.2),pos=(0,0,.12)))
        scene.build(n_envs=257)
        self.addCleanup(scene.destroy)
        scene.step(update_visualizer=False)
        solver=scene.sim.rigid_solver
        with self.assertRaises(Exception):
            solver.constraint_failure_diagnostic()
        state = solver.constraint_solver.constraint_state
        snapshot = solver._constraint_failure_snapshot
        warmstart = np.arange(np.prod(state.qacc_ws.shape), dtype=gs.np_float).reshape(state.qacc_ws.shape)
        acceleration = warmstart * 2 + 1
        is_warmstart = np.arange(scene.n_envs) % 2 == 0
        state.qacc_ws.from_numpy(warmstart)
        solver.dyn_state.dofs.acc.from_numpy(acceleration)
        state.is_warmstart.from_numpy(is_warmstart)
        capture_pre_solve(solver.dyn_state, state, snapshot)
        assert_equal(qd_to_numpy(snapshot.pre_solve_qacc_ws), warmstart)
        assert_equal(qd_to_numpy(snapshot.pre_solve_dof_acc), acceleration)
        assert_equal(qd_to_numpy(snapshot.pre_solve_is_warmstart), is_warmstart)
        qacc=solver.constraint_solver.constraint_state.qacc
        values=qd_to_numpy(qacc)
        values[1,2]=np.nan
        values[2,3]=np.inf
        values[0,-1]=-np.inf
        qacc.from_numpy(values)
        solver.capture_constraint_failure_snapshot(17)
        first=solver.constraint_failure_diagnostic()
        self.assertEqual(first["environment_index"],2)
        self.assertEqual(first["invalid_environment_indices"],[2,3,scene.n_envs-1])
        self.assertEqual(first["invalid_dof_indices"],[1])
        assert_equal(first["constraint"]["qacc_ws_before_solve"], warmstart[:, 2])
        assert_equal(first["dofs"]["acc_before_solve"], acceleration[:, 2])
        self.assertTrue(first["constraint"]["is_warmstart_before_solve"])
        self.assertEqual(first["simulation"]["substep_global"],17)
        json.dumps(first,allow_nan=False)
        values[:]=np.nan
        qacc.from_numpy(values)
        state.qacc_ws.fill(-1)
        solver.dyn_state.dofs.acc.fill(-2)
        state.is_warmstart.fill(False)
        capture_pre_solve(solver.dyn_state, state, snapshot)
        assert_equal(qd_to_numpy(snapshot.pre_solve_qacc_ws), warmstart)
        assert_equal(qd_to_numpy(snapshot.pre_solve_dof_acc), acceleration)
        assert_equal(qd_to_numpy(snapshot.pre_solve_is_warmstart), is_warmstart)
        solver.capture_constraint_failure_snapshot(18)
        self.assertEqual(first,solver.constraint_failure_diagnostic())
        with tempfile.TemporaryDirectory(dir=Path.cwd()/"build") as directory:
            path=Path(directory)/"failure.json"
            solver.set_constraint_failure_log_path(path)
            solver._write_constraint_failure()
            self.assertEqual(json.loads(path.read_text()),first)
        solver.clear_constraint_failure_snapshot()
        capture_pre_solve(solver.dyn_state, state, solver._constraint_failure_snapshot)
        values.fill(0)
        values[0,1]=-np.inf
        qacc.from_numpy(values)
        solver.capture_constraint_failure_snapshot(19)
        second=solver.constraint_failure_diagnostic()
        self.assertEqual(second["environment_index"],1)
        self.assertEqual(second["invalid_environment_indices"],[1])
        self.assertEqual(second["simulation"]["substep_global"],19)
        self.assertEqual(second["constraint"]["qacc"][0],"-inf")
        assert_equal(second["constraint"]["qacc_ws_before_solve"], -np.ones_like(warmstart[:, 1]))
        assert_equal(second["dofs"]["acc_before_solve"], -2 * np.ones_like(acceleration[:, 1]))
        self.assertFalse(second["constraint"]["is_warmstart_before_solve"])


if __name__ == "__main__":
    unittest.main()
