"""CPU geometry checks; no scene, robot assets, or learning framework required."""
import unittest

import numpy as np

from genesis.utils.terrain_exact import _build_exact_surface_index


class ExactTerrainInputTests(unittest.TestCase):
    def test_empty_surface_preserves_heightfield_cells(self):
        index = _build_exact_surface_index((3, 4), 0.5, np.zeros((2, 3), bool), [])
        np.testing.assert_array_equal(index["exact_surface_cell_offsets"], np.zeros(7))
        self.assertEqual(index["exact_surface_triangles"].shape, (0, 3, 3))

    def test_horizontal_and_vertical_faces_are_indexed(self):
        triangles = np.array([
            [[0.1, 0.1, 1], [0.9, 0.1, 1], [0.1, 0.9, 1]],
            [[0.1, 0.1, 0], [0.9, 0.1, 0], [0.1, 0.1, 1]],
        ])
        index = _build_exact_surface_index((3, 3), 1.0, [[True, False], [False, False]], triangles)
        np.testing.assert_array_equal(index["exact_surface_cell_offsets"], [0, 2, 2, 2, 2])
        np.testing.assert_array_equal(index["exact_surface_feature_indices"], [0, 1])
        np.testing.assert_allclose(index["exact_surface_normals"], [[0, 0, 1], [0, -1, 0]])

    def test_invalid_surfaces_fail_instead_of_filling_missing_geometry(self):
        valid = np.array([[[0.1, 0.1, 1], [0.9, 0.1, 1], [0.1, 0.9, 1]]])
        cases = [
            (np.ones((2, 2), bool), [], "require exact surface triangles"),
            (np.ones((2, 2), bool), valid, "no indexed surface feature"),
            (np.zeros((1, 1), bool), valid, "mask shape mismatch"),
            (np.zeros((2, 2), bool), np.zeros((1, 3, 3)), "degenerate"),
            (np.zeros((2, 2), bool), valid * np.nan, "non-finite"),
            (np.zeros((2, 2), bool), valid + 10, "outside"),
        ]
        for mask, triangles, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                _build_exact_surface_index((3, 3), 1.0, mask, triangles)


if __name__ == "__main__":
    unittest.main()
