"""Exact triangle surfaces for heightfield Terrain contact and height queries.

Payload keys: triangles (N,3,3), cell_mask (rows-1,cols-1), mesh_vertices
(V,3), mesh_faces (F,3). Coordinates are terrain-local metres, winding points
out of the solid. The mesh supplies rendering/AABB; contact uses triangles
and a terrain-cell CSR index. Masked heightfield cells are replaced.
Heightfield interpretation/conversion belongs to the caller, not this engine.
"""
import math
import numpy as np
import trimesh
import fast_simplification
import genesis as gs

def _build_exact_surface_index(
    heightfield_shape,
    horizontal_scale,
    exact_cell_mask,
    triangles,
):
    """Build a terrain-cell CSR index for exact horizontal and vertical faces."""
    cell_rows = heightfield_shape[0] - 1
    cell_cols = heightfield_shape[1] - 1
    mask = np.asarray(exact_cell_mask, dtype=bool)
    if mask.shape != (cell_rows, cell_cols):
        raise ValueError(
            "exact terrain cell mask shape mismatch: "
            f"expected {(cell_rows, cell_cols)}, got {mask.shape}"
        )

    surface_triangles = np.asarray(triangles, dtype=np.float32)
    if surface_triangles.shape == (0,):
        surface_triangles = np.empty((0, 3, 3), dtype=np.float32)
    if surface_triangles.ndim != 3 or surface_triangles.shape[1:] != (3, 3):
        raise ValueError(
            "exact terrain triangles must have shape (N, 3, 3); "
            f"got {surface_triangles.shape}"
        )

    n_cells = cell_rows * cell_cols
    if len(surface_triangles) == 0:
        if np.any(mask):
            raise ValueError("exact terrain cells require exact surface triangles")
        return {
            "exact_surface_cell_mask": mask.astype(np.uint8).ravel(),
            "exact_surface_cell_offsets": np.zeros(n_cells + 1, dtype=np.int32),
            "exact_surface_feature_indices": np.empty(0, dtype=np.int32),
            "exact_surface_triangles": surface_triangles,
            "exact_surface_normals": np.empty((0, 3), dtype=np.float32),
        }

    cross = np.cross(
        surface_triangles[:, 1] - surface_triangles[:, 0],
        surface_triangles[:, 2] - surface_triangles[:, 0],
    )
    norm = np.linalg.norm(cross, axis=1)
    if np.any(~np.isfinite(surface_triangles)) or np.any(~np.isfinite(norm)):
        raise ValueError("exact terrain surface contains non-finite values")
    if np.any(norm <= np.finfo(np.float32).eps):
        raise ValueError("exact terrain surface contains degenerate triangles")
    normals = (cross / norm[:, None]).astype(np.float32)

    cell_indices = []
    feature_indices = []
    epsilon = max(float(horizontal_scale) * 1e-6, 1e-8)
    for feature_index, triangle in enumerate(surface_triangles):
        xy_min = triangle[:, :2].min(axis=0) - epsilon
        xy_max = triangle[:, :2].max(axis=0) + epsilon
        row_min = max(0, math.floor(xy_min[0] / horizontal_scale))
        row_max = min(cell_rows - 1, math.floor(xy_max[0] / horizontal_scale))
        col_min = max(0, math.floor(xy_min[1] / horizontal_scale))
        col_max = min(cell_cols - 1, math.floor(xy_max[1] / horizontal_scale))
        if row_min > row_max or col_min > col_max:
            raise ValueError(
                "exact terrain triangle lies outside the heightfield: "
                f"feature={feature_index}"
            )
        for row in range(row_min, row_max + 1):
            start = row * cell_cols + col_min
            stop = row * cell_cols + col_max + 1
            cell_indices.extend(range(start, stop))
            feature_indices.extend([feature_index] * (stop - start))

    indexed_cells = np.asarray(cell_indices, dtype=np.int32)
    indexed_features = np.asarray(feature_indices, dtype=np.int32)
    order = np.argsort(indexed_cells, kind="stable")
    indexed_cells = indexed_cells[order]
    indexed_features = indexed_features[order]
    counts = np.bincount(indexed_cells, minlength=n_cells)
    cumulative_counts = np.cumsum(counts, dtype=np.int64)
    if cumulative_counts[-1] > np.iinfo(np.int32).max:
        raise ValueError("exact terrain surface index exceeds int32 capacity")
    offsets = np.zeros(n_cells + 1, dtype=np.int32)
    offsets[1:] = cumulative_counts.astype(np.int32)

    missing = mask.ravel() & (counts == 0)
    if np.any(missing):
        first_missing = int(np.flatnonzero(missing)[0])
        raise ValueError(
            "exact terrain cell has no indexed surface feature: "
            f"cell={(first_missing // cell_cols, first_missing % cell_cols)}"
        )

    return {
        "exact_surface_cell_mask": mask.astype(np.uint8).ravel(),
        "exact_surface_cell_offsets": offsets,
        "exact_surface_feature_indices": indexed_features,
        "exact_surface_triangles": surface_triangles,
        "exact_surface_normals": normals,
    }


def prepare_surface(shape, scale, payload, surface):
    if set(payload) != {"triangles", "cell_mask", "mesh_vertices", "mesh_faces"}:
        raise ValueError("exact_surface requires triangles, cell_mask, mesh_vertices, mesh_faces")
    index = _build_exact_surface_index(shape, scale, payload["cell_mask"], payload["triangles"])
    vertices = np.asarray(payload["mesh_vertices"], dtype=np.float32)
    faces = np.asarray(payload["mesh_faces"], dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("Invalid exact surface mesh vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("Invalid exact surface mesh faces")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Exact surface mesh face index out of bounds")
    triangles = index["exact_surface_triangles"]
    if len(triangles) and (np.any(triangles.min(axis=(0,1)) < vertices.min(axis=0)-1e-5)
                          or np.any(triangles.max(axis=(0,1)) > vertices.max(axis=0)+1e-5)):
        raise ValueError("Contact triangles exceed the geometry AABB")
    tmesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    # Same lossless rendering/AABB simplification as upstream Terrain.
    # Keep the original mesh for SDF and exact contact triangles separately.
    vertices, faces = fast_simplification.simplify(
        tmesh.vertices, tmesh.faces, target_count=0, lossless=True
    )
    display = trimesh.Trimesh(vertices=vertices, faces=faces)
    vmesh = gs.Mesh.from_trimesh(mesh=display, surface=surface, metadata={})
    mesh = gs.Mesh.from_trimesh(mesh=display, surface=gs.surfaces.Collision(),
                               metadata={"sdf_mesh": tmesh})
    return vmesh, mesh, index


import torch
import quadrants as qd
from genesis.utils import array_class
from genesis.utils import geom as gu
from genesis.utils.misc import broadcast_tensor

@qd.kernel(fastcache=True)
def kernel_get_exact_height(
    envs_idx: qd.types.ndarray(),
    link_idx: qd.i32,
    horizontal_scale: float,
    positions: qd.types.ndarray(),
    height_field: qd.types.ndarray(),
    cell_mask: qd.types.ndarray(),
    offsets: qd.types.ndarray(),
    features: qd.types.ndarray(),
    triangles: qd.types.ndarray(),
    heights: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    rigid_config: qd.template(),
    tilt_tolerance: float,
    is_per_env: qd.template(),
):
    # For a normalized quaternion, qx^2 + qy^2 equals sin(tilt / 2)^2, independent of yaw
    tilt_sin_half = qd.sin(0.5 * tilt_tolerance)
    tilt_sin_half_sq = tilt_sin_half * tilt_sin_half
    qd.loop_config(serialize=qd.static(rigid_config.para_level == gs.PARA_LEVEL.NEVER))
    for i_b_, i_p in qd.ndrange(heights.shape[0], heights.shape[1]):
        i_b = envs_idx[i_b_]
        i_b_pos = i_b_ if qd.static(is_per_env) else 0
        link_pos = dyn_state.links.pos[link_idx, i_b]
        link_quat = dyn_state.links.quat[link_idx, i_b]
        query_pos = qd.Vector([positions[i_b_pos, i_p, 0], positions[i_b_pos, i_p, 1], link_pos[2]], dt=gs.qd_float)
        local_pos = gu.qd_inv_transform_by_trans_quat(query_pos, link_pos, link_quat)
        row = local_pos[0] / horizontal_scale
        col = local_pos[1] / horizontal_scale
        row_max = height_field.shape[0] - 1
        col_max = height_field.shape[1] - 1

        is_valid = (
            link_quat[1] * link_quat[1] + link_quat[2] * link_quat[2] <= tilt_sin_half_sq
            and row >= -1.0
            and row <= row_max + 1
            and col >= -1.0
            and col <= col_max + 1
        )
        if is_valid:
            row = qd.min(qd.max(row, 0.0), qd.cast(row_max, gs.qd_float))
            col = qd.min(qd.max(col, 0.0), qd.cast(col_max, gs.qd_float))
            i_row = qd.min(qd.cast(qd.floor(row), gs.qd_int), row_max - 1)
            i_col = qd.min(qd.cast(qd.floor(col), gs.qd_int), col_max - 1)
            frac_row = row - i_row
            frac_col = col - i_col

            h00 = height_field[i_row, i_col]
            h10 = height_field[i_row + 1, i_col]
            h01 = height_field[i_row, i_col + 1]
            h11 = height_field[i_row + 1, i_col + 1]
            if frac_row + frac_col <= 1.0:
                heights[i_b_, i_p] = h00 + frac_row * (h10 - h00) + frac_col * (h01 - h00) + link_pos[2]
            else:
                heights[i_b_, i_p] = h11 + (1.0 - frac_col) * (h10 - h11) + (1.0 - frac_row) * (h01 - h11) + link_pos[2]
            cell = i_row * col_max + i_col
            z = heights[i_b_, i_p]-link_pos[2]
            if cell_mask[cell] != 0:
                z = -qd.math.inf
            x, y = row*horizontal_scale, col*horizontal_scale
            for ref in range(offsets[cell], offsets[cell+1]):
                f = features[ref]
                a = qd.Vector([triangles[f,0,j] for j in qd.static(range(3))])
                b = qd.Vector([triangles[f,1,j] for j in qd.static(range(3))])-a
                c = qd.Vector([triangles[f,2,j] for j in qd.static(range(3))])-a
                det = b[0]*c[1]-b[1]*c[0]
                if det > 1e-10:
                    u = ((x-a[0])*c[1]-(y-a[1])*c[0])/det
                    v = (b[0]*(y-a[1])-b[1]*(x-a[0]))/det
                    if u >= -1e-5 and v >= -1e-5 and u+v <= 1.0+1e-5:
                        z = qd.max(z, a[2]+u*b[2]+v*c[2])
            heights[i_b_, i_p] = z+link_pos[2] if z > -qd.math.inf else qd.math.nan
        else:
            heights[i_b_, i_p] = qd.math.nan

def query_heights(terrain, positions, envs_idx=None):
    self = terrain._solver
    link_idx = terrain.base_link_idx

    positions = torch.as_tensor(positions, dtype=gs.tc_float, device=gs.device)
    if positions.ndim == 0 or positions.ndim > 3 or positions.shape[-1] != 2:
        gs.raise_exception("`positions` must have shape (2,), (n_points, 2), or (n_envs, n_points, 2).")

    envs_idx = self._scene._sanitize_envs_idx(envs_idx)
    n_envs = len(envs_idx)
    is_single_position = positions.ndim == 1
    n_points = 1 if is_single_position else positions.shape[-2]
    if n_points == 0:
        gs.raise_exception("`positions` must contain at least one position.")

    is_per_env = positions.ndim == 3 and positions.shape[0] != 1
    positions = broadcast_tensor(
        positions, gs.tc_float, (n_envs if is_per_env else 1, n_points, 2), ("envs_idx", "positions", "")
    )

    heights = torch.empty((n_envs, n_points), dtype=gs.tc_float, device=gs.device)
    kernel_get_exact_height(
        envs_idx,
        link_idx,
        terrain._morph.horizontal_scale,
        positions.contiguous(),
        terrain._terrain_height_field,
        terrain._exact_query_data["cell_mask"],
        terrain._exact_query_data["cell_offsets"],
        terrain._exact_query_data["feature_indices"],
        terrain._exact_query_data["triangles"],
        heights,
        self.dyn_state,
        self.rigid_config,
        tilt_tolerance=0.001,
        is_per_env=is_per_env,
    )

    if self.n_envs == 0:
        heights = heights[0]
    if is_single_position:
        heights = heights[..., 0]
    return heights
