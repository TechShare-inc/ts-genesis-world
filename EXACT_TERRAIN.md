# Exact triangle surfaces for Terrain

This fork extends Genesis 1.4.0 Terrain with optional triangle contact surfaces.
Without `exact_surface`, the upstream heightfield path is unchanged. A caller
supplies geometry; the engine does not interpret robot, task, or terrain-generator
settings.

`gs.morphs.Terrain(exact_surface=payload, ...)` accepts:

| Key | Shape | Meaning |
| --- | --- | --- |
| `triangles` | `(N, 3, 3)` | Contact triangles, with winding pointing out of the solid |
| `cell_mask` | `(rows - 1, cols - 1)` | Heightfield cells whose original contact surface is replaced |
| `mesh_vertices` | `(V, 3)` | Rendering and bounding mesh vertices |
| `mesh_faces` | `(F, 3)` | Rendering mesh triangle indices |

Coordinates are in terrain-local metres. Heightfield dimensions and scales must
agree with the payload. Geometry generation and conversion are caller-owned.
The engine constructs a cell index, rejects invalid or missing triangle geometry,
and uses the triangles for both contact and `TerrainEntity.get_terrain_height`.
Height queries retain the upstream position shapes and environment selection.

Scenes using exact contact currently support one collision Terrain. The engine
rejects multiple collision Terrains in that case. These triangles are terrain
surfaces, not an arbitrary closed-mesh collision replacement.

CPU input checks: `python -m unittest discover -s tests -p test_exact_terrain_input.py`.
These checks cover indexing and input validation, not GPU contact dynamics.

The visualizer also supports attaching a viewer to a built scene, hiding and
showing it, and detaching it while preserving camera rendering. GL resources are
released in their owning context and recreated when that context changes.

Rigid solvers capture the first non-finite constraint acceleration before
integration. `set_constraint_failure_log_path(path)` enables JSON output on the
solver error; `constraint_failure_diagnostic()` reads the immutable snapshot and
raises if no failure has occurred. Schema 3 records current line-search values
and actual inertial properties, with sorted contacts mapped back to raw contact
indices. No robot, training framework, or output directory is assumed.

With a working GPU (and a display for the viewer test), run:

```sh
python -m unittest discover -s tests -p test_exact_terrain_gpu.py
python -m unittest discover -s tests -p test_constraint_snapshot_gpu.py
python -m unittest discover -s tests -p test_viewer_lifecycle_gpu.py
```
