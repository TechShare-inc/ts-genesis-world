"""Top and vertical-face contact on a generic stepped surface."""
import unittest

import genesis as gs
import numpy as np
import torch


class ExactTerrainContactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        gs.init(backend=gs.gpu, logging_level="warning")

    def test_top_support_and_wall_contact(self):
        vertices=np.array([[0,0,0],[1,0,0],[1,2,0],[0,2,0],
                           [1,0,.2],[2,0,.2],[2,2,.2],[1,2,.2]],np.float32)
        faces=np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7],[1,4,7],[1,7,2]],np.int32)
        triangles = vertices[faces].copy()
        vertices = np.concatenate((vertices, np.array([[0,0,-.1],[2,0,-.1],[2,2,-.1],[0,2,-.1]],np.float32)))
        faces = np.concatenate((faces,np.array([[8,10,9],[8,11,10],[0,3,11],[0,11,8],
            [5,9,10],[5,10,6],[8,9,5],[8,5,4],[8,4,1],[8,1,0],
            [11,6,10],[11,7,6],[11,2,7],[11,3,2]],np.int32)))
        payload={"triangles":triangles,"cell_mask":np.ones((2,2),bool),
                 "mesh_vertices":vertices,"mesh_faces":faces}
        scene=gs.Scene(show_viewer=False,sim_options=gs.options.SimOptions(dt=.005),
                       rigid_options=gs.options.RigidOptions(constraint_solver=gs.constraint_solver.Newton))
        terrain=scene.add_entity(gs.morphs.Terrain(height_field=np.array([[0,0,0],[.2,.2,.2],[.2,.2,.2]]),
            horizontal_scale=1.,vertical_scale=1.,exact_surface=payload),material=gs.materials.Rigid(friction=2.))
        top=scene.add_entity(gs.morphs.Box(size=(.1,.1,.1),pos=(1.5,1,.35)),material=gs.materials.Rigid(friction=2.))
        wall=scene.add_entity(gs.morphs.Sphere(radius=.06,pos=(.8,1,.065)),material=gs.materials.Rigid(friction=2.))
        scene.build(n_envs=4)
        self.addCleanup(scene.destroy)
        heights=terrain.get_terrain_height([[.5,1],[1.5,1]])
        np.testing.assert_allclose(heights.cpu().numpy(),np.broadcast_to([0,.2],heights.shape),atol=1e-6)
        wall_contact=False
        for _ in range(150):
            wall.set_dofs_velocity(torch.ones((4,1),device=gs.device),dofs_idx_local=[0])
            scene.step(update_visualizer=False)
            contacts=wall.get_contacts()
            wall_contact |= bool((contacts["valid_mask"] & (contacts["normal"][...,0].abs()>.9)).any())
        positions=top.get_pos(relative=False)
        self.assertTrue(torch.isfinite(positions).all())
        torch.testing.assert_close(positions[:,2],torch.full((4,),.25,device=gs.device),atol=.01,rtol=0)
        self.assertTrue(wall_contact)
        self.assertLess(float(wall.get_pos(relative=False)[:,0].max()),.96)


if __name__ == "__main__":
    unittest.main()
