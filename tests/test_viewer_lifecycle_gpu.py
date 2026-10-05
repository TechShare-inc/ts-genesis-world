"""Interactive attach/hide/show/detach with an existing offscreen camera."""
import unittest
import threading
from unittest.mock import patch

import genesis as gs
import numpy as np


class ViewerLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        gs.init(backend=gs.gpu, logging_level="warning")

    def test_viewer_lifecycle_preserves_camera_rendering(self):
        thread_errors = []
        hook = patch.object(threading, "excepthook", side_effect=lambda error: thread_errors.append(error.exc_value))
        hook.start()
        self.addCleanup(hook.stop)
        self.addCleanup(lambda: self.assertEqual(thread_errors, []))
        scene = gs.Scene(show_viewer=False,
            viewer_options=gs.options.ViewerOptions(res=(320,240),realtime_factor=None),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]))
        scene.add_entity(gs.morphs.Plane())
        scene.add_entity(gs.morphs.Box(size=(.2,.2,.2),pos=(0,0,.1)))
        camera = scene.add_camera(res=(64,64),pos=(1,1,1),lookat=(0,0,.1),GUI=False)
        scene.build(n_envs=4)
        self.addCleanup(scene.destroy)
        visualizer=scene._visualizer
        original_options = scene.options.viewer.model_dump()
        reference = camera.render()[0]
        for operation in (visualizer.show_viewer, visualizer.hide_viewer,
                          visualizer.show_viewer, visualizer.detach_viewer,
                          visualizer.show_viewer):
            print("viewer operation:", operation.__name__, flush=True)
            self.assertTrue(operation())
            self.assertEqual(scene.options.viewer.model_dump(), original_options)
            visualizer.update(force=True)
            rgb=camera.render()[0]
            self.assertEqual(rgb.shape,reference.shape)
            self.assertTrue(np.isfinite(rgb).all())
            self.assertGreater(np.std(rgb),0)


if __name__ == "__main__":
    unittest.main()
