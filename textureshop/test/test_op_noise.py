import unittest

import imageio as ii
import moderngl as mg
import numpy as np

from ..op_base import Init
from ..op_noise import CPURandom, FBMNoise, Gradient
from ..util import npwrite


GL = mg.create_standalone_context()
init = Init(size=(256, 256), gl=GL)


class NoiseTest(unittest.TestCase):

    def test_noise(self):
        print("[+] Testing noise node")

        size = (init.W, init.H, 4)

        cpu_random_a = CPURandom().in_node(init).out_node()
        assert np.amax(cpu_random_a) <= 1.0
        assert np.amin(cpu_random_a) >= 0.0
        assert size == cpu_random_a.shape

        cpu_random_b = CPURandom().in_node(init, 1.0, 100.0).out_node()
        assert np.amax(cpu_random_b) <= 100.0
        assert np.amin(cpu_random_b) >= 1.0
        assert size == cpu_random_b.shape

        debug_cpurandom_a = np.multiply(cpu_random_a, 255.0).astype(np.uint8)
        debug_cpurandom_a[:, :, -1] = 255
        ii.imwrite("cpu_random_a.png", debug_cpurandom_a)
        print("\t[+][+] output cpu_random_a.png generated.")

        # quality test requires human eye
        fbm_a = FBMNoise().in_node(init, cpu_random_a).out_node()
        assert np.amax(fbm_a) <= 1.0
        assert np.amin(fbm_a) >= 0.0
        assert size == fbm_a.shape

        debug_fbm_a = np.multiply(fbm_a, 255.0).astype(np.uint8)
        debug_fbm_a[:, :, -1] = 255
        ii.imwrite("fbm_a.png", debug_fbm_a)
        print("\t[+][+] output fbm_a.png generated.")

    def test_gaussian_blur(self):
        # TODO: write test
        print("Gaussian Blur test case not exists")
        assert True

    def test_gradient(self):
        init = Init(size=(128, 128))
        a = Gradient().in_node(init, Gradient.GRAD_HOR_RIGHT).out_node()
        b = Gradient().in_node(init, Gradient.GRAD_RAD_OUT).out_node()
        c = Gradient().in_node(init, Gradient.GRAD_RAD_IN).out_node()

        npwrite("a.png", a)
        npwrite("b.png", b)
        npwrite("c.png", c)

        # TODO: write test
        print("Gradient test case not finished")
        assert False


if __name__ == "__main__":
    unittest.main()
