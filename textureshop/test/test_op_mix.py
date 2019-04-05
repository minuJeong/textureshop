import unittest

import moderngl as mg
import numpy as np

from ..op_base import Init
from ..op_mix import Mix, Smoothstep, Rotate
from ..op_noise import FBMNoise, Gradient
from .. util import npwrite


GL = mg.create_standalone_context()
init = Init(size=(2, 2), gl=GL)


class OperationTest(unittest.TestCase):

    def test_mix_smoothstep(self):
        print("[+] Testing mix/smoothstep node")

        PATIENCE = 1e-5

        in_a = np.multiply(np.ones((2, 2, 4)), 0.23)
        in_b = np.multiply(np.ones((2, 2, 4)), 1.05)
        in_c = np.multiply(np.ones((2, 2, 4)), 0.25)

        print("\t[+][+] Testing mix..")
        mix_a = Mix().in_node(init, in_a, in_b, in_c).out_node()
        np_mix_a = in_a * (1.0 - in_c) + in_b * in_c
        assert np.all(np.isclose(mix_a, np_mix_a, atol=PATIENCE))

        noise = FBMNoise().in_node(init).out_node()
        mix_b = Mix().in_node(init, 0.0, 1.0, noise).out_node()
        assert np.all(np.isclose(noise, mix_b, atol=PATIENCE))

        print("\t[+][+] Testing smoothstep..")
        smoothstep_a = Smoothstep().in_node(init, in_a, in_b, in_c).out_node()

        def numpy_smoothstep(a, b, c):
            t = np.clip((c - a) / (b - a), 0.0, 1.0)
            return t * t * (3.0 - 2.0 * t)
        np_smoothstep_a = numpy_smoothstep(in_a, in_b, in_c)
        assert np.all(np.isclose(smoothstep_a, np_smoothstep_a, atol=PATIENCE))

    def test_rotate(self):
        print("[+] Testing rotate.. (requires Gradient to be already working)")

        init = Init(size=(128, 128))
        a = Gradient().in_node(init, Gradient.GRAD_HOR_RIGHT).out_node()
        b = Gradient().in_node(init, Gradient.GRAD_RAD_OUT).out_node()
        c = Gradient().in_node(init, Gradient.GRAD_RAD_IN).out_node()

        out = Rotate().in_node(init, a, 0.5).out_node()
        npwrite("1.png", out)
        out = Rotate().in_node(init, b, 0.5).out_node()
        npwrite("2.png", out)
        out = Rotate().in_node(init, c, 0.5).out_node()
        npwrite("3.png", out)

        # TODO: write test
        print("Rotate test case not finished")
        assert False
