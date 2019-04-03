import unittest

import numpy as np
import moderngl as mg

from op_base import Init
from op_mix import Mix
from op_mix import Smoothstep


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
        self.assertTrue(np.all(np.isclose(mix_a, np_mix_a, atol=PATIENCE)))

        from op_noise import FBMNoise
        noise = FBMNoise().in_node(init).out_node()
        mix_b = Mix().in_node(init, 0.0, 1.0, noise).out_node()
        self.assertTrue(np.all(np.isclose(noise, mix_b, atol=PATIENCE)))

        print("\t[+][+] Testing smoothstep..")
        smoothstep_a = Smoothstep().in_node(init, in_a, in_b, in_c).out_node()

        def numpy_smoothstep(a, b, c):
            t = np.clip((c - a) / (b - a), 0.0, 1.0)
            return t * t * (3.0 - 2.0 * t)
        np_smoothstep_a = numpy_smoothstep(in_a, in_b, in_c)
        self.assertTrue(np.all(np.isclose(smoothstep_a, np_smoothstep_a, atol=PATIENCE)))


if __name__ == "__main__":
    unittest.main()
