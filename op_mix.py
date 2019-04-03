
import math

import numpy as np
import moderngl as mg

from base import Init
from base import Base
from util import _value_to_ndarray


class Mix(Base):
    """ mix(a, b, k) """

    def in_node(self, in_node, in_a, in_b, in_c):
        self.W, self.H = in_node.W, in_node.H
        self.in_a = _value_to_ndarray(in_a, self.W, self.H)
        self.in_b = _value_to_ndarray(in_b, self.W, self.H)
        self.in_c = _value_to_ndarray(in_c, self.W, self.H)

        cs_path = "./gl/mix.glsl"
        self.cs = self.get_cs(cs_path, {"%CALC%": "_mix"})

        return self

    def out_node(self):
        a_buffer = self.in_a.astype(np.float32)
        b_buffer = self.in_b.astype(np.float32)
        c_buffer = self.in_c.astype(np.float32)

        out_buffer = np.zeros((self.W, self.H, 4))
        out_buffer = out_buffer.astype(np.float32)

        self.cs_in_a = self.gl.buffer(a_buffer.tobytes())
        self.cs_in_a.bind_to_storage_buffer(1)
        self.cs_in_b = self.gl.buffer(b_buffer.tobytes())
        self.cs_in_b.bind_to_storage_buffer(2)
        self.cs_in_c = self.gl.buffer(c_buffer.tobytes())
        self.cs_in_c.bind_to_storage_buffer(3)

        self.cs_out = self.gl.buffer(out_buffer.tobytes())
        self.cs_out.bind_to_storage_buffer(0)

        gx, gy = math.ceil(self.W / 32), math.ceil(self.H / 32)
        self.cs.run(gx, gy)

        data = self.cs_out.read()
        data = np.frombuffer(data, dtype="f4")
        data = data.reshape((self.W, self.H, 4))
        return data


class Smoothstep(Base):
    """ smoothstep(a, b, x) """

    def in_node(self, in_node, in_a, in_b, in_c):
        self.W, self.H = in_node.W, in_node.H
        self.in_a = _value_to_ndarray(in_a, self.W, self.H)
        self.in_b = _value_to_ndarray(in_b, self.W, self.H)
        self.in_c = _value_to_ndarray(in_c, self.W, self.H)

        cs_path = "./gl/mix.glsl"
        self.cs = self.get_cs(cs_path, {"%CALC%": "_smoothstep"})

        return self

    def out_node(self):
        a_buffer = self.in_a.astype(np.float32)
        b_buffer = self.in_b.astype(np.float32)
        c_buffer = self.in_c.astype(np.float32)

        out_buffer = np.zeros((self.W, self.H, 4))
        out_buffer = out_buffer.astype(np.float32)

        self.cs_in_a = self.gl.buffer(a_buffer.tobytes())
        self.cs_in_a.bind_to_storage_buffer(1)
        self.cs_in_b = self.gl.buffer(b_buffer.tobytes())
        self.cs_in_b.bind_to_storage_buffer(2)
        self.cs_in_c = self.gl.buffer(c_buffer.tobytes())
        self.cs_in_c.bind_to_storage_buffer(3)

        self.cs_out = self.gl.buffer(out_buffer.tobytes())
        self.cs_out.bind_to_storage_buffer(0)

        gx, gy = math.ceil(self.W / 32), math.ceil(self.H / 32)
        self.cs.run(gx, gy)

        data = self.cs_out.read()
        data = np.frombuffer(data, dtype="f4")
        data = data.reshape((self.W, self.H, 4))
        return data


if __name__ == "__main__":

    import unittest

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

            from noise import FBMNoise
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

    unittest.main()
