import math

import numpy as np

from op_base import Base
from util import _value_to_ndarray


class Mix(Base):
    """ mix(a, b, k) """

    @Base.in_node_wrapper
    def in_node(self, in_node, in_a, in_b, in_c):
        self.in_a = _value_to_ndarray(in_a, self.W, self.H)
        self.in_b = _value_to_ndarray(in_b, self.W, self.H)
        self.in_c = _value_to_ndarray(in_c, self.W, self.H)

        cs_path = "./gl/mix.glsl"
        self.cs = self.get_cs(cs_path, {"%CALC%": "_mix"})

    @Base.out_node_wrapper
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

        return self.cs_out.read()


class Smoothstep(Base):
    """ smoothstep(a, b, x) """

    @Base.in_node_wrapper
    def in_node(self, in_node, in_a, in_b, in_c):
        self.in_a = _value_to_ndarray(in_a, self.W, self.H)
        self.in_b = _value_to_ndarray(in_b, self.W, self.H)
        self.in_c = _value_to_ndarray(in_c, self.W, self.H)

        cs_path = "./gl/mix.glsl"
        self.cs = self.get_cs(cs_path, {"%CALC%": "_smoothstep"})

    @Base.out_node_wrapper
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

        return self.cs_out.read()
