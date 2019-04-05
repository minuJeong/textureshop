import math

import numpy as np

from .op_base import Base
from .util import _value_to_ndarray


class Num(Base):
    """ simple number """

    def __init__(self, size=(512, 512), gl=None, value=0.0):
        super(Num, self).__init__()
        self.W, self.H = size[0], size[1]
        self.value = value

    @Base.in_node_wrapper
    def in_node(self, in_node, value=None):
        if value is not None:
            self.value = value

    @Base.out_node_wrapper
    def out_node(self):
        data = np.ones((self.W, self.H, 4))
        data = data.astype(np.float32)
        data = np.multiply(data, self.value)
        return data


class Add(Base):
    """ simple Add """

    @Base.in_node_wrapper
    def in_node(self, in_node, in_a, in_b):
        self.in_a = _value_to_ndarray(in_a, self.W, self.H)
        self.in_b = _value_to_ndarray(in_b, self.W, self.H)

        cs_path = "./gl/math.glsl"
        self.cs = self.get_cs(cs_path, {"%CALC%": "_add"})

    @Base.out_node_wrapper
    def out_node(self):
        a_buffer = self.in_a.astype(np.float32)
        b_buffer = self.in_b.astype(np.float32)
        out_buffer = np.zeros((self.W, self.H, 4))
        out_buffer = out_buffer.astype(np.float32)

        self.cs_in_a = self.gl.buffer(a_buffer.tobytes())
        self.cs_in_a.bind_to_storage_buffer(1)
        self.cs_in_b = self.gl.buffer(b_buffer.tobytes())
        self.cs_in_b.bind_to_storage_buffer(2)
        self.cs_out = self.gl.buffer(out_buffer.tobytes())
        self.cs_out.bind_to_storage_buffer(0)

        gx, gy = math.ceil(self.W / 32), math.ceil(self.H / 32)
        self.cs.run(gx, gy)

        return self.cs_out.read()


class Multiply(Base):
    """ simple Multiply """

    @Base.in_node_wrapper
    def in_node(self, in_node, in_a, in_b):
        self.in_a = _value_to_ndarray(in_a, self.W, self.H)
        self.in_b = _value_to_ndarray(in_b, self.W, self.H)

        cs_path = "./gl/math.glsl"
        self.cs = self.get_cs(cs_path, {"%CALC%": "_mul"})

    @Base.out_node_wrapper
    def out_node(self):
        a_buffer = self.in_a.astype(np.float32)
        b_buffer = self.in_b.astype(np.float32)
        out_buffer = np.zeros((self.W, self.H, 4))
        out_buffer = out_buffer.astype(np.float32)

        self.cs_in_a = self.gl.buffer(a_buffer.tobytes())
        self.cs_in_a.bind_to_storage_buffer(1)
        self.cs_in_b = self.gl.buffer(b_buffer.tobytes())
        self.cs_in_b.bind_to_storage_buffer(2)
        self.cs_out = self.gl.buffer(out_buffer.tobytes())
        self.cs_out.bind_to_storage_buffer(0)

        gx, gy = math.ceil(self.W / 32), math.ceil(self.H / 32)
        self.cs.run(gx, gy)

        return self.cs_out.read()


class Divide(Base):
    """ simple Divide """

    @Base.in_node_wrapper
    def in_node(self, in_node, in_a, in_b):
        self.in_a = _value_to_ndarray(in_a, self.W, self.H)
        self.in_b = _value_to_ndarray(in_b, self.W, self.H)

        cs_path = "./gl/math.glsl"
        self.cs = self.get_cs(cs_path, {"%CALC%": "_div"})

    @Base.out_node_wrapper
    def out_node(self):
        a_buffer = self.in_a.astype(np.float32)
        b_buffer = self.in_b.astype(np.float32)
        out_buffer = np.zeros((self.W, self.H, 4))
        out_buffer = out_buffer.astype(np.float32)

        self.cs_in_a = self.gl.buffer(a_buffer.tobytes())
        self.cs_in_a.bind_to_storage_buffer(1)
        self.cs_in_b = self.gl.buffer(b_buffer.tobytes())
        self.cs_in_b.bind_to_storage_buffer(2)
        self.cs_out = self.gl.buffer(out_buffer.tobytes())
        self.cs_out.bind_to_storage_buffer(0)

        gx, gy = math.ceil(self.W / 32), math.ceil(self.H / 32)
        self.cs.run(gx, gy)

        return self.cs_out.read()


class Clamp(Base):
    """ x = max(min(x, MAX), MIN) """

    @Base.in_node_wrapper
    def in_node(self, in_node, value, min_value=0.0, max_value=1.0):
        self.value = _value_to_ndarray(value, self.W, self.H)

        self.min_value = min_value
        self.max_value = max_value

        cs_path = "./gl/math.glsl"
        self.cs = self.get_cs(cs_path, {"%CALC%": "_clamp"})

    @Base.out_node_wrapper
    def out_node(self):
        out_buffer = np.zeros((self.W, self.H, 4))
        out_buffer = out_buffer.astype(np.float32)
        self.cs_out = self.gl.buffer(out_buffer.tobytes())
        self.cs_out.bind_to_storage_buffer(0)

        in_buffer = self.value.astype(np.float32)
        in_buffer = self.gl.buffer(in_buffer.tobytes())
        in_buffer.bind_to_storage_buffer(1)

        self.cs["u_clamp_min_value"].value = self.min_value
        self.cs["u_clamp_max_value"].value = self.max_value

        gx, gy = math.ceil(self.W / 32), math.ceil(self.H / 32)
        self.cs.run(gx, gy)

        return self.cs_out.read()


class OneMinus(Base):
    """ 1.0 - x """

    @Base.in_node_wrapper
    def in_node(self, in_node, value):
        self.value = _value_to_ndarray(value, self.W, self.H)

        cs_path = "./gl/math.glsl"
        self.cs = self.get_cs(cs_path, {"%CALC%": "_oneminus"})

    @Base.out_node_wrapper
    def out_node(self):
        out_buffer = np.zeros((self.W, self.H, 4))
        out_buffer = out_buffer.astype(np.float32)
        self.cs_out = self.gl.buffer(out_buffer.tobytes())
        self.cs_out.bind_to_storage_buffer(0)

        in_buffer = self.value.astype(np.float32)
        in_buffer = self.gl.buffer(in_buffer.tobytes())
        in_buffer.bind_to_storage_buffer(1)

        gx, gy = math.ceil(self.W / 32), math.ceil(self.H / 32)
        self.cs.run(gx, gy)

        return self.cs_out.read()


class Sin(Base):
    """ sin(x) """

    @Base.in_node_wrapper
    def in_node(self, in_node, value):
        self.value = _value_to_ndarray(value, self.W, self.H)

        cs_path = "./gl/math.glsl"
        self.cs = self.get_cs(cs_path, {"%CALC%": "_sin"})

    @Base.out_node_wrapper
    def out_node(self):
        out_buffer = np.zeros((self.W, self.H, 4))
        out_buffer = out_buffer.astype(np.float32)
        self.cs_out = self.gl.buffer(out_buffer.tobytes())
        self.cs_out.bind_to_storage_buffer(0)

        in_buffer = self.value.astype(np.float32)
        in_buffer = self.gl.buffer(in_buffer.tobytes())
        in_buffer.bind_to_storage_buffer(1)

        gx, gy = math.ceil(self.W / 32), math.ceil(self.H / 32)

        self.cs.run(gx, gy)

        return self.cs_out.read()


class Cos(Base):
    """ cos(x) """

    @Base.in_node_wrapper
    def in_node(self, in_node, value):
        self.value = _value_to_ndarray(value, self.W, self.H)

        cs_path = "./gl/math.glsl"
        self.cs = self.get_cs(cs_path, {"%CALC%": "_cos"})

    @Base.out_node_wrapper
    def out_node(self):
        out_buffer = np.zeros((self.W, self.H, 4))
        out_buffer = out_buffer.astype(np.float32)
        self.cs_out = self.gl.buffer(out_buffer.tobytes())
        self.cs_out.bind_to_storage_buffer(0)

        in_buffer = self.value.astype(np.float32)
        in_buffer = self.gl.buffer(in_buffer.tobytes())
        in_buffer.bind_to_storage_buffer(1)

        gx, gy = math.ceil(self.W / 32), math.ceil(self.H / 32)

        self.cs.run(gx, gy)

        return self.cs_out.read()


class Tan(Base):
    """ tan(x) """

    @Base.in_node_wrapper
    def in_node(self, in_node, value):
        self.value = _value_to_ndarray(value, self.W, self.H)

        cs_path = "./gl/math.glsl"
        self.cs = self.get_cs(cs_path, {"%CALC%": "_tan"})

    @Base.out_node_wrapper
    def out_node(self):
        out_buffer = np.zeros((self.W, self.H, 4))
        out_buffer = out_buffer.astype(np.float32)
        self.cs_out = self.gl.buffer(out_buffer.tobytes())
        self.cs_out.bind_to_storage_buffer(0)

        in_buffer = self.value.astype(np.float32)
        in_buffer = self.gl.buffer(in_buffer.tobytes())
        in_buffer.bind_to_storage_buffer(1)

        gx, gy = math.ceil(self.W / 32), math.ceil(self.H / 32)

        self.cs.run(gx, gy)

        return self.cs_out.read()


class Asin(Base):
    """ asin(x) """

    @Base.in_node_wrapper
    def in_node(self, in_node, value):
        self.value = _value_to_ndarray(value, self.W, self.H)

        cs_path = "./gl/math.glsl"
        self.cs = self.get_cs(cs_path, {"%CALC%": "_asin"})

    @Base.out_node_wrapper
    def out_node(self):
        out_buffer = np.zeros((self.W, self.H, 4))
        out_buffer = out_buffer.astype(np.float32)
        self.cs_out = self.gl.buffer(out_buffer.tobytes())
        self.cs_out.bind_to_storage_buffer(0)

        in_buffer = self.value.astype(np.float32)
        in_buffer = self.gl.buffer(in_buffer.tobytes())
        in_buffer.bind_to_storage_buffer(1)

        gx, gy = math.ceil(self.W / 32), math.ceil(self.H / 32)

        self.cs.run(gx, gy)

        return self.cs_out.read()


class Acos(Base):
    """ acos(x) """

    @Base.in_node_wrapper
    def in_node(self, in_node, value):
        self.value = _value_to_ndarray(value, self.W, self.H)

        cs_path = "./gl/math.glsl"
        self.cs = self.get_cs(cs_path, {"%CALC%": "_acos"})

    @Base.out_node_wrapper
    def out_node(self):
        out_buffer = np.zeros((self.W, self.H, 4))
        out_buffer = out_buffer.astype(np.float32)
        self.cs_out = self.gl.buffer(out_buffer.tobytes())
        self.cs_out.bind_to_storage_buffer(0)

        in_buffer = self.value.astype(np.float32)
        in_buffer = self.gl.buffer(in_buffer.tobytes())
        in_buffer.bind_to_storage_buffer(1)

        gx, gy = math.ceil(self.W / 32), math.ceil(self.H / 32)
        self.cs.run(gx, gy)

        return self.cs_out.read()


class Atan2(Base):
    """ atan2(y, x) """

    @Base.in_node_wrapper
    def in_node(self, in_node, in_a, in_b):
        self.in_a = _value_to_ndarray(in_a, self.W, self.H)
        self.in_b = _value_to_ndarray(in_b, self.W, self.H)

        cs_path = "./gl/math.glsl"
        self.cs = self.get_cs(cs_path, {"%CALC%": "_atan2"})

    @Base.out_node_wrapper
    def out_node(self):
        a_buffer = self.in_a.astype(np.float32)
        b_buffer = self.in_b.astype(np.float32)
        out_buffer = np.zeros((self.W, self.H, 4))
        out_buffer = out_buffer.astype(np.float32)

        self.cs_in_a = self.gl.buffer(a_buffer.tobytes())
        self.cs_in_a.bind_to_storage_buffer(1)
        self.cs_in_b = self.gl.buffer(b_buffer.tobytes())
        self.cs_in_b.bind_to_storage_buffer(2)
        self.cs_out = self.gl.buffer(out_buffer.tobytes())
        self.cs_out.bind_to_storage_buffer(0)

        gx, gy = math.ceil(self.W / 32), math.ceil(self.H / 32)
        self.cs.run(gx, gy)

        return self.cs_out.read()


class SinH(Base):
    """ acos(x) """

    @Base.in_node_wrapper
    def in_node(self, in_node, value):
        self.value = _value_to_ndarray(value, self.W, self.H)

        cs_path = "./gl/math.glsl"
        self.cs = self.get_cs(cs_path, {"%CALC%": "_sinh"})

    @Base.out_node_wrapper
    def out_node(self):
        out_buffer = np.zeros((self.W, self.H, 4))
        out_buffer = out_buffer.astype(np.float32)
        self.cs_out = self.gl.buffer(out_buffer.tobytes())
        self.cs_out.bind_to_storage_buffer(0)

        in_buffer = self.value.astype(np.float32)
        in_buffer = self.gl.buffer(in_buffer.tobytes())
        in_buffer.bind_to_storage_buffer(1)

        gx, gy = math.ceil(self.W / 32), math.ceil(self.H / 32)

        self.cs.run(gx, gy)

        return self.cs_out.read()


class CosH(Base):
    """ acos(x) """

    @Base.in_node_wrapper
    def in_node(self, in_node, value):
        self.value = _value_to_ndarray(value, self.W, self.H)

        cs_path = "./gl/math.glsl"
        self.cs = self.get_cs(cs_path, {"%CALC%": "_cosh"})

    @Base.out_node_wrapper
    def out_node(self):
        out_buffer = np.zeros((self.W, self.H, 4))
        out_buffer = out_buffer.astype(np.float32)
        self.cs_out = self.gl.buffer(out_buffer.tobytes())
        self.cs_out.bind_to_storage_buffer(0)

        in_buffer = self.value.astype(np.float32)
        in_buffer = self.gl.buffer(in_buffer.tobytes())
        in_buffer.bind_to_storage_buffer(1)

        gx, gy = math.ceil(self.W / 32), math.ceil(self.H / 32)

        self.cs.run(gx, gy)

        return self.cs_out.read()


class TanH(Base):
    """ acos(x) """

    @Base.in_node_wrapper
    def in_node(self, in_node, value):
        self.value = _value_to_ndarray(value, self.W, self.H)

        cs_path = "./gl/math.glsl"
        self.cs = self.get_cs(cs_path, {"%CALC%": "_tanh"})

    @Base.out_node_wrapper
    def out_node(self):
        out_buffer = np.zeros((self.W, self.H, 4))
        out_buffer = out_buffer.astype(np.float32)
        self.cs_out = self.gl.buffer(out_buffer.tobytes())
        self.cs_out.bind_to_storage_buffer(0)

        in_buffer = self.value.astype(np.float32)
        in_buffer = self.gl.buffer(in_buffer.tobytes())
        in_buffer.bind_to_storage_buffer(1)

        gx, gy = math.ceil(self.W / 32), math.ceil(self.H / 32)

        self.cs.run(gx, gy)

        return self.cs_out.read()


class Power(Base):
    """ pow(a, b) """

    @Base.in_node_wrapper
    def in_node(self, in_node, in_a, in_b):
        self.in_a = _value_to_ndarray(in_a, self.W, self.H)
        self.in_b = _value_to_ndarray(in_b, self.W, self.H)

        cs_path = "./gl/math.glsl"
        self.cs = self.get_cs(cs_path, {"%CALC%": "_pow"})

    @Base.out_node_wrapper
    def out_node(self):
        a_buffer = self.in_a.astype(np.float32)
        b_buffer = self.in_b.astype(np.float32)
        out_buffer = np.zeros((self.W, self.H, 4))
        out_buffer = out_buffer.astype(np.float32)

        self.cs_in_a = self.gl.buffer(a_buffer.tobytes())
        self.cs_in_a.bind_to_storage_buffer(1)
        self.cs_in_b = self.gl.buffer(b_buffer.tobytes())
        self.cs_in_b.bind_to_storage_buffer(2)
        self.cs_out = self.gl.buffer(out_buffer.tobytes())
        self.cs_out.bind_to_storage_buffer(0)

        gx, gy = math.ceil(self.W / 32), math.ceil(self.H / 32)
        self.cs.run(gx, gy)

        return self.cs_out.read()


class Log_Natural(Base):
    """ log(a) """

    @Base.in_node_wrapper
    def in_node(self, in_node, in_a):
        self.in_a = _value_to_ndarray(in_a, self.W, self.H)

        cs_path = "./gl/math.glsl"
        self.cs = self.get_cs(cs_path, {"%CALC%": "_log"})

    @Base.out_node_wrapper
    def out_node(self):
        a_buffer = self.in_a.astype(np.float32)
        out_buffer = np.zeros((self.W, self.H, 4))
        out_buffer = out_buffer.astype(np.float32)

        self.cs_in_a = self.gl.buffer(a_buffer.tobytes())
        self.cs_in_a.bind_to_storage_buffer(1)
        self.cs_out = self.gl.buffer(out_buffer.tobytes())
        self.cs_out.bind_to_storage_buffer(0)

        gx, gy = math.ceil(self.W / 32), math.ceil(self.H / 32)
        self.cs.run(gx, gy)

        return self.cs_out.read()


class Log_2(Base):
    """ log(a) """

    @Base.in_node_wrapper
    def in_node(self, in_node, in_a):
        self.in_a = _value_to_ndarray(in_a, self.W, self.H)

        cs_path = "./gl/math.glsl"
        self.cs = self.get_cs(cs_path, {"%CALC%": "_log2"})

    @Base.out_node_wrapper
    def out_node(self):
        a_buffer = self.in_a.astype(np.float32)
        out_buffer = np.zeros((self.W, self.H, 4))
        out_buffer = out_buffer.astype(np.float32)

        self.cs_in_a = self.gl.buffer(a_buffer.tobytes())
        self.cs_in_a.bind_to_storage_buffer(1)
        self.cs_out = self.gl.buffer(out_buffer.tobytes())
        self.cs_out.bind_to_storage_buffer(0)

        gx, gy = math.ceil(self.W / 32), math.ceil(self.H / 32)
        self.cs.run(gx, gy)

        return self.cs_out.read()
