
import math

import numpy as np

from op_base import Base
from util import _value_to_ndarray


class Num(Base):
    """ simple number """

    def __init__(self, size=(512, 512), gl=None, value=0.0):
        super(Num, self).__init__()
        self.W, self.H = size[0], size[1]
        self.value = value

    def in_node(self, in_node, value=None):
        self.W, self.H = in_node.W, in_node.H

        if value is not None:
            self.value = value

        return self

    def out_node(self):
        data = np.ones((self.W, self.H, 4))
        data = data.astype(np.float32)
        data = np.multiply(data, self.value)
        return data


class Add(Base):
    """ simple Add """

    def in_node(self, in_node, in_a, in_b):
        self.W, self.H = in_node.W, in_node.H
        self.in_a = _value_to_ndarray(in_a, self.W, self.H)
        self.in_b = _value_to_ndarray(in_b, self.W, self.H)

        cs_path = "./gl/math.glsl"
        self.cs = self.get_cs(cs_path, {"%CALC%": "_add"})

        return self

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

        data = self.cs_out.read()
        data = np.frombuffer(data, dtype="f4")
        data = data.reshape((self.W, self.H, 4))
        return data


class Multiply(Base):
    """ simple Multiply """

    def in_node(self, in_node, in_a, in_b):
        self.W, self.H = in_node.W, in_node.H
        self.in_a = _value_to_ndarray(in_a, self.W, self.H)
        self.in_b = _value_to_ndarray(in_b, self.W, self.H)

        cs_path = "./gl/math.glsl"
        self.cs = self.get_cs(cs_path, {"%CALC%": "_mul"})

        return self

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

        data = self.cs_out.read()
        data = np.frombuffer(data, dtype="f4")
        data = data.reshape((self.W, self.H, 4))
        return data


class Divide(Base):
    """ simple Divide """

    def in_node(self, in_node, in_a, in_b):
        self.W, self.H = in_node.W, in_node.H
        self.in_a = _value_to_ndarray(in_a, self.W, self.H)
        self.in_b = _value_to_ndarray(in_b, self.W, self.H)

        cs_path = "./gl/math.glsl"
        self.cs = self.get_cs(cs_path, {"%CALC%": "_div"})

        return self

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

        data = self.cs_out.read()
        data = np.frombuffer(data, dtype="f4")
        data = data.reshape((self.W, self.H, 4))
        return data


class Clamp(Base):
    """ x = max(min(x, MAX), MIN) """

    def in_node(self, in_node, value, min_value=0.0, max_value=1.0):
        self.W, self.H = in_node.W, in_node.H

        self.value = _value_to_ndarray(value, self.W, self.H)

        self.min_value = min_value
        self.max_value = max_value

        cs_path = "./gl/math.glsl"
        self.cs = self.get_cs(cs_path, {"%CALC%": "_clamp"})

        return self

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

        data = self.cs_out.read()
        data = np.frombuffer(data, dtype="f4")
        data = data.reshape((self.W, self.H, 4))
        return data


class OneMinus(Base):
    """ 1.0 - x """

    def in_node(self, in_node, value):
        self.W, self.H = in_node.W, in_node.H

        self.value = _value_to_ndarray(value, self.W, self.H)

        cs_path = "./gl/math.glsl"
        self.cs = self.get_cs(cs_path, {"%CALC%": "_oneminus"})

        return self

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

        data = self.cs_out.read()
        data = np.frombuffer(data, dtype="f4")
        data = data.reshape((self.W, self.H, 4))
        return data


class Sin(Base):
    """ sin(x) """

    def in_node(self, in_node, value):
        self.W, self.H = in_node.W, in_node.H

        self.value = _value_to_ndarray(value, self.W, self.H)

        cs_path = "./gl/math.glsl"
        self.cs = self.get_cs(cs_path, {"%CALC%": "_sin"})

        return self

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

        data = self.cs_out.read()
        data = np.frombuffer(data, dtype="f4")
        data = data.reshape((self.W, self.H, 4))
        return data


class Cos(Base):
    """ cos(x) """

    def in_node(self, in_node, value):
        self.W, self.H = in_node.W, in_node.H

        self.value = _value_to_ndarray(value, self.W, self.H)

        cs_path = "./gl/math.glsl"
        self.cs = self.get_cs(cs_path, {"%CALC%": "_cos"})

        return self

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

        data = self.cs_out.read()
        data = np.frombuffer(data, dtype="f4")
        data = data.reshape((self.W, self.H, 4))
        return data


class Tan(Base):
    """ tan(x) """

    def in_node(self, in_node, value):
        self.W, self.H = in_node.W, in_node.H

        self.value = _value_to_ndarray(value, self.W, self.H)

        cs_path = "./gl/math.glsl"
        self.cs = self.get_cs(cs_path, {"%CALC%": "_tan"})

        return self

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

        data = self.cs_out.read()
        data = np.frombuffer(data, dtype="f4")
        data = data.reshape((self.W, self.H, 4))
        return data


class Asin(Base):
    """ asin(x) """

    def in_node(self, in_node, value):
        self.W, self.H = in_node.W, in_node.H

        self.value = _value_to_ndarray(value, self.W, self.H)

        cs_path = "./gl/math.glsl"
        self.cs = self.get_cs(cs_path, {"%CALC%": "_asin"})

        return self

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

        data = self.cs_out.read()
        data = np.frombuffer(data, dtype="f4")
        data = data.reshape((self.W, self.H, 4))
        return data


class Acos(Base):
    """ acos(x) """

    def in_node(self, in_node, value):
        self.W, self.H = in_node.W, in_node.H

        self.value = _value_to_ndarray(value, self.W, self.H)

        cs_path = "./gl/math.glsl"
        self.cs = self.get_cs(cs_path, {"%CALC%": "_acos"})

        return self

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

        data = self.cs_out.read()
        data = np.frombuffer(data, dtype="f4")
        data = data.reshape((self.W, self.H, 4))
        return data


class Atan2(Base):
    """ atan2(y, x) """

    def in_node(self, in_node, in_a, in_b):
        self.W, self.H = in_node.W, in_node.H
        self.in_a = _value_to_ndarray(in_a, self.W, self.H)
        self.in_b = _value_to_ndarray(in_b, self.W, self.H)

        cs_path = "./gl/math.glsl"
        self.cs = self.get_cs(cs_path, {"%CALC%": "_atan2"})

        return self

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

        data = self.cs_out.read()
        data = np.frombuffer(data, dtype="f4")
        data = data.reshape((self.W, self.H, 4))
        return data


class SinH(Base):
    """ acos(x) """

    def in_node(self, in_node, value):
        self.W, self.H = in_node.W, in_node.H

        self.value = _value_to_ndarray(value, self.W, self.H)

        cs_path = "./gl/math.glsl"
        self.cs = self.get_cs(cs_path, {"%CALC%": "_sinh"})

        return self

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

        data = self.cs_out.read()
        data = np.frombuffer(data, dtype="f4")
        data = data.reshape((self.W, self.H, 4))
        return data


class CosH(Base):
    """ acos(x) """

    def in_node(self, in_node, value):
        self.W, self.H = in_node.W, in_node.H

        self.value = _value_to_ndarray(value, self.W, self.H)

        cs_path = "./gl/math.glsl"
        self.cs = self.get_cs(cs_path, {"%CALC%": "_cosh"})

        return self

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

        data = self.cs_out.read()
        data = np.frombuffer(data, dtype="f4")
        data = data.reshape((self.W, self.H, 4))
        return data


class TanH(Base):
    """ acos(x) """

    def in_node(self, in_node, value):
        self.W, self.H = in_node.W, in_node.H

        self.value = _value_to_ndarray(value, self.W, self.H)

        cs_path = "./gl/math.glsl"
        self.cs = self.get_cs(cs_path, {"%CALC%": "_tanh"})

        return self

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

        data = self.cs_out.read()
        data = np.frombuffer(data, dtype="f4")
        data = data.reshape((self.W, self.H, 4))
        return data


class Power(Base):
    """ pow(a, b) """

    def in_node(self, in_node, in_a, in_b):
        self.W, self.H = in_node.W, in_node.H
        self.in_a = _value_to_ndarray(in_a, self.W, self.H)
        self.in_b = _value_to_ndarray(in_b, self.W, self.H)

        cs_path = "./gl/math.glsl"
        self.cs = self.get_cs(cs_path, {"%CALC%": "_pow"})

        return self

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

        data = self.cs_out.read()
        data = np.frombuffer(data, dtype="f4")
        data = data.reshape((self.W, self.H, 4))
        return data


class Log_Natural(Base):
    """ log(a) """

    def in_node(self, in_node, in_a):
        self.W, self.H = in_node.W, in_node.H
        self.in_a = _value_to_ndarray(in_a, self.W, self.H)

        cs_path = "./gl/math.glsl"
        self.cs = self.get_cs(cs_path, {"%CALC%": "_log"})

        return self

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

        data = self.cs_out.read()
        data = np.frombuffer(data, dtype="f4")
        data = data.reshape((self.W, self.H, 4))
        return data


class Log_2(Base):
    """ log(a) """

    def in_node(self, in_node, in_a):
        self.W, self.H = in_node.W, in_node.H
        self.in_a = _value_to_ndarray(in_a, self.W, self.H)

        cs_path = "./gl/math.glsl"
        self.cs = self.get_cs(cs_path, {"%CALC%": "_log2"})

        return self

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

        data = self.cs_out.read()
        data = np.frombuffer(data, dtype="f4")
        data = data.reshape((self.W, self.H, 4))
        return data
