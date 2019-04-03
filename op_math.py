
import math

import numpy as np
import moderngl as mg

from base import Init
from base import Base
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


if __name__ == "__main__":

    import unittest

    GL = mg.create_standalone_context()
    init = Init(size=(1, 1), gl=GL)

    class InitializeTest(unittest.TestCase):

        def test_num(self):
            print("[+] Testing const num")

            num_a = Num(value=0.1).in_node(init).out_node()
            self.assertEqual(num_a[0, 0, 0], np.float32(0.1))

            num_b = Num().in_node(init, 10.0).out_node()
            self.assertEqual(num_b[0, 0, 0], np.float32(10.0))

            num_c = Num(value=0.5).out_node()
            self.assertEqual(num_c[0, 0, 0], np.float32(0.5))

            self.assertIsInstance(num_a, np.ndarray)
            self.assertIsInstance(num_b, np.ndarray)
            self.assertIsInstance(num_c, np.ndarray)

        def test_init(self):
            print("[+] Testing init")

            num_a = Num().in_node(init, 0.1).out_node()
            self.assertEqual(num_a.shape, (1, 1, 4))

            num_b = Num(value=0.1).in_node(init).out_node()
            self.assertEqual(num_b[0, 0, 0], np.float32(0.1))

            self.assertIsInstance(num_a, np.ndarray)
            self.assertIsInstance(num_b, np.ndarray)

    class OperationTest(unittest.TestCase):

        def test_add(self):
            print("[+] Testing add node")

            num_a = Num(value=0.1).in_node(init).out_node()
            num_b = Num(value=0.2).in_node(init).out_node()

            add_a = Add().in_node(init, num_a, num_b).out_node()
            self.assertEqual(add_a.shape, (1, 1, 4))
            self.assertTrue(np.all(add_a[0, 0] == 0.3))

            add_b = Add().in_node(init, num_a, 0.3).out_node()
            self.assertEqual(add_b[0, 0, 0], np.float32(0.4))

            add_c = Add().in_node(init, 0.2, 0.6).out_node()
            self.assertEqual(add_c[0, 0, 0], np.float32(0.8))

            self.assertIsInstance(num_a, np.ndarray)
            self.assertIsInstance(num_b, np.ndarray)
            self.assertIsInstance(add_a, np.ndarray)
            self.assertIsInstance(add_b, np.ndarray)
            self.assertIsInstance(add_c, np.ndarray)

        def test_mul(self):
            print("[+] Testing mul node")

            num_a = Num(value=0.5).in_node(init).out_node()
            num_b = Num(value=10.0).in_node(init).out_node()

            mult = Multiply().in_node(init, num_a, num_b).out_node()
            self.assertEqual(mult[0, 0, 0], np.float32(5.0))

            self.assertIsInstance(num_a, np.ndarray)
            self.assertIsInstance(num_b, np.ndarray)
            self.assertIsInstance(mult, np.ndarray)

        def test_div(self):
            print("[+] Testing div node")

            div_a = Divide().in_node(init, 0.6, 0.2).out_node()
            self.assertEqual(div_a[0, 0, 0], np.float32(3.0))

            num_a = Num(value=0.5).out_node()
            num_b = Num(value=2.5).out_node()
            div_b = Divide().in_node(init, num_a, num_b).out_node()
            self.assertEqual(div_b[0, 0, 0], np.float32(0.2))

            div_c = Divide().in_node(init, 0.6, 0.2).out_node()
            self.assertEqual(div_c[0, 0, 0], np.float32(3.0))

            self.assertIsInstance(num_a, np.ndarray)
            self.assertIsInstance(num_b, np.ndarray)
            self.assertIsInstance(div_a, np.ndarray)
            self.assertIsInstance(div_b, np.ndarray)
            self.assertIsInstance(div_c, np.ndarray)

        def test_clamp(self):
            print("[+] Testing clamp node")

            clamped_a = Clamp().in_node(init, 1.2).out_node()
            self.assertEqual(clamped_a[0, 0, 0], np.float32(1.0))
            self.assertIsInstance(clamped_a, np.ndarray)

        def test_oneminus(self):
            print("[+] Testing oneminus node")

            oneminus_a = OneMinus().in_node(init, 0.25).out_node()
            self.assertEqual(oneminus_a[0, 0, 0], np.float32(0.75))
            oneminus_b = OneMinus().in_node(init, oneminus_a).out_node()
            self.assertEqual(oneminus_b[0, 0, 0], np.float32(0.25))

            self.assertIsInstance(oneminus_a, np.ndarray)
            self.assertIsInstance(oneminus_b, np.ndarray)

        def test_basic_trigonometries(self):
            print("[+] Testing basic trigonometries node")

            # accpet error less than 0.0001
            PATIENCE = 1e-4

            print("\t[+][+] Testing sin..")
            sin_a = Sin().in_node(init, 0.00).out_node()
            sin_b = Sin().in_node(init, 0.75).out_node()
            sin_c = Sin().in_node(init, 1.50).out_node()
            sin_d = Sin().in_node(init, 222.0).out_node()
            self.assertLess(abs(sin_a[0, 0, 0] - math.sin(0.00)), PATIENCE)
            self.assertLess(abs(sin_b[0, 0, 0] - math.sin(0.75)), PATIENCE)
            self.assertLess(abs(sin_c[0, 0, 0] - math.sin(1.50)), PATIENCE)
            self.assertLess(abs(sin_d[0, 0, 0] - math.sin(222.0)), PATIENCE)

            print("\t[+][+] Testing cos..")
            cos_a = Cos().in_node(init, 0.00).out_node()
            cos_b = Cos().in_node(init, 0.75).out_node()
            cos_c = Cos().in_node(init, 1.50).out_node()
            cos_d = Cos().in_node(init, 222.0).out_node()
            self.assertLess(abs(cos_a[0, 0, 0] - math.cos(0.00)), PATIENCE)
            self.assertLess(abs(cos_b[0, 0, 0] - math.cos(0.75)), PATIENCE)
            self.assertLess(abs(cos_c[0, 0, 0] - math.cos(1.50)), PATIENCE)
            self.assertLess(abs(cos_d[0, 0, 0] - math.cos(222.0)), PATIENCE)

            print("\t[+][+] Testing tan..")
            tan_a = Tan().in_node(init, 0.00).out_node()
            tan_b = Tan().in_node(init, 0.75).out_node()
            tan_c = Tan().in_node(init, 1.50).out_node()
            tan_d = Tan().in_node(init, 222.0).out_node()
            self.assertLess(abs(tan_a[0, 0, 0] - math.tan(0.00)), PATIENCE)
            self.assertLess(abs(tan_b[0, 0, 0] - math.tan(0.75)), PATIENCE)
            self.assertLess(abs(tan_c[0, 0, 0] - math.tan(1.50)), PATIENCE)
            self.assertLess(abs(tan_d[0, 0, 0] - math.tan(222.0)), PATIENCE)

        def test_arc_trigonometries(self):
            print("[+] Testing arc trigonometries node")

            # accpet error less than 0.0001
            PATIENCE = 1e-4

            print("[+][+] Testing asin..")
            asin_a = Asin().in_node(init, 1.0).out_node()
            asin_b = Asin().in_node(init, 0.1).out_node()
            asin_c = Asin().in_node(init, -1.0).out_node()
            asin_d = Asin().in_node(init, -0.5).out_node()

            self.assertLess(abs(asin_a[0, 0, 0] - math.asin(1.0)), PATIENCE)
            self.assertLess(abs(asin_b[0, 0, 0] - math.asin(0.1)), PATIENCE)
            self.assertLess(abs(asin_c[0, 0, 0] - math.asin(-1.0)), PATIENCE)
            self.assertLess(abs(asin_d[0, 0, 0] - math.asin(-0.5)), PATIENCE)

            print("\t[+][+] Testing acos..")
            acos_a = Acos().in_node(init, 1.0).out_node()
            acos_b = Acos().in_node(init, 0.1).out_node()
            acos_c = Acos().in_node(init, -1.0).out_node()
            acos_d = Acos().in_node(init, -0.5).out_node()

            self.assertLess(abs(acos_a[0, 0, 0] - math.acos(1.0)), PATIENCE)
            self.assertLess(abs(acos_b[0, 0, 0] - math.acos(0.1)), PATIENCE)
            self.assertLess(abs(acos_c[0, 0, 0] - math.acos(-1.0)), PATIENCE)
            self.assertLess(abs(acos_d[0, 0, 0] - math.acos(-0.5)), PATIENCE)

            print("\t[+][+] Testing atan2..")
            atan2_a = Atan2().in_node(init, 1.0, 2.0).out_node()
            atan2_b = Atan2().in_node(init, 0.1, 23.2).out_node()
            atan2_c = Atan2().in_node(init, 10.5, 0.11).out_node()
            atan2_d = Atan2().in_node(init, -100.0, 0.0).out_node()

            self.assertLess(abs(atan2_a[0, 0, 0] - math.atan2(1.0, 2.0)), PATIENCE)
            self.assertLess(abs(atan2_b[0, 0, 0] - math.atan2(0.1, 23.2)), PATIENCE)
            self.assertLess(abs(atan2_c[0, 0, 0] - math.atan2(10.5, 0.11)), PATIENCE)
            self.assertLess(abs(atan2_d[0, 0, 0] - math.atan2(-100.0, 0.0)), PATIENCE)

        def test_hyperbolic_trigonometries(self):
            print("[+] Testing hyperbolic trigonometries node")

            # accpet error less than 0.0001
            PATIENCE = 1e-4

            print("\t[+][+] Testing sinh..")
            sinh_a = SinH().in_node(init, 1.0).out_node()
            sinh_b = SinH().in_node(init, 0.1).out_node()
            sinh_c = SinH().in_node(init, 2.0).out_node()

            self.assertLess(abs(sinh_a[0, 0, 0] - math.sinh(1.0)), PATIENCE)
            self.assertLess(abs(sinh_b[0, 0, 0] - math.sinh(0.1)), PATIENCE)
            self.assertLess(abs(sinh_c[0, 0, 0] - math.sinh(2.0)), PATIENCE)

            print("\t[+][+] Testing cosh..")
            cosh_a = CosH().in_node(init, 1.0).out_node()
            cosh_b = CosH().in_node(init, 0.1).out_node()
            cosh_c = CosH().in_node(init, 2.0).out_node()

            self.assertLess(abs(cosh_a[0, 0, 0] - math.cosh(1.0)), PATIENCE)
            self.assertLess(abs(cosh_b[0, 0, 0] - math.cosh(0.1)), PATIENCE)
            self.assertLess(abs(cosh_c[0, 0, 0] - math.cosh(2.0)), PATIENCE)

            print("\t[+][+] Testing tanh..")
            tanh_a = TanH().in_node(init, 1.0).out_node()
            tanh_b = TanH().in_node(init, 0.1).out_node()
            tanh_c = TanH().in_node(init, 2.0).out_node()

            self.assertLess(abs(tanh_a[0, 0, 0] - math.tanh(1.0)), PATIENCE)
            self.assertLess(abs(tanh_b[0, 0, 0] - math.tanh(0.1)), PATIENCE)
            self.assertLess(abs(tanh_c[0, 0, 0] - math.tanh(2.0)), PATIENCE)

        def test_power_log(self):
            print("[+] Testing power/log node")

            pow_a = Power().in_node(init, 2.0, 2.0).out_node()
            pow_b = Power().in_node(init, 1.1, 10.0).out_node()
            pow_c = Power().in_node(init, 2.0, 10.0).out_node()
            pow_d = Power().in_node(init, 5.3, 2.1).out_node()

            self.assertTrue(np.all(np.isclose(pow_a, np.power(2.0, 2.0))))
            self.assertTrue(np.all(np.isclose(pow_b, np.power(1.1, 10.0))))
            self.assertTrue(np.all(np.isclose(pow_c, np.power(2.0, 10.0))))
            self.assertTrue(np.all(np.isclose(pow_d, np.power(5.3, 2.1))))

            log_a = Log_Natural().in_node(init, 12.53).out_node()
            log_b = Log_Natural().in_node(init, 22.23).out_node()

            self.assertTrue(np.all(np.isclose(log_a, np.log(12.53))))
            self.assertTrue(np.all(np.isclose(log_b, np.log(22.23))))

            log2_a = Log_2().in_node(init, 12.53).out_node()
            log2_b = Log_2().in_node(init, 22.23).out_node()

            self.assertTrue(np.all(np.isclose(log2_a, np.log2(12.53))))
            self.assertTrue(np.all(np.isclose(log2_b, np.log2(22.23))))

    unittest.main()
