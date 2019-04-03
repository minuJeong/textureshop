
import math

import numpy as np
import moderngl as mg

from base import Base


def _value_to_ndarray(value, w, h):
    converted = None
    if isinstance(value, (np.ndarray)):
        converted = value

    elif isinstance(value, (float, int)):
        value = float(value)
        arr = np.ones((w, h, 4))
        converted = np.multiply(arr, value)

    else:
        raise NotADirectoryError(
            "[Clamp] given value type {} is not implemented".format(type(value)))

    return converted


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

        cs_path = "./gl/add.glsl"
        self.cs = self.get_cs(cs_path)

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

        cs_path = "./gl/mul.glsl"
        self.cs = self.get_cs(cs_path)

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

        cs_path = "./gl/div.glsl"
        self.cs = self.get_cs(cs_path)

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

        cs_path = "./gl/clamp.glsl"
        self.cs = self.get_cs(cs_path)

        return self

    def out_node(self):
        out_buffer = np.zeros((self.W, self.H, 4))
        out_buffer = out_buffer.astype(np.float32)
        self.cs_out = self.gl.buffer(out_buffer.tobytes())
        self.cs_out.bind_to_storage_buffer(0)

        in_buffer = self.value.astype(np.float32)
        in_buffer = self.gl.buffer(in_buffer.tobytes())
        in_buffer.bind_to_storage_buffer(1)

        self.cs["u_min_value"].value = self.min_value
        self.cs["u_max_value"].value = self.max_value

        gx, gy = math.ceil(self.W / 32), math.ceil(self.H / 32)
        self.cs.run(gx, gy)

        data = self.cs_out.read()
        data = np.frombuffer(data, dtype="f4")
        data = data.reshape((self.W, self.H, 4))
        return data


if __name__ == "__main__":

    import unittest

    from base import Init

    GL = mg.create_standalone_context()
    init = Init(size=(1, 1), gl=GL)

    class InitializeTest(unittest.TestCase):

        def test_num(self):
            num_a = Num(value=0.1).in_node(init).out_node()
            self.assertEqual(num_a[0, 0, 0], np.float32(0.1))

            num_b = Num().in_node(init, 10.0).out_node()
            self.assertEqual(num_b[0, 0, 0], np.float32(10.0))

            num_c = Num(value=0.5).out_node()
            self.assertEqual(num_c[0, 0, 0], np.float32(0.5))

        def test_init(self):
            num_a = Num().in_node(init, 0.1).out_node()
            self.assertEqual(num_a.shape, (1, 1, 4))

            num_b = Num(value=0.1).in_node(init).out_node()
            self.assertEqual(num_b[0, 0, 0], np.float32(0.1))

    class OperationTest(unittest.TestCase):

        def test_add(self):
            num_a = Num(value=0.1).in_node(init).out_node()
            num_b = Num(value=0.2).in_node(init).out_node()

            add_a = Add().in_node(init, num_a, num_b).out_node()
            self.assertEqual(add_a.shape, (1, 1, 4))
            self.assertTrue(np.all(add_a[0, 0] == 0.3))

            add_b = Add().in_node(init, num_a, 0.3).out_node()
            self.assertEqual(add_b[0, 0, 0], np.float32(0.4))

            add_c = Add().in_node(init, 0.2, 0.6).out_node()
            self.assertEqual(add_c[0, 0, 0], np.float32(0.8))

        def test_mul(self):
            num_a = Num(value=0.5).in_node(init).out_node()
            num_b = Num(value=10.0).in_node(init).out_node()

            mult = Multiply().in_node(init, num_a, num_b).out_node()
            self.assertEqual(mult[0, 0, 0], np.float32(5.0))

        def test_div(self):
            div_a = Divide().in_node(init, 0.6, 0.2).out_node()
            self.assertEqual(div_a[0, 0, 0], np.float32(3.0))

            num_a = Num(value=0.5).out_node()
            num_b = Num(value=2.5).out_node()
            div_b = Divide().in_node(init, num_a, num_b).out_node()
            self.assertEqual(div_b[0, 0, 0], np.float32(0.2))

            div_c = Divide().in_node(init, 0.6, 0.2).out_node()
            self.assertEqual(div_c[0, 0, 0], np.float32(3.0))

        def test_clamp(self):
            clamped_a = Clamp().in_node(init, 1.2).out_node()
            self.assertEqual(clamped_a[0, 0, 0], np.float32(1.0))

    unittest.main()
