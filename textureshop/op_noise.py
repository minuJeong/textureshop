import math

import moderngl as mg
import numpy as np

from .op_base import Base
from .util import cpu_noise


class CPURandom(Base):
    """ simple uniform random (using CPU) """

    @Base.in_node_wrapper
    def in_node(self, in_node, min_value=0.0, max_value=1.0):
        self.min_value, self.max_value = min_value, max_value

    @Base.out_node_wrapper
    def out_node(self):
        min_v, max_v = self.min_value, self.max_value
        size = (self.W, self.H, 4)
        data = np.random.uniform(min_v, max_v, size)
        data = data.astype(np.float32)
        return data


class FBMNoise(Base):
    """ fractional brownian motion noise """

    def set_noisetex(self, noise_tex, bytes_size=None):
        if isinstance(noise_tex, type(mg.texture)):
            self.u_noise_tex = noise_tex
            return

        if isinstance(noise_tex, (np.ndarray)):
            s = noise_tex.shape
            size = (s[0], s[1])
            channels = 1
            if len(s) > 2:
                channels = s[2]

            data = noise_tex.astype(np.float32).tobytes()
            self.u_noise_tex = self.gl.texture(size, channels, data, dtype="f4")
            return

        if isinstance(noise_tex, (bytes, bytearray)):
            if not bytes_size:
                raise Exception("[FBM Noise] noise_tex coming in with bytes, but size not specified")
                return
            self.u_noise_tex = self.gl.texture(bytes_size, 4, noise_tex, dtype="f4")
            return

        raise NotImplementedError(
            "setting noise from {} is not implemented".format(type(noise_tex)))

    @Base.in_node_wrapper
    def in_node(self, in_node, noise_tex=None, num_octaves=5):
        cs_path = "./gl/fbm_noise.glsl"
        self.cs = self.get_cs(cs_path)
        self.cs["u_octaves"].value = num_octaves

        if noise_tex is not None:
            self.set_noisetex(noise_tex)

        if not hasattr(self, "u_noise_tex"):
            print("[FBM] gen cpu noise as base..")
            cpu_noise_data = cpu_noise(self.W, self.H)
            self.set_noisetex(cpu_noise_data, (self.W, self.H))

        out_buffer = np.zeros((self.W, self.H, 4))
        out_buffer = out_buffer.astype(np.float32)
        self.cs_out = self.gl.buffer(out_buffer.tobytes())

    @Base.out_node_wrapper
    def out_node(self):
        self.u_noise_tex.use(0)

        self.cs_out.bind_to_storage_buffer(0)

        gx, gy = math.ceil(self.W / 32), math.ceil(self.H / 32)
        self.cs.run(gx, gy)

        return self.cs_out.read()


class GaussianBlur(Base):
    """ [TODO] Work In Progress """

    @Base.in_node_wrapper
    def in_node(self, in_node, values, deviation_h=5, deviation_v=5):
        pass

    @Base.out_node_wrapper
    def out_node(self):
        return []


class SumSineWave(Base):
    """ [TODO] Work In Progress """

    @Base.in_node_wrapper
    def in_node(self, in_node, octaves=12):
        cs_path = "./gl/sumsinewave.glsl"
        self.cs = self.get_cs(cs_path)

        if "u_octaves" in self.cs:
            self.cs["u_octaves"].value = octaves

        self.out_height = np.zeros((self.W, self.H, 4))
        self.out_height = self.out_height.astype(np.float32)
        self.out_height = self.gl.buffer(self.out_height.tobytes())

    @Base.out_node_wrapper
    def out_node(self):

        gx, gy = math.ceil(self.W / 32), math.ceil(self.H / 32)
        self.cs.run(gx, gy)

        return self.out_height.read()


class Gradient(Base):
    """ simple gradient """

    # available gradient types
    GRAD_HOR_LEFT = 0
    GRAD_HOR_RIGHT = 1
    GRAD_VER_UP = 2
    GRAD_VER_DOWN = 3
    GRAD_RAD_IN = 4
    GRAD_RAD_OUT = 5

    @Base.in_node_wrapper
    def in_node(self, in_node, grad_type=GRAD_HOR_LEFT):
        _grad = None

        if grad_type == Gradient.GRAD_HOR_LEFT:
            _grad = "_horizontal_left_grid"

        elif grad_type == Gradient.GRAD_HOR_RIGHT:
            _grad = "_horizontal_right_grid"

        elif grad_type == Gradient.GRAD_VER_UP:
            _grad = "_vertical_up_grid"

        elif grad_type == Gradient.GRAD_VER_DOWN:
            _grad = "_vertical_down_grid"

        elif grad_type == Gradient.GRAD_RAD_IN:
            _grad = "_radial_in_grid"

        elif grad_type == Gradient.GRAD_RAD_OUT:
            _grad = "_radial_out_grid"

        else:
            raise Exception("gradient type: {} is not implemented".format(grad_type))

        cs_path = "./gl/gradient.glsl"
        self.cs = self.get_cs(cs_path, {"%TYPE%": _grad})

        out_buffer = np.zeros((self.W, self.H, 4))
        out_buffer = out_buffer.astype(np.float32)
        self.cs_out = self.gl.buffer(out_buffer.tobytes())

    @Base.out_node_wrapper
    def out_node(self):
        self.cs_out.bind_to_storage_buffer(0)
        gx, gy = math.ceil(self.W / 32), math.ceil(self.H / 32)
        self.cs.run(gx, gy)
        return self.cs_out.read()
