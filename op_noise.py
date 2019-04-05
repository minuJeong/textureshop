import math

import moderngl as mg
import numpy as np

from op_base import Base
from util import cpu_noise


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
            print("[FBM Noise] noise texture not coming in, generate cpu noise as base.")
            cpu_noise_data = cpu_noise(self.W, self.H)
            self.set_noisetex(cpu_noise_data, (self.W, self.H))

    @Base.out_node_wrapper
    def out_node(self):
        self.u_noise_tex.use(0)

        out_buffer = np.zeros((self.W, self.H, 4))
        out_buffer = out_buffer.astype(np.float32)

        self.cs_out = self.gl.buffer(out_buffer.tobytes())
        self.cs_out.bind_to_storage_buffer(0)

        gx, gy = math.ceil(self.W / 32), math.ceil(self.H / 32)
        self.cs.run(gx, gy)

        return self.cs_out.read()


class GaussianBlur(Base):

    @Base.in_node_wrapper
    def in_node(self, in_node, values, deviation_h=5, deviation_v=5):
        pass

    @Base.out_node_wrapper
    def out_node(self):
        return []


class SumSineWave(Base):

    @Base.in_node_wrapper
    def in_node(self, in_node, octaves=12):
        cs_path = "./gl/sumsinewave.glsl"
        self.cs = self.get_cs(cs_path)

        if "u_octaves" in self.cs:
            self.cs["u_octaves"].value = octaves

        self.out_height_2 = np.zeros((self.W, self.H, 4))
        self.out_height_2 = self.out_height_2.astype(np.float32)
        self.out_height_2 = self.gl.buffer(self.out_height_2.tobytes())

    @Base.out_node_wrapper
    def out_node(self):

        gx, gy = math.ceil(self.W / 32), math.ceil(self.H / 32)
        self.cs.run(gx, gy)

        return self.out_height_2.read()
