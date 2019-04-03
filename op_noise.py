
import math

import moderngl as mg
import numpy as np

from op_base import Base
from util import cpu_noise


class CPURandom(Base):
    """ simple uniform random (using CPU) """

    def in_node(self, in_node, min_value=0.0, max_value=1.0):
        self.W, self.H = in_node.W, in_node.H
        self.min_value, self.max_value = min_value, max_value
        return self

    def out_node(self):
        noised = np.random.uniform(
            self.min_value, self.max_value, (self.W, self.H, 4))
        return noised.astype(np.float32)


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

    def in_node(self, in_node, noise_tex=None):
        self.W, self.H = in_node.W, in_node.H

        cs_path = "./gl/fbm_noise.glsl"
        self.cs = self.get_cs(cs_path)

        if noise_tex is not None:
            self.set_noisetex(noise_tex)

        if not hasattr(self, "u_noise_tex"):
            print("[FBM Noise] noise texture not coming in, generate cpu noise as base.")
            cpu_noise_data = cpu_noise(self.W, self.H)
            self.set_noisetex(cpu_noise_data, (self.W, self.H))

        return self

    def out_node(self):
        self.u_noise_tex.use(0)

        out_buffer = np.zeros((self.W, self.H, 4))
        out_buffer = out_buffer.astype(np.float32)

        self.cs_out = self.gl.buffer(out_buffer.tobytes())
        self.cs_out.bind_to_storage_buffer(0)

        gx, gy = math.ceil(self.W / 32), math.ceil(self.H / 32)
        self.cs.run(gx, gy)

        data = self.cs_out.read()
        data = np.frombuffer(data, dtype="f4")
        data = data.reshape((self.W, self.H, 4))
        return data
