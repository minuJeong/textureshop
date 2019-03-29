
import math

import moderngl as mg
import numpy as np

from base import Base
from util import cpu_noise
from util import to_glbytes


class FBMNoise(Base):
    """Fractional brownian motion noise"""

    def set_noisetex(self, noise_tex, bytes_size=None):
        if isinstance(noise_tex, type(mg.texture)):
            self.u_noise_tex = noise_tex
            return

        if isinstance(noise_tex, type(np.ndarray)):
            s = noise_tex.shape
            size = (s[0], s[1])
            channels = s[2]
            data = to_glbytes(noise_tex)
            self.u_noise_tex = self.gl.texture(size, channels, data, dtype="f4")
            return

        if isinstance(noise_tex, (bytes, bytearray)):
            if not bytes_size:
                raise Exception("noise_tex coming in with bytes, but size not specified")
                return
            self.u_noise_tex = self.gl.texture(bytes_size, 4, noise_tex, dtype="f4")
            return

        raise NotImplementedError(
            "setting noise from {} is not implemented".format(type(noise_tex)))

    def in_node(self, in_node, noise_tex=None):
        self.W, self.H = in_node.W, in_node.H

        cs_path = "./gl/fbm_noise.glsl"
        self.cs = self.get_cs(cs_path)

        if noise_tex:
            self.set_noisetex(noise_tex)

        if not hasattr(self, "u_noise_tex"):
            print("noise texture not coming in, using cpu noise.")
            cpu_noise_data = cpu_noise(self.W, self.H)
            self.set_noisetex(cpu_noise_data, (self.W, self.H))
        self.u_noise_tex.use(0)

        out_buffer = np.zeros((self.W, self.H, 4))
        out_buffer = out_buffer.astype(np.float32)

        self.cs_out = self.gl.buffer(out_buffer.tobytes())
        self.cs_out.bind_to_storage_buffer(0)

    def out_node(self):
        gx, gy = math.ceil(self.W / 32), math.ceil(self.H / 32)
        self.cs.run(gx, gy)

        data = self.cs_out.read()
        data = np.frombuffer(data, dtype="f4")
        data = data.reshape((self.W, self.H, 4))
        return data
