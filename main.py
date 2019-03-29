
import math
from functools import wraps

import numpy as np
import moderngl as mg
import imageio as ii


W, H = 800, 800


def to_glbytes(f, *args, **kargs):
    wraps(f)

    def _(*args, **kargs):
        data = f(*args, **kargs)
        return data.astype(np.float32).tobytes()
    return _


@to_glbytes
def noise(w, h):
    return np.random.uniform(0.0, 1.0, (w, h, 4))


def main():
    global W, H

    gl = mg.create_standalone_context()
    cs = gl.compute_shader(open("./gl/compute.glsl").read())

    if "u_width" in cs:
        cs["u_width"].value = W
    if "u_height" in cs:
        cs["u_height"].value = H

    u_noise_tex = gl.texture((W, H), 4, noise(W, H), dtype="f4")
    u_noise_tex.use(0)

    cs_out = gl.buffer(np.zeros((W, H, 4)).astype(np.float32).tobytes())
    cs_out.bind_to_storage_buffer(0)

    gx, gy = math.ceil(W / 32), math.ceil(H / 32)
    cs.run(gx, gy)

    data = cs_out.read()
    data = np.frombuffer(data, dtype="f4")
    data = data.reshape((W, H, 4))
    data = np.multiply(data, 255.0)
    data = data.astype(np.uint8)

    ii.imwrite("output.png", data)

    print(data[0, 0])


if __name__ == "__main__":
    main()
