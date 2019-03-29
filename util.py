
from functools import wraps

import numpy as np
import imageio as ii


def to_glbytes(f, *args, **kargs):
    wraps(f)

    def _(*args, **kargs):
        data = f(*args, **kargs)
        return data.astype(np.float32).tobytes()
    return _


@to_glbytes
def cpu_noise(w, h):
    return np.random.uniform(0.0, 1.0, (w, h, 4))


def npwrite(path: str, data: np.ndarray):
    data = np.multiply(data, 255.0)
    data = data.astype(np.uint8)
    ii.imwrite(path, data)
