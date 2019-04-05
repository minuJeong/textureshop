from functools import wraps

import imageio as ii
import numpy as np


def to_glbytes(f, *args, **kargs):
    wraps(f)

    def _(*args, **kargs):
        data = f(*args, **kargs)
        return data.astype(np.float32).tobytes()
    return _


@to_glbytes
def cpu_noise(w, h):
    return np.random.uniform(0.0, 1.0, (w, h, 4))


def _serialize_data_for_output(data):
    data = data[::-1]
    data = np.multiply(data, 255.0)
    data = data.astype(np.uint8)
    return data


def npwrite(path, data: np.ndarray):
    data = _serialize_data_for_output(data)
    ii.imwrite(path, data)


def npappend(writer, data: np.ndarray):
    data = _serialize_data_for_output(data)
    writer.append_data(data)


def _value_to_ndarray(value, w: int, h: int):
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
