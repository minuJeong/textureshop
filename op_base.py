
import os
from functools import wraps

import numpy as np
import moderngl as mg


class Base(object):

    GL = None

    def __init__(self, size=(512, 512), gl=None):
        super(Base, self).__init__()

        self.W, self.H = size[0], size[1]
        if gl:
            self.gl = gl
        else:
            if Base.GL:
                self.gl = Base.GL
            else:
                self.gl = mg.create_standalone_context()
                print("[Base] New GL context with id: {}".format(id(self.gl)))
        Base.GL = self.gl

    @staticmethod
    def in_node_wrapper(f):
        @wraps(f)
        def _(self, in_node, *args, **kargs):
            self.W, self.H = in_node.W, in_node.H
            f(self, in_node, *args, **kargs)
            return self
        return _

    @staticmethod
    def out_node_wrapper(f):
        @wraps(f)
        def _(self):
            data = f(self)
            data = np.frombuffer(data, dtype="f4")
            data = data.reshape((self.W, self.H, 4))
            return data
        return _

    def get_cs(self, cs_path, inject={}):
        context = None
        if not os.path.isabs(cs_path):
            dirpath = os.path.dirname(__file__)
            cs_path = "{}/{}".format(dirpath, cs_path)
        with open(cs_path, 'r') as fp:
            context = fp.read()

        for k, v in inject.items():
            context = context.replace(k, v)

        cs = self.gl.compute_shader(context)

        if "u_width" in cs:
            cs["u_width"].value = self.W
        if "u_height" in cs:
            cs["u_height"].value = self.H

        return cs

    def in_node(self, in_node: 'Base'):
        raise NotImplementedError("Do not use Base node directly")

    def out_node(self) -> 'Base':
        raise NotImplementedError("Do not use Base node directly")


class Init(Base):

    def __init__(self, size=(512, 512), gl=None):
        super(Init, self).__init__(size, gl)

    def get_gl(self):
        return self.gl

    def out_node(self):
        return np.zeros((self.W, self.H, 4))