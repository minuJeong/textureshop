
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
                print("new GL context with id: {}".format(id(self.gl)))
        Base.GL = self.gl

    def get_cs(self, cs_path):
        cs = self.gl.compute_shader(open(cs_path).read())

        if "u_width" in cs:
            cs["u_width"].value = self.W
        if "u_height" in cs:
            cs["u_height"].value = self.H

        return cs

    def in_node(self, in_node: 'Base'):
        raise NotImplemented

    def out_node(self) -> 'Base':
        raise NotImplemented


class Init(Base):

    def __init__(self, size=(512, 512), gl=None):
        super(Init, self).__init__(size, gl)

    def get_gl(self):
        return self.gl

    def out_node(self):
        return np.zeros((self.W, self.H, 4))
