
import moderngl as mg


class Base(object):

    def __init__(self, gl):
        super(Base, self).__init__()
        self.gl = gl

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

    def __init__(self, gl=None):
        self.W, self.H = 512, 512

        if gl:
            self.gl = gl
        else:
            self.gl = mg.create_standalone_context()

    def get_gl(self):
        return self.gl
