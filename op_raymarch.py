
import math

import numpy as np

from op_base import Base


class Raymarch(Base):
    """ raymarch node """

    class LightInfo(object):
        u_lightpos = (2.0, 4.0, 3.0)

    @Base.in_node_wrapper
    def in_node(self, in_node, distance_field, light=None, steps=64):
        cs_path = "./gl/raymarch.glsl"
        self.cs_scene = self.get_cs(cs_path, {"%DIST_FIELD%": distance_field})

        if "u_maxsteps" in self.cs_scene:
            self.cs_scene["u_maxsteps"].value = steps

        if not light:
            light = Raymarch.LightInfo()
        self.light = light

        cs_post_path = "./gl/raymarch_post.glsl"
        self.cs_post = self.get_cs(cs_post_path)

    @Base.out_node_wrapper
    def out_node(self):
        self.cs_post["u_lightpos"].value = self.light.u_lightpos
        gx, gy = math.ceil(self.W / 32), math.ceil(self.H / 32)

        _buffer = np.zeros((self.W, self.H, 4))
        _buffer = _buffer.astype(np.float32)
        _buffer = _buffer.tobytes()

        scene_basecolor = self.gl.buffer(_buffer)
        scene_basecolor.bind_to_storage_buffer(0)
        scene_normal = self.gl.buffer(_buffer)
        scene_normal.bind_to_storage_buffer(1)

        self.cs_scene.run(gx, gy)

        post_out = self.gl.buffer(_buffer)
        post_in_basecolor = self.gl.buffer(scene_basecolor.read())
        post_in_normal = self.gl.buffer(scene_normal.read())

        post_out.bind_to_storage_buffer(0)
        post_in_basecolor.bind_to_storage_buffer(1)
        post_in_normal.bind_to_storage_buffer(2)

        self.cs_post.run(gx, gy)

        return post_out.read()


from op_base import Init
from util import npwrite

init = Init((512, 512))
dff = """
float r1 = 2.5;
float s1 = sphere(p, r1);
d = min(d, s1);
"""
steps = 64
light = Raymarch.LightInfo()
light.u_lightpos = (-2.0, 3.0, 5.0)
output = Raymarch().in_node(init, dff, light, steps).out_node()
npwrite("raymarched.png", output)
