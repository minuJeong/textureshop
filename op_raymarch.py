
import math

import numpy as np

from op_base import Base


class Raymarch(Base):
    """ raymarch node """

    class CameraInfo(object):
        u_campos = (0.0, 0.5, -5.0)
        u_camtarget = (0.0, 0.0, 0.0)

    @Base.in_node_wrapper
    def in_node(self, in_node, distance_field, lightinfo=None, caminfo=None, steps=64):
        cs_path = "./gl/raymarch.glsl"
        self.cs = self.get_cs(cs_path, {
            "%DIST_FIELD%": distance_field,
            "%NEAR%": "0.001",
            "%SURFACE%": "0.0001"
        })

        if "u_maxsteps" in self.cs:
            self.cs["u_maxsteps"].value = steps
        self.set_caminfo(caminfo)
        self.set_lightinfo(lightinfo)

        _buffer = np.zeros((self.W, self.H, 4))
        _buffer = _buffer.astype(np.float32)
        _buffer = _buffer.tobytes()

        self.depth = self.gl.buffer(_buffer)
        self.color = self.gl.buffer(_buffer)
        self.normal = self.gl.buffer(_buffer)
        self.shadow = self.gl.buffer(_buffer)

        self.g_buffer = DeferredLight.GBuffer()
        self.g_buffer.depth = self.depth
        self.g_buffer.color = self.color
        self.g_buffer.normal = self.normal
        self.g_buffer.shadow = self.shadow

    def set_caminfo(self, caminfo=None):
        if not caminfo:
            caminfo = Raymarch.CameraInfo()

        if "u_campos" in self.cs:
            self.cs["u_campos"].value = caminfo.u_campos

        if "u_camtarget" in self.cs:
            self.cs["u_camtarget"].value = caminfo.u_camtarget

    def set_lightinfo(self, lightinfo=None):
        if not lightinfo:
            lightinfo = DeferredLight.LightInfo()

        if "u_lightpos" in self.cs:
            self.cs["u_lightpos"].value = lightinfo.u_lightpos

    def out_node(self):
        self.depth.bind_to_storage_buffer(0)
        self.color.bind_to_storage_buffer(1)
        self.normal.bind_to_storage_buffer(2)
        self.shadow.bind_to_storage_buffer(3)

        gx, gy = math.ceil(self.W / 32), math.ceil(self.H / 32)
        self.cs.run(gx, gy)

        return self.g_buffer


class DeferredLight(Base):

    class LightInfo(object):
        u_lightpos = (2.0, 4.0, 3.0)
        u_shadow_intensity = 0.35

    class GBuffer(object):
        depth = None
        color = None
        normal = None
        shadow = None

    @Base.in_node_wrapper
    def in_node(self, in_node, g_buffer, lightinfo):
        cs_post_path = "./gl/raymarch_post.glsl"
        self.cs = self.get_cs(cs_post_path)

        _buffer = np.zeros((self.W, self.H, 4))
        _buffer = _buffer.astype(np.float32)
        _buffer = _buffer.tobytes()
        self.post_out = self.gl.buffer(_buffer)
        self.set_g_buffer(g_buffer)
        self.set_lightinfo(lightinfo)

    def set_lightinfo(self, lightinfo):
        if not lightinfo:
            lightinfo = Raymarch.LightInfo()
        if "u_lightpos" in self.cs:
            self.cs["u_lightpos"].value = lightinfo.u_lightpos
        if "u_shadow_intensity" in self.cs:
            self.cs["u_shadow_intensity"].value = lightinfo.u_shadow_intensity

    def set_g_buffer(self, g_buffer):
        _buffer = np.zeros((self.W, self.H, 4))
        _buffer = _buffer.astype(np.float32)
        _buffer = _buffer.tobytes()

        if not g_buffer:
            g_buffer = DeferredLight.GBuffer()
            g_buffer.depth = self.gl.buffer(_buffer)
            g_buffer.color = self.gl.buffer(_buffer)
            g_buffer.normal = self.gl.buffer(_buffer)
            g_buffer.shadow = self.gl.buffer(_buffer)

        self.in_depth = g_buffer.depth
        self.in_color = g_buffer.color
        self.in_normal = g_buffer.normal
        self.in_shadow = g_buffer.shadow

    @Base.out_node_wrapper
    def out_node(self):
        self.post_out.bind_to_storage_buffer(0)
        self.in_depth.bind_to_storage_buffer(1)
        self.in_color.bind_to_storage_buffer(2)
        self.in_normal.bind_to_storage_buffer(3)
        self.in_shadow.bind_to_storage_buffer(4)

        gx, gy = math.ceil(self.W / 32), math.ceil(self.H / 32)
        self.cs.run(gx, gy)

        return self.post_out.read()


from op_base import Init
from util import npappend


dff = """
float d = FAR;
float d1;
float d2;
vec3 c1;
vec3 c2;

// 2 boxes
{
    vec3 r1 = vec3(1.35);
    vec3 r2 = vec3(1.45);

    vec3 j1 = p - vec3(-2.0, 1.0, 0.0);
    j1 = rot_z(0.35) * j1;
    j1 = rot_y(0.35) * j1;

    vec3 j2 = p - vec3(+2.0, 1.0, 0.0);
    j2 = rot_z(-0.35) * j2;

    float b1 = box(j1, r1);
    float b2 = box(j2, r2);

    d1 = blend(b1, b2, 0.75);
    if (d1 < d)
    {
        c1 = vec3(0.2, 0.2, 1.0);
    }
    d = d1;
}

// 2 spheres
{
    float r1 = 2.25;
    float r2 = 2.25;
    float r3 = 2.25;

    vec3 j1 = p - vec3(-2.0, 0.0, +1.5);
    vec3 j2 = p - vec3(+2.0, 0.0, +1.5);
    vec3 j3 = p - vec3(+0.0, 0.0, -1.5);

    float s1 = sphere(j1, r1);
    float s2 = sphere(j2, r2);
    float s3 = sphere(j3, r3);

    d2 = min(s3, min(s1, s2));
    if (s1 < d || s2 < d || s3 < d)
    {
        c2 = vec3(1.0, 0.2, 0.2);
    }
    d = blend(d1, d2, 0.75);
}

d = blend(d1, d2, 0.85);
float cr = (d - d1) / (d2 - d1);
cr = clamp(cr, 0.0, 1.0);
if (w_need_color)
{
    w_color = mix(c1, c2, cr);
}
return d;

"""

init = Init((512, 512))
steps = 32

lightinfo = DeferredLight.LightInfo()
lightinfo.u_lightpos = (-2.0, 3.0, -5.0)
lightinfo.u_shadow_intensity = 0.25

camerainfo = Raymarch.CameraInfo()

import imageio as ii

raymarch_node = Raymarch().in_node(init, dff, lightinfo, camerainfo, steps)
light_node = DeferredLight().in_node(init, raymarch_node.out_node(), lightinfo)
output_writer = ii.get_writer("raymarched.mp4", fps=60)

import time
start_time = time.time()
for i in range(120):
    t = i * 0.052
    x = math.cos(t) * 7.0
    z = math.sin(t) * 7.0

    camerainfo.u_campos = (x, 5.0, z)
    camerainfo.u_camtarget = (0.0, 1.0, 0.0)
    raymarch_node.set_caminfo(camerainfo)

    # do raymarch
    g_buffer = raymarch_node.out_node()

    # do lighting
    light_node.set_g_buffer(g_buffer)
    output = light_node.out_node()

    # record
    npappend(output_writer, output)

output_writer.close()
