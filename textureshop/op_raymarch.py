import math

import numpy as np

from .op_base import Base


class LightInfo(object):
    u_lightpos = (2.0, 4.0, 3.0)
    u_shadow_intensity = 0.35


class GBuffer(object):
    depth = None
    color = None
    normal = None
    shadow = None


class CameraInfo(object):
    u_campos = (0.0, 0.5, -5.0)
    u_camtarget = (0.0, 0.0, 0.0)


class Raymarch(Base):
    """ raymarch node """

    @Base.in_node_wrapper
    def in_node(self, in_node, distance_field, lightinfo=None, caminfo=None, steps=32):
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

        self.g_buffer = GBuffer()
        self.g_buffer.depth = self.depth
        self.g_buffer.color = self.color
        self.g_buffer.normal = self.normal
        self.g_buffer.shadow = self.shadow

        self.time = 0

    def set_caminfo(self, caminfo=None):
        if not caminfo:
            caminfo = CameraInfo()

        if "u_campos" in self.cs:
            self.cs["u_campos"].value = caminfo.u_campos

        if "u_camtarget" in self.cs:
            self.cs["u_camtarget"].value = caminfo.u_camtarget

    def set_lightinfo(self, lightinfo=None):
        if not lightinfo:
            lightinfo = LightInfo()

        if "u_lightpos" in self.cs:
            self.cs["u_lightpos"].value = lightinfo.u_lightpos

    def out_node(self):
        self.time += 0.1
        if "u_time" in self.cs:
            self.cs["u_time"].value = self.time

        self.depth.bind_to_storage_buffer(0)
        self.color.bind_to_storage_buffer(1)
        self.normal.bind_to_storage_buffer(2)
        self.shadow.bind_to_storage_buffer(3)

        gx, gy = math.ceil(self.W / 32), math.ceil(self.H / 32)
        self.cs.run(gx, gy)

        return self.g_buffer


class DeferredLight(Base):

    @Base.in_node_wrapper
    def in_node(self, in_node, bxdf, g_buffer, lightinfo=None, caminfo=None):
        cs_post_path = "./gl/raymarch_post.glsl"
        self.cs = self.get_cs(cs_post_path, {
            "%BXDF%": bxdf
        })

        _buffer = np.zeros((self.W, self.H, 4))
        _buffer = _buffer.astype(np.float32)
        _buffer = _buffer.tobytes()
        self.post_out = self.gl.buffer(_buffer)
        self.set_g_buffer(g_buffer)
        self.set_lightinfo(lightinfo)
        self.set_caminfo(caminfo)

    def set_lightinfo(self, lightinfo):
        if not lightinfo:
            lightinfo = LightInfo()
        if "u_lightpos" in self.cs:
            self.cs["u_lightpos"].value = lightinfo.u_lightpos
        if "u_shadow_intensity" in self.cs:
            self.cs["u_shadow_intensity"].value = lightinfo.u_shadow_intensity

    def set_caminfo(self, caminfo=None):
        if not caminfo:
            caminfo = CameraInfo()

        if "u_campos" in self.cs:
            self.cs["u_campos"].value = caminfo.u_campos

        if "u_camtarget" in self.cs:
            self.cs["u_camtarget"].value = caminfo.u_camtarget

    def set_g_buffer(self, g_buffer):
        _buffer = np.zeros((self.W, self.H, 4))
        _buffer = _buffer.astype(np.float32)
        _buffer = _buffer.tobytes()

        if not g_buffer:
            g_buffer = GBuffer()
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
