import math
import unittest

import imageio as ii

from ..op_base import Init
from ..op_raymarch import Raymarch, DeferredLight, LightInfo, CameraInfo
from ..util import npappend


class RaymarchTest(unittest.TestCase):

    def test_raymarch(self):
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

            s1 += dot(cos(p) * 0.05, vec3(0.5));
            s2 += dot(cos(p) * 0.05, vec3(0.5));
            s3 += dot(cos(p) * 0.05, vec3(0.5));

            d2 = min(s3, min(s1, s2));
            float sph_d = blend(d1, d2, 0.75);
            if (sph_d < d)
            {
                c2 = vec3(1.0, 0.2, 0.2);
            }
            d = sph_d;
        }

        d = blend(d1, d2, 0.65);
        d += dot(cos(p * 12.0), vec3(sin(u_time) * 0.5));

        float cr = (d - d1) / (d2 - d1);
        cr = clamp(cr, 0.0, 1.0);
        if (w_need_color)
        {
            w_color = mix(c1, c2, cr);
        }
        return d;

        """

        bxdf = """
        vec3 L = normalize(u_lightpos);
        float ndl = dot(normal, L);
        ndl = max(ndl, 0.0);

        vec3 V = normalize(u_campos);

        vec3 H = L + V;
        H = normalize(H);

        float shadow_value = clamp(shadow, 0.0, 1.0);
        float shadow_influence = mix(1.0, shadow, u_shadow_intensity);

        vec3 rgb = color * ndl * shadow_influence;
        return rgb;
        """

        init = Init((512, 512))
        steps = 32

        lightinfo = LightInfo()
        lightinfo.u_lightpos = (-2.0, 3.0, -5.0)
        lightinfo.u_shadow_intensity = 0.000001

        caminfo = CameraInfo()

        raymarch_node = Raymarch().in_node(init, dff, lightinfo, caminfo, steps)
        light_node = DeferredLight().in_node(init, bxdf, raymarch_node.out_node(), lightinfo, caminfo)
        output_writer = ii.get_writer("raymarched.mp4", fps=30)

        for i in range(120):
            t = i * 0.052
            x = math.cos(t) * 7.0
            z = math.sin(t) * 7.0

            caminfo.u_campos = (x, 5.0, z)
            caminfo.u_camtarget = (0.0, 1.0, 0.0)
            raymarch_node.set_caminfo(caminfo)

            # do raymarch
            g_buffer = raymarch_node.out_node()

            # do lighting
            light_node.set_g_buffer(g_buffer)
            output = light_node.out_node()

            # record
            npappend(output_writer, output)

        output_writer.close()
