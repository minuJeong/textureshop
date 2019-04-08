
#version 440

#define LX 32
#define LY 32
#define PI 3.141592

#define NEAR %NEAR%
#define FAR 50.0
#define SURFACE %SURFACE%

layout(local_size_x=LX, local_size_y=LY) in;
layout(binding=0) buffer out_depth
{
    float o_dep[];
};

layout(binding=1) buffer out_basecolor
{
    vec4 o_col[];
};

layout(binding=2) buffer out_normal
{
    vec4 o_nrm[];
};

layout(binding=3) buffer out_shadow
{
    vec4 o_shw[];
};

uniform int u_width;
uniform int u_height;
uniform int u_maxsteps;
uniform float u_time;

uniform vec3 u_campos = vec3(0.0, 0.5, -5.0);
uniform vec3 u_camtarget = vec3(0.0, 0.0, 0.0);
uniform vec3 u_lightpos = vec3(1.0, 1.0, 0.0);

bool w_need_color = false;
vec3 w_color;


mat3 rot_x(float e)
{
    float ce = cos(e);
    float se = sin(e);
    return mat3(
        1.0, 0.0, 0.0,
        0.0, ce, -se,
        0.0, se, ce
    );
}

mat3 rot_y(float e)
{
    float ce = cos(e);
    float se = sin(e);
    return mat3(
        ce, 0.0, -se,
        0.0, 1.0, 0.0,
        se, 0.0, ce
    );
}

mat3 rot_z(float e)
{
    float ce = cos(e);
    float se = sin(e);
    return mat3(
        ce, -se, 0.0,
        se, ce, 0.0,
        0.0, 0.0, 1.0
    );
}

mat3 lookat(vec3 o, vec3 t, float roll)
{
    vec3 row_0 = vec3(sin(0.0), cos(0.0), 0.0);
    vec3 row_1 = normalize(t - o);
    vec3 row_2 = normalize(cross(row_1, row_0));
    vec3 row_3 = normalize(cross(row_2, row_1));
    return mat3(row_2, row_3, row_1);
}

float sphere(vec3 p, float r)
{
    return length(p) - r;
}

float box(vec3 p, vec3 b)
{
    vec3 d = abs(p) - b;
    float partial = length(max(d, 0.0));
    float full = min(max(d.x, max(d.y, d.z)), 0.0);
    return partial + full;
}

float blend(float a, float b, float k)
{
    float h = 0.5 + 0.5 * (a - b) / k;
    h = clamp(h, 0.0, 1.0);
    float t = mix(a, b, h);
    return t - k * h * (1.0 - h);
}

float world(vec3 p)
{
%DIST_FIELD%
}

float raymarch(vec3 o, vec3 r)
{
    float t = NEAR;
    float d;
    int i;
    for (i = u_maxsteps; i > 0; i--)
    {
        vec3 p = o + r * t;
        d = world(p);
        if (d < SURFACE)
        {
            return t;
        }
        t += d;
    }

    return t;
}

vec3 normal_at(vec3 p)
{
    vec2 e = vec2(0.001, 0.0);
    return normalize(vec3(
        world(p + e.xyy) - world(p - e.xyy),
        world(p + e.yxy) - world(p - e.yxy),
        world(p + e.yyx) - world(p - e.yyx)
    ));
}

float soft_shadow(vec3 o, vec3 r)
{
    float k = 8.4;

    float t = 0.02;
    float res = 1.0;
    float ph = 1e+8;

    for (int i = 0; i < u_maxsteps; i++)
    {
        float h = world(o + r * t);
        if (h < SURFACE)
        {
            return 0.0;
        }

        float y = h * h / (2.0 * ph);
        float d = sqrt(h * h - y * y);
        res = min(res, k * d / max(0.0, t - y));
        ph = h;
        t += h;
    }
    return res;
}

void main()
{
    vec2 wh = vec2(u_width, u_height);
    vec2 xy;
    vec2 uv;
    {
        xy.x = int(gl_LocalInvocationID.x + gl_WorkGroupID.x * LX);
        xy.y = int(gl_LocalInvocationID.y + gl_WorkGroupID.y * LY);
    
        xy = min(xy, wh);
        uv = xy / wh;
    }

    mat3 _look = lookat(u_campos, u_camtarget, 0.0);
    vec3 o = u_campos;
    vec3 r = normalize(vec3((uv - 0.5) * 2.0, 1.0));
    r = _look * r;

    w_need_color = true;
    float travel = raymarch(o, r);
    w_need_color = false;

    vec3 rgb = vec3(0.2, 0.2, 0.4);
    vec3 normal = vec3(0.5, 0.5, 1.0);
    float shadow = 0.0;
    if (travel < FAR)
    {
        vec3 P = o + r * travel;
        rgb.xyz = w_color;
        normal = normal_at(P);
        shadow = soft_shadow(P, -u_lightpos);
    }

    int i = int(xy.x + xy.y * wh.x);
    o_dep[i].x = travel;

    o_col[i].xyz = rgb;
    o_col[i].w = 1.0;

    o_nrm[i].xyz = normal.xyz;
    o_nrm[i].w = 1.0;

    o_shw[i].x = shadow;
}
