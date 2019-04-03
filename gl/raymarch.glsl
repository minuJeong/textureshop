
#version 440

#define LX 32
#define LY 32
#define PI 3.141592

#define NEAR 0.5
#define FAR 100.0
#define SURFACE 0.0001

layout(local_size_x=LX, local_size_y=LY) in;
layout(binding=0) buffer out_basecolor
{
    vec4 o_col[];
};

layout(binding=1) buffer out_normal
{
    vec4 o_nrm[];
};

uniform int u_width;
uniform int u_height;
uniform int u_maxsteps;

uniform vec3 u_lightpos = vec3(2.0, 4.0, 3.0);


float sphere(vec3 p, float r)
{
    return length(p) - r;
}

float world(vec3 p)
{
float d = FAR;
{
%DIST_FIELD%
}
return d;
}

float raymarch(vec3 o, vec3 r)
{
    float t = NEAR;
    float d;
    for (int i = u_maxsteps; i > 0; i--)
    {
        vec3 p = o + r * t;
        d = world(p);
        if (d < SURFACE)
        {
            return t;
        }
        t += d;
    }
    return FAR;
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

    vec3 o = vec3(0.0, 0.0, -10.0);
    vec3 r = normalize(vec3((uv - 0.5) * 2.0, 1.0));
    vec3 rgb;
    vec3 normal;

    rgb.xyz = vec3(0.0, 0.0, 0.4);
    normal.xyz = vec3(0.5, 0.5, 1.0);

    float t = raymarch(o, r);
    if (t < FAR)
    {
        vec3 P = o + r * t;
        normal = normal_at(P);

        rgb.xyz = vec3(0.8, 0.23, 0.5);
    }

    int i = int(xy.x + xy.y * wh.x);
    o_col[i].xyz = rgb;
    o_col[i].w = 1.0;

    o_nrm[i].xyz = normal.xyz;
    o_nrm[i].w = 1.0;
}
