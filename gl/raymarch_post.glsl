
#version 440

#define LX 32
#define LY 32
#define PI 3.141592

#define NEAR 0.5
#define FAR 100.0
#define SURFACE 0.0001

layout(local_size_x=LX, local_size_y=LY) in;
layout(binding=1) buffer in_basecolor
{
    vec4 i_col[];
};

layout(binding=2) buffer in_normal
{
    vec4 i_nrm[];
};

layout(binding=0) buffer out_color
{
    vec4 o_col[];
};

uniform int u_width;
uniform int u_height;

uniform vec3 u_lightpos;


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

    int i = int(xy.x + xy.y * wh.x);

    vec3 rgb = i_col[i].xyz;
    vec3 normal = i_nrm[i].xyz;

    vec3 L = normalize(-u_lightpos);
    float ndl = dot(normal, L);
    ndl = max(ndl, 0.0);

    o_col[i].xyz = rgb * ndl;
    o_col[i].w = 1.0;
}
