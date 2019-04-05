
#version 440

#define LX 32
#define LY 32
#define PI 3.141592

#define NEAR 0.5
#define FAR 100.0
#define SURFACE 0.0001

layout(local_size_x=LX, local_size_y=LY) in;
layout(binding=1) buffer in_depth
{
    float i_dep[];
};

layout(binding=2) buffer in_color
{
    vec4 i_col[];
};

layout(binding=3) buffer in_normal
{
    vec4 i_nrm[];
};

layout(binding=3) buffer in_shadow
{
    vec4 i_shw[];
};

layout(binding=0) buffer out_color
{
    vec4 o_col[];
};

uniform int u_width;
uniform int u_height;

uniform vec3 u_campos = vec3(0.0, 0.5, -5.0);
uniform vec3 u_camtarget = vec3(0.0, 0.0, 0.0);
uniform vec3 u_lightpos;
uniform float u_shadow_intensity;


vec3 BXDF(float depth, vec3 color, vec3 normal, float shadow)
{
%BXDF%
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

    int i = int(xy.x + xy.y * wh.x);

    float depth = i_dep[i].x;
    vec3 color = i_col[i].xyz;
    vec3 normal = i_nrm[i].xyz;
    float shadow = i_shw[i].x;
    vec3 rgb = BXDF(depth, color, normal, shadow);

    o_col[i].xyz = rgb;
    o_col[i].w = 1.0;
}
