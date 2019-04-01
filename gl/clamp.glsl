
#version 440

#define LX 32
#define LY 32
#define PI 3.141592

layout(local_size_x=LX, local_size_y=LY) in;
layout(binding=0) buffer out_buffer
{
    vec4 o_col[];
};

layout(binding=1) buffer in_buffer
{
    vec4 in_col[];
};

uniform int u_width;
uniform int u_height;

uniform float u_min_value;
uniform float u_max_value;

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
    o_col[i] = clamp(in_col[i], u_min_value, u_max_value);
}
