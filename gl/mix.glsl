
#version 440

#define LX 32
#define LY 32
#define PI 3.141592

#define OPERATION %CALC%

layout(local_size_x=LX, local_size_y=LY) in;
layout(binding=0) buffer out_buffer
{
    vec4 o_col[];
};

layout(binding=1) buffer a_buffer
{
    vec4 a_col[];
};

layout(binding=2) buffer b_buffer
{
    vec4 b_col[];
};

layout(binding=3) buffer c_buffer
{
    vec4 c_col[];
};

uniform int u_width;
uniform int u_height;


vec4 _mix(vec4 a, vec4 b, vec4 c)
{
    return mix(a, b, c);
}

vec4 _smoothstep(vec4 a, vec4 b, vec4 c)
{
    return smoothstep(a, b, c);
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
    o_col[i] = OPERATION(a_col[i], b_col[i], c_col[i]);
}
