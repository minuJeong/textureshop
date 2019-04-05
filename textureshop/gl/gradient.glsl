
#version 440

#define LX 32
#define LY 32
#define PI 3.141592
#define ROOT2 1.4142135623731

#define GRAD_TYPE %TYPE%

layout(local_size_x=LX, local_size_y=LY) in;
layout(binding=0) buffer out_buffer
{
    vec4 o_col[];
};

uniform int u_width;
uniform int u_height;

vec4 _horizontal_left_grid(vec2 uv)
{
    float r = uv.x;
    return vec4(r, r, r, 1.0);
}

vec4 _horizontal_right_grid(vec2 uv)
{
    float r = 1.0 - uv.x;
    return vec4(r, r, r, 1.0);
}

vec4 _vertical_up_grid(vec2 uv)
{
    float r = uv.y;
    return vec4(r, r, r, 1.0);
}

vec4 _vertical_down_grid(vec2 uv)
{
    float r = 1.0 - uv.y;
    return vec4(r, r, r, 1.0);
}

vec4 _radial_in_grid(vec2 uv)
{
    uv = uv * 2.0 - 1.0;
    float rr = uv.x * uv.x + uv.y * uv.y;
    float r = sqrt(rr) / ROOT2;
    return vec4(r, r, r, 1.0);
}

vec4 _radial_out_grid(vec2 uv)
{
    uv = uv * 2.0 - 1.0;
    float rr = uv.x * uv.x + uv.y * uv.y;
    float r = 1.0  - sqrt(rr) / ROOT2;
    return vec4(r, r, r, 1.0);
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
    o_col[i] = GRAD_TYPE(uv);
}
