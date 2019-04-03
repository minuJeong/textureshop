
#version 440

#define LX 32
#define LY 32
#define PI 3.141592

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

uniform int u_width;
uniform int u_height;

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

    float eps = 0.000001;
    vec4 base;
    base.x = abs(b_col[i].x) - eps < 0.0 ? eps : b_col[i].x;
    base.y = abs(b_col[i].x) - eps < 0.0 ? eps : b_col[i].y;
    base.z = abs(b_col[i].x) - eps < 0.0 ? eps : b_col[i].z;
    base.w = abs(b_col[i].x) - eps < 0.0 ? eps : b_col[i].w;

    o_col[i] = a_col[i] * (1.0 / base);
}
