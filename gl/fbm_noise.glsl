
#version 440

#define LX 32
#define LY 32
#define PI 3.141592

layout(local_size_x=LX, local_size_y=LY) in;
layout(binding=0) buffer out_buffer
{
    vec4 o_col[];
};

uniform sampler2D u_noise_tex;

uniform int u_width;
uniform int u_height;
uniform int u_octaves;


float fbm(vec2 x, int octaves)
{
    float v = 0.0;
    float a = 0.5;
    vec2 shift = vec2(100);

    mat2 rot = mat2(cos(0.5), sin(0.5), -sin(0.5), cos(0.5));
    for (int i = 0; i < octaves; ++i)
    {
        v += a * texture(u_noise_tex, x).x;
        x = rot * x * 2.0 + shift;
        a *= 0.5;
    }
    return v;
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

    vec2 nuv = uv * 0.01;
    float n = fbm(nuv, u_octaves);

    n = clamp(n, 0.0, 1.0);

    vec4 rgba;
    rgba.xyz = vec3(n);
    rgba.w = 1.0;

    int i = int(xy.x + xy.y * wh.x);
    o_col[i] = rgba;
}
