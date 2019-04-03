
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

uniform int u_width;
uniform int u_height;
uniform float u_clamp_min_value;
uniform float u_clamp_max_value;

vec4 _add(vec4 a, vec4 b)
{
    return a + b;
}

vec4 _mul(vec4 a, vec4 b)
{
    return a * b;
}

vec4 _div(vec4 a, vec4 b)
{
    float eps = 0.000001;
    vec4 base;
    base.x = abs(b.x) - eps < 0.0 ? eps : b.x;
    base.y = abs(b.x) - eps < 0.0 ? eps : b.y;
    base.z = abs(b.x) - eps < 0.0 ? eps : b.z;
    base.w = abs(b.x) - eps < 0.0 ? eps : b.w;
    return a / base;
}

vec4 _clamp(vec4 a, vec4 b)
{
    return clamp(a, u_clamp_min_value, u_clamp_max_value);
}

vec4 _oneminus(vec4 a, vec4 b)
{
    return 1.0 - a;
}

vec4 _sin(vec4 a, vec4 b)
{
    return sin(a);
}

vec4 _cos(vec4 a, vec4 b)
{
    return cos(a);
}

vec4 _tan(vec4 a, vec4 b)
{
    return tan(a);
}

vec4 _asin(vec4 a, vec4 b)
{
    return asin(a);
}

vec4 _acos(vec4 a, vec4 b)
{
    return acos(a);
}

vec4 _atan2(vec4 a, vec4 b)
{
    return atan(a, b);
}

vec4 _cosh(vec4 a, vec4 b)
{
    return cosh(a);
}

vec4 _sinh(vec4 a, vec4 b)
{
    return sinh(a);
}

vec4 _tanh(vec4 a, vec4 b)
{
    return tanh(a);
}

vec4 _pow(vec4 a, vec4 b)
{
    return pow(a, b);
}

vec4 _log(vec4 a, vec4 b)
{
    return log(a);
}

vec4 _log2(vec4 a, vec4 b)
{
    return log2(a);
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
    o_col[i] = OPERATION(a_col[i], b_col[i]);
}
