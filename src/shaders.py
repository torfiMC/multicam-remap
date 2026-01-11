from OpenGL import GL

VERT_SRC = r"""
#version 120
attribute vec3 a_pos;
attribute vec2 a_uv;

uniform mat4 u_mvp;

varying vec2 v_uv;

void main() {
    v_uv = a_uv;
    gl_Position = u_mvp * vec4(a_pos, 1.0);
}
"""

FRAG_SRC = r"""
#version 120
uniform sampler2D u_src;     // combined source frame (left|right)
uniform sampler2D u_lookup;  // RGBA8 packed UV lookup (u16 in RG, v16 in BA)
uniform float u_uv_offset_x; // U offset 
uniform float u_uv_scale_x;  // U scale

varying vec2 v_uv;

vec2 unpack_uv_rgba8(vec4 t) {
    // t is 0..1; reconstruct u16/v16 from RG/BA bytes
    float r = floor(t.r * 255.0 + 0.5);
    float g = floor(t.g * 255.0 + 0.5);
    float b = floor(t.b * 255.0 + 0.5);
    float a = floor(t.a * 255.0 + 0.5);

    float u16 = r * 256.0 + g;
    float v16 = b * 256.0 + a;

    return vec2(u16 / 65535.0, v16 / 65535.0);
}

void main() {
    vec4 lu = texture2D(u_lookup, v_uv);

    // Invalid pixels: check if the lookup value is black (0,0,0,0)
    // We cannot use alpha alone because alpha contains part of the V coordinate.
    // 1/255 ~= 0.0039. (1/255)^2 ~= 0.000015.
    // Previous threshold 0.0001 (10e-5) was too high, discarding valid low values (1/255, 2/255).
    // Use a threshold smaller than (1/255)^2.
    if (dot(lu, lu) < 0.000001) {
        discard; // Discard invalid pixels to allow multiple spheres to overlap/composite
    }

    vec2 src_uv = unpack_uv_rgba8(lu);
    
    // Scale and offset for side-by-side layout
    src_uv.x = src_uv.x * u_uv_scale_x + u_uv_offset_x;
    
    gl_FragColor = texture2D(u_src, src_uv);
}
"""

FRAG_SRC_FLOAT = r"""
#version 120
uniform sampler2D u_src;     // combined source frame (left|right)
uniform sampler2D u_lookup;  // RG16F float lookup (u in R, v in G)
uniform float u_uv_offset_x; // U offset in source texture (e.g. 0.0 or 0.5)
uniform float u_uv_scale_x;  // U scale in source texture (e.g. 0.5 or 1.0)

varying vec2 v_uv;

void main() {
    // Sample high-precision float texture directly
    vec2 lu = texture2D(u_lookup, v_uv).rg;

    // Check for sentinel value (e.g. negative) to indicate invalid mapping
    if (lu.x < -0.0001) {
        discard; 
    }

    vec2 src_uv = lu;
    
    // Scale and offset for extraction from atlas/side-by-side
    src_uv.x = src_uv.x * u_uv_scale_x + u_uv_offset_x;
    
    gl_FragColor = texture2D(u_src, src_uv);
}
"""

FRAG_SRC_FLOAT_SOFTBORDER = r"""
#version 120
uniform sampler2D u_src;     // combined source frame (left|right)
uniform sampler2D u_lookup;  // RG16F float lookup (u in R, v in G)
uniform sampler2D u_mask;    // L8/LUMINANCE mask, sampled in equirect space
uniform float u_uv_offset_x; // U offset in source texture (e.g. 0.0 or 0.5)
uniform float u_uv_scale_x;  // U scale in source texture (e.g. 0.5 or 1.0)

varying vec2 v_uv;

void main() {
    vec2 lu = texture2D(u_lookup, v_uv).rg;

    // Preserve the existing low-end validity check (sentinel) to avoid sampling
    // bogus UVs outside the usable area.
    if (lu.x < -0.0001) {
        discard;
    }

    vec2 src_uv = lu;
    src_uv.x = src_uv.x * u_uv_scale_x + u_uv_offset_x;

    vec3 rgb = texture2D(u_src, src_uv).rgb;
    float a = texture2D(u_mask, v_uv).r;

    gl_FragColor = vec4(rgb, a);
}
"""

def compile_shader(src: str, shader_type):
    sh = GL.glCreateShader(shader_type)
    GL.glShaderSource(sh, src)
    GL.glCompileShader(sh)
    ok = GL.glGetShaderiv(sh, GL.GL_COMPILE_STATUS)
    if not ok:
        log = GL.glGetShaderInfoLog(sh).decode("utf-8", "replace")
        raise RuntimeError(f"Shader compile failed:\n{log}")
    return sh


def link_program(vs, fs):
    prog = GL.glCreateProgram()
    GL.glAttachShader(prog, vs)
    GL.glAttachShader(prog, fs)
    GL.glLinkProgram(prog)
    ok = GL.glGetProgramiv(prog, GL.GL_LINK_STATUS)
    if not ok:
        log = GL.glGetProgramInfoLog(prog).decode("utf-8", "replace")
        raise RuntimeError(f"Program link failed:\n{log}")
    return prog
