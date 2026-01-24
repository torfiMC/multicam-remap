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
uniform float u_view_scale;  // >1 = zoom in (sample smaller portion)
uniform vec2 u_view_offset;  // delta from center (0.5,0.5) in UV space

varying vec2 v_uv;

vec2 apply_view(vec2 uv) {
    vec2 centered = (uv - vec2(0.5)) / max(u_view_scale, 1e-6) + vec2(0.5) + u_view_offset;
    centered.x = centered.x - floor(centered.x); // wrap horizontally
    centered.y = clamp(centered.y, 0.0, 1.0);
    return centered;
}

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
    vec2 lookup_uv = apply_view(v_uv);
    vec4 lu = texture2D(u_lookup, lookup_uv);

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

FRAG_SRC_RAW = r"""
#version 120
uniform sampler2D u_src;
uniform float u_uv_offset_x;
uniform float u_uv_scale_x;
uniform float u_orientation_rad;

varying vec2 v_uv;

void main() {
    vec2 src_uv = vec2(v_uv.x, 1.0 - v_uv.y);
    vec2 centered = src_uv - vec2(0.5, 0.5);
    float s = sin(u_orientation_rad);
    float c = cos(u_orientation_rad);
    vec2 rotated = vec2(centered.x * c - centered.y * s, centered.x * s + centered.y * c);
    src_uv = rotated + vec2(0.5, 0.5);
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
uniform float u_view_scale;  // >1 = zoom in (sample smaller portion)
uniform vec2 u_view_offset;  // delta from center (0.5,0.5) in UV space

varying vec2 v_uv;

vec2 apply_view(vec2 uv) {
    vec2 centered = (uv - vec2(0.5)) / max(u_view_scale, 1e-6) + vec2(0.5) + u_view_offset;
    centered.x = centered.x - floor(centered.x); // wrap horizontally for 360 pano
    centered.y = clamp(centered.y, 0.0, 1.0);
    return centered;
}

void main() {
    vec2 lookup_uv = apply_view(v_uv);
    // Sample high-precision float texture directly
    vec2 lu = texture2D(u_lookup, lookup_uv).rg;

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
uniform float u_view_scale;  // >1 = zoom in (sample smaller portion)
uniform vec2 u_view_offset;  // delta from center (0.5,0.5) in UV space

varying vec2 v_uv;

vec2 apply_view(vec2 uv) {
    vec2 centered = (uv - vec2(0.5)) / max(u_view_scale, 1e-6) + vec2(0.5) + u_view_offset;
    centered.x = centered.x - floor(centered.x);
    centered.y = clamp(centered.y, 0.0, 1.0);
    return centered;
}

void main() {
    vec2 lookup_uv = apply_view(v_uv);
    vec2 lu = texture2D(u_lookup, lookup_uv).rg;

    // Preserve the existing low-end validity check (sentinel) to avoid sampling
    // bogus UVs outside the usable area.
    if (lu.x < -0.0001) {
        discard;
    }

    vec2 src_uv = lu;
    src_uv.x = src_uv.x * u_uv_scale_x + u_uv_offset_x;

    vec3 rgb = texture2D(u_src, src_uv).rgb;
    float a = texture2D(u_mask, lookup_uv).r;

    gl_FragColor = vec4(rgb, a);
}
"""

FRAG_SRC_PANO = r"""
#version 120
uniform sampler2D u_src;          // camera frame
uniform mat3 u_cam_rot;           // world->camera rotation
uniform float u_focal_x;          // f_pix / width
uniform float u_focal_y;          // f_pix / height
uniform float u_uv_offset_x;      // slice offset for dual feeds
uniform float u_uv_scale_x;       // slice scale for dual feeds
uniform float u_view_scale;       // pano zoom
uniform vec2 u_view_offset;       // pano pan offset
uniform int u_model;              // 0=fisheye equidistant, 1=rectilinear

varying vec2 v_uv;

vec2 apply_view(vec2 uv) {
    vec2 centered = (uv - vec2(0.5)) / max(u_view_scale, 1e-6) + vec2(0.5) + u_view_offset;
    centered.x = centered.x - floor(centered.x); // wrap horizontally for 360
    centered.y = clamp(centered.y, 0.0, 1.0);
    return centered;
}

void main() {
    vec2 uv = apply_view(v_uv);
    float lon = (0.5 - uv.x) * 6.28318530718; // flip X so yaw aligns with sphere
    float lat = (0.5 - uv.y) * 3.14159265359; // pi

    float cl = cos(lat);
    float sl = sin(lat);
    float sx = sin(lon);
    float cx = cos(lon);

    vec3 d_world = vec3(cl * sx, sl, cl * cx);
    vec3 d_cam = u_cam_rot * d_world;

    // Mirror both X and Y in camera space (pano/equirect only).
    d_cam.x = d_cam.x;
    d_cam.y = -d_cam.y;

    if (d_cam.z <= 0.0) {
        discard;
    }

    float u_tex;
    float v_tex;
    if (u_model == 0) {
        float theta = acos(clamp(d_cam.z, -1.0, 1.0));
        float r_xy = length(d_cam.xy);
        if (r_xy < 1e-8) {
            u_tex = 0.5;
            v_tex = 0.5;
        } else {
            float r = theta;
            u_tex = 0.5 + u_focal_x * (r * d_cam.x / r_xy);
            v_tex = 0.5 + u_focal_y * (r * d_cam.y / r_xy);
        }
    } else {
        u_tex = 0.5 + u_focal_x * (d_cam.x / d_cam.z);
        v_tex = 0.5 + u_focal_y * (d_cam.y / d_cam.z);
    }

    u_tex = u_tex * u_uv_scale_x + u_uv_offset_x;

    if (u_tex < 0.0 || u_tex > 1.0 || v_tex < 0.0 || v_tex > 1.0) {
        discard;
    }

    vec3 rgb = texture2D(u_src, vec2(u_tex, v_tex)).rgb;
    gl_FragColor = vec4(rgb, 1.0);
}
"""

GRID_VERT_SRC = r"""
#version 120
attribute vec3 a_pos;
uniform mat4 u_mvp;

void main() {
    gl_Position = u_mvp * vec4(a_pos, 1.0);
}
"""

GRID_FRAG_SRC = r"""
#version 120
uniform vec4 u_color;

void main() {
    gl_FragColor = u_color;
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
