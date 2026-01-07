#!/usr/bin/env python3
import sys
import os
import time
import math
import ctypes
import argparse
import yaml
import numpy as np
import glfw
from OpenGL import GL

# Local imports
from src.capture import CameraDevice
from src.lens import LensView
from src.geometry import make_inside_sphere, make_quad
from src.shaders import compile_shader, link_program, VERT_SRC, FRAG_SRC_FLOAT
from src.math_utils import mat4_perspective, mat4_from_yaw_pitch_roll

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cameras.yaml", help="Path to camera config file")
    args = ap.parse_args()

    # Load Config
    if not os.path.exists(args.config):
        print("Config file not found.")
        # Create default config if needed logic here... or just fail
        raise RuntimeError(f"Config file {args.config} not found.")

    with open(args.config, 'r') as f:
        config_data = yaml.safe_load(f)
    
    cam_configs = config_data.get('cameras', [])
    if not cam_configs:
         raise RuntimeError("No cameras defined in config file.")

    # Initialize GLFW
    if not glfw.init():
        raise RuntimeError("glfw.init() failed")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)

    win_w, win_h = 1280, 720
    window = glfw.create_window(win_w, win_h, "Multi-Camera Sphere Viewer", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create window")
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    # Initialize Devices and Lenses
    device_registry = {} # id_str -> CameraDevice
    devices = [] # Unique list of devices to update
    lenses = []
    
    try:
        for cc in cam_configs:
            dev_id = str(cc.get("id", "0"))
            
            # Retrieve or Create Device
            if dev_id in device_registry:
                dev = device_registry[dev_id]
                print(f"Reusing existing device {dev_id} for '{cc.get('name')}'")
            else:
                print(f"Initializing new device {dev_id} for '{cc.get('name')}'")
                dev = CameraDevice(cc)
                device_registry[dev_id] = dev
                devices.append(dev)
            
            # Create Lens
            # Lenses are 1:1 with config entries now
            lenses.append(LensView(dev, cc))

    except Exception as e:
        glfw.terminate()
        raise e

    # Geometry
    PROJ_FOV = 180.0
    verts, uvs, indices = make_inside_sphere(
        lat_steps=64, lon_steps=64, radius=10.0, fov_deg=PROJ_FOV
    )
    q_verts, q_uvs, q_indices = make_quad()

    # Generic VBOs
    vbo_pos = GL.glGenBuffers(1)
    vbo_uv = GL.glGenBuffers(1)
    ebo = GL.glGenBuffers(1)

    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo_pos)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, verts.nbytes, verts, GL.GL_STATIC_DRAW)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo_uv)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, uvs.nbytes, uvs, GL.GL_STATIC_DRAW)
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, ebo)
    GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL.GL_STATIC_DRAW)

    q_vbo_pos = GL.glGenBuffers(1)
    q_vbo_uv = GL.glGenBuffers(1)
    q_ebo = GL.glGenBuffers(1)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, q_vbo_pos)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, q_verts.nbytes, q_verts, GL.GL_STATIC_DRAW)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, q_vbo_uv)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, q_uvs.nbytes, q_uvs, GL.GL_STATIC_DRAW)
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, q_ebo)
    GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, q_indices.nbytes, q_indices, GL.GL_STATIC_DRAW)

    # Shader (Using Float variant primarily)
    vs = compile_shader(VERT_SRC, GL.GL_VERTEX_SHADER)
    fs = compile_shader(FRAG_SRC_FLOAT, GL.GL_FRAGMENT_SHADER)
    prog = link_program(vs, fs)

    # Uniform locs
    a_pos = GL.glGetAttribLocation(prog, "a_pos")
    a_uv = GL.glGetAttribLocation(prog, "a_uv")
    u_mvp = GL.glGetUniformLocation(prog, "u_mvp")
    u_src = GL.glGetUniformLocation(prog, "u_src")
    u_lookup = GL.glGetUniformLocation(prog, "u_lookup")
    u_uv_offset_x = GL.glGetUniformLocation(prog, "u_uv_offset_x")
    u_uv_scale_x  = GL.glGetUniformLocation(prog, "u_uv_scale_x")

    # Render Loop State
    yaw = 0.0
    pitch = 0.0
    roll = 0.0
    fov = 70.0
    view_sphere = True
    
    # Edit Mode State
    edit_mode = False
    sel_lens_idx = 0
    sel_attr_idx = 0
    ATTR_NAMES = ["Yaw", "Pitch", "Roll", "Orientation"]
    
    dragging = False
    last_x, last_y = 0.0, 0.0
    mouse_sens = 0.12

    def clamp_pitch(p): return max(-89.0, min(89.0, p))

    def save_config_file():
        print(f"[edit] Saving configuration to {args.config}...")
        try:
            # Sync lenses back to configs
            for i, lens in enumerate(lenses):
                cfg = cam_configs[i]
                cfg['yaw'] = float(lens.world_yaw)
                cfg['pitch'] = float(lens.world_pitch)
                cfg['roll'] = float(lens.world_roll)
                cfg['orientation'] = float(lens.orientation)
            
            with open(args.config, 'w') as f:
                yaml.dump({'cameras': cam_configs}, f, sort_keys=False)
            print("[edit] Save complete.")
        except Exception as e:
            print(f"[edit] Error saving config: {e}")

    def on_key_wrapper(win, key, scancode, action, mods):
        nonlocal yaw, pitch, roll, fov, view_sphere
        nonlocal edit_mode, sel_lens_idx, sel_attr_idx

        if action not in (glfw.PRESS, glfw.REPEAT): return
        
        if key == glfw.KEY_ESCAPE: 
            glfw.set_window_should_close(win, True)
        
        # --- Edit Mode Handling ---
        if key == glfw.KEY_E:
            edit_mode = not edit_mode
            if edit_mode:
                print("-" * 40)
                print("EDIT MODE ENABLED")
                print("Controls: [E] Exit/Save, [C] Cycle Cam, [A] Cycle Attr, [+ / -] Adjust")
                cam_name = cam_configs[sel_lens_idx].get('name', f'Cam {sel_lens_idx}')
                print(f"Current Selection: {cam_name} | Attribute: {ATTR_NAMES[sel_attr_idx]}")
                print("-" * 40)
            else:
                save_config_file()
                print("EDIT MODE DISABLED")

        if edit_mode:
            # Edit Controls
            lens = lenses[sel_lens_idx]
            cam_name = cam_configs[sel_lens_idx].get('name', f'Cam {sel_lens_idx}')
            val_changed = False

            if key == glfw.KEY_C:
                sel_lens_idx = (sel_lens_idx + 1) % len(lenses)
                lens = lenses[sel_lens_idx]
                cam_name = cam_configs[sel_lens_idx].get('name', f'Cam {sel_lens_idx}')
                print(f"[edit] Selected Camera: {cam_name}")
                
            elif key == glfw.KEY_A:
                sel_attr_idx = (sel_attr_idx + 1) % 4
                print(f"[edit] Selected Attribute: {ATTR_NAMES[sel_attr_idx]}")
                
            elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD, glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
                # Adjust value
                delta = 10.0 if key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD) else -10.0
                
                if sel_attr_idx == 0: # Yaw
                    lens.world_yaw += delta
                    print(f"[edit] {cam_name} Yaw -> {lens.world_yaw}")
                elif sel_attr_idx == 1: # Pitch
                    lens.world_pitch += delta
                    print(f"[edit] {cam_name} Pitch -> {lens.world_pitch}")
                elif sel_attr_idx == 2: # Roll
                    lens.world_roll += delta
                    print(f"[edit] {cam_name} Roll -> {lens.world_roll}")
                elif sel_attr_idx == 3: # Orientation
                    lens.orientation += delta
                    print(f"[edit] {cam_name} Orientation -> {lens.orientation}")

        else:
            # Standard View Controls
            if key == glfw.KEY_R: 
                yaw = pitch = roll = 0.0; fov = 70.0
            elif key == glfw.KEY_Q: 
                roll -= 2.0
            # elif key == glfw.KEY_E: # Moved to Edit Mode Toggle
            #     pass 
            elif key == glfw.KEY_V:
                view_sphere = not view_sphere
                print(f"[view] Mode: {'Sphere' if view_sphere else 'Equirect'}")

    def on_mouse_wrapper(win, button, action, mods):
        nonlocal dragging, last_x, last_y
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                dragging = True
                last_x, last_y = glfw.get_cursor_pos(win)
            elif action == glfw.RELEASE:
                dragging = False

    def on_cursor_wrapper(win, x, y):
        nonlocal yaw, pitch, last_x, last_y
        if not dragging: return
        dx = x - last_x; dy = y - last_y
        last_x, last_y = x, y
        yaw += dx * mouse_sens
        pitch += dy * mouse_sens
        pitch = clamp_pitch(pitch)

    def on_scroll_wrapper(win, xoff, yoff):
        nonlocal fov
        fov -= yoff * 2.0
        fov = max(20.0, min(180.0, fov))

    glfw.set_key_callback(window, on_key_wrapper)
    glfw.set_mouse_button_callback(window, on_mouse_wrapper)
    glfw.set_cursor_pos_callback(window, on_cursor_wrapper)
    glfw.set_scroll_callback(window, on_scroll_wrapper)

    GL.glDisable(GL.GL_CULL_FACE)
    GL.glDisable(GL.GL_DEPTH_TEST)

    last_t = time.time()
    frames = 0

    while not glfw.window_should_close(window):
        glfw.poll_events()

        # Update all cameras
        for dev in devices:
            dev.update()
            dev.upload_texture()
        
        # Clear
        fb_w, fb_h = glfw.get_framebuffer_size(window)
        GL.glViewport(0, 0, fb_w, fb_h)
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        aspect = fb_w / float(fb_h if fb_h else 1)

        if view_sphere:
            proj = mat4_perspective(fov, aspect, 0.1, 100.0)
            view = mat4_from_yaw_pitch_roll(yaw, pitch, roll)

            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo_pos)
            GL.glEnableVertexAttribArray(a_pos)
            GL.glVertexAttribPointer(a_pos, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, ctypes.c_void_p(0))
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo_uv)
            GL.glEnableVertexAttribArray(a_uv)
            GL.glVertexAttribPointer(a_uv, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, ctypes.c_void_p(0))
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, ebo)

            GL.glUseProgram(prog)
            GL.glUniform1i(u_src, 0)
            GL.glUniform1i(u_lookup, 1)

            # Render in reverse order (Painter's Algorithm) so first camera in list is drawn "on top"
            for lens in reversed(lenses): 
                 m_world = mat4_from_yaw_pitch_roll(lens.world_yaw, lens.world_pitch, lens.world_roll)
                 m_orient = mat4_from_yaw_pitch_roll(0.0, 0.0, lens.orientation)
                 # Apply orientation first (local), then world placement
                 model = m_world @ m_orient
                 
                 mvp = proj @ view @ model
                 
                 GL.glUniformMatrix4fv(u_mvp, 1, GL.GL_FALSE, mvp.T)
                 GL.glUniform1f(u_uv_offset_x, lens.uv_offset_x)
                 GL.glUniform1f(u_uv_scale_x, lens.uv_scale_x)
                 
                 GL.glActiveTexture(GL.GL_TEXTURE0)
                 GL.glBindTexture(GL.GL_TEXTURE_2D, lens.camera.tex_id)
                 
                 GL.glActiveTexture(GL.GL_TEXTURE1)
                 GL.glBindTexture(GL.GL_TEXTURE_2D, lens.tex_lookup)
                 
                 GL.glDrawElements(GL.GL_TRIANGLES, indices.size, GL.GL_UNSIGNED_INT, ctypes.c_void_p(0))

        else:
            # Flat Equirect View using Quads
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, q_vbo_pos)
            GL.glEnableVertexAttribArray(a_pos)
            GL.glVertexAttribPointer(a_pos, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, ctypes.c_void_p(0))
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, q_vbo_uv)
            GL.glEnableVertexAttribArray(a_uv)
            GL.glVertexAttribPointer(a_uv, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, ctypes.c_void_p(0))
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, q_ebo)
            
            GL.glUseProgram(prog)
            GL.glUniform1i(u_src, 0)
            GL.glUniform1i(u_lookup, 1)
            
            count = len(lenses)
            cols = math.ceil(math.sqrt(count))
            rows = math.ceil(count / cols)
            
            tile_w = fb_w // cols
            tile_h = fb_h // rows
            
            for i, lens in enumerate(lenses):
                 r = i // cols
                 c = i % cols
                 GL.glViewport(c * tile_w, fb_h - (r+1) * tile_h, tile_w, tile_h)
                 
                 mvp = np.identity(4, dtype=np.float32)
                 mvp[1,1] = -1.0 # Flip Y
                 
                 GL.glUniformMatrix4fv(u_mvp, 1, GL.GL_FALSE, mvp)
                 GL.glUniform1f(u_uv_offset_x, lens.uv_offset_x)
                 GL.glUniform1f(u_uv_scale_x, lens.uv_scale_x)
                 
                 GL.glActiveTexture(GL.GL_TEXTURE0)
                 GL.glBindTexture(GL.GL_TEXTURE_2D, lens.camera.tex_id)
                 GL.glActiveTexture(GL.GL_TEXTURE1)
                 GL.glBindTexture(GL.GL_TEXTURE_2D, lens.tex_lookup)
                 
                 GL.glDrawElements(GL.GL_TRIANGLES, q_indices.size, GL.GL_UNSIGNED_INT, ctypes.c_void_p(0))

        glfw.swap_buffers(window)
        
        frames += 1
        now = time.time()
        if now - last_t >= 2.0:
            fps = frames / (now - last_t)
            #print(f"[perf] FPS={fps:.1f}")
            frames = 0
            last_t = now

    for dev in devices:
        dev.cap.release()
    glfw.terminate()

if __name__ == "__main__":
    main()
