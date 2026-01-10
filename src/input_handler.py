
import glfw
from src.constants import ATTR_NAMES, KEY_ADJUST_LARGE, KEY_ADJUST_SMALL, MOUSE_SENSITIVITY, SCROLL_SENSITIVITY

class InputHandler:
    def __init__(self, app):
        self.app = app
        self.dragging = False
        self.last_x, self.last_y = 0.0, 0.0
    
    def on_key(self, win, key, scancode, action, mods):
        if action not in (glfw.PRESS, glfw.REPEAT): return
        
        if key == glfw.KEY_ESCAPE: 
            glfw.set_window_should_close(win, True)
        
        # Edit Toggle
        if key == glfw.KEY_E:
            if not getattr(self.app, 'lenses', None):
                print("[edit] No active cameras/lenses to edit.")
                self.app.edit_mode = False
                return

            self.app.edit_mode = not self.app.edit_mode
            if self.app.edit_mode:
                print("-" * 40)
                print("EDIT MODE ENABLED")
                print("Controls: [E] Exit/Save, [C] Cycle Cam, [A] Cycle Attr, [+ / -] Adjust")
                cfgs = getattr(self.app, 'lens_configs', None) or self.app.cam_configs
                cam_name = cfgs[self.app.sel_lens_idx].get('name', f'Cam {self.app.sel_lens_idx}')
                print(f"Current Selection: {cam_name} | Attribute: {ATTR_NAMES[self.app.sel_attr_idx]}")
                print("-" * 40)
            else:
                self.app.save_config()
                print("EDIT MODE DISABLED")

        if self.app.edit_mode:
            self._handle_edit_keys(key, mods)
        else:
            self._handle_view_keys(key)

    def _handle_edit_keys(self, key, mods):
        if not self.app.lenses:
            return

        cfgs = getattr(self.app, 'lens_configs', None) or self.app.cam_configs
        lens = self.app.lenses[self.app.sel_lens_idx]
        cam_name = cfgs[self.app.sel_lens_idx].get('name', f'Cam {self.app.sel_lens_idx}')

        if key == glfw.KEY_C:
            self.app.sel_lens_idx = (self.app.sel_lens_idx + 1) % len(self.app.lenses)
            lens = self.app.lenses[self.app.sel_lens_idx]
            cam_name = cfgs[self.app.sel_lens_idx].get('name', f'Cam {self.app.sel_lens_idx}')
            print(f"[edit] Selected Camera: {cam_name}")
            
        elif key == glfw.KEY_A:
            self.app.sel_attr_idx = (self.app.sel_attr_idx + 1) % 4
            print(f"[edit] Selected Attribute: {ATTR_NAMES[self.app.sel_attr_idx]}")
            
        elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD, glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
            delta = KEY_ADJUST_LARGE if key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD) else -KEY_ADJUST_LARGE
            if mods & glfw.MOD_SHIFT:
                    delta = KEY_ADJUST_SMALL if delta > 0 else -KEY_ADJUST_SMALL

            if self.app.sel_attr_idx == 0:   lens.world_yaw += delta
            elif self.app.sel_attr_idx == 1: lens.world_pitch += delta
            elif self.app.sel_attr_idx == 2: lens.world_roll += delta
            elif self.app.sel_attr_idx == 3: lens.orientation += delta
            
            print(f"[edit] {cam_name} {ATTR_NAMES[self.app.sel_attr_idx]} -> {getattr(lens, ['world_yaw', 'world_pitch', 'world_roll', 'orientation'][self.app.sel_attr_idx])}")

    def _handle_view_keys(self, key):
        if key == glfw.KEY_R: 
            self.app.yaw = self.app.pitch = self.app.roll = 0.0; self.app.fov = 70.0
        elif key == glfw.KEY_Q: 
            self.app.roll -= 2.0
        elif key == glfw.KEY_V:
            self.app.view_sphere = not self.app.view_sphere
            print(f"[view] Mode: {'Sphere' if self.app.view_sphere else 'Equirect'}")

    def on_mouse(self, win, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                self.dragging = True
                self.last_x, self.last_y = glfw.get_cursor_pos(win)
            elif action == glfw.RELEASE:
                self.dragging = False

    def on_cursor(self, win, x, y):
        if not self.dragging: return
        dx = x - self.last_x
        dy = y - self.last_y
        self.last_x, self.last_y = x, y
        
        self.app.yaw += dx * MOUSE_SENSITIVITY
        self.app.pitch += dy * MOUSE_SENSITIVITY
        self.app.pitch = max(-89.0, min(89.0, self.app.pitch))

    def on_scroll(self, win, xoff, yoff):
        self.app.fov -= yoff * SCROLL_SENSITIVITY
        self.app.fov = max(20.0, min(180.0, self.app.fov))
