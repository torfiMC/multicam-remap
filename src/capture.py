import cv2
import subprocess
import numpy as np
from OpenGL import GL

def query_dshow_options_ffmpeg(device_index: int):
    """
    Uses ffmpeg to list DirectShow device options for the given index.
    Outputs to console.
    """
    print(f"[video] Querying capabilities for dshow index {device_index} via ffmpeg...")
    try:
        # HARDCODED override for debugging as per user request
        dev_name = "USB Camera"
        print(f"[video] Using hardcoded device name: '{dev_name}'")
        
        cmd_opt_str = f'ffmpeg -list_options true -f dshow -i video="{dev_name}"'
        res_opt = subprocess.run(cmd_opt_str, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace', shell=True)
        
        print(f"[video] Supported formats for '{dev_name}':")
        for line in res_opt.stderr.splitlines():
            if "pixel_format" in line or "vcodec" in line or "fps" in line:
                 txt = line.split("]", 1)[-1].strip() if "]" in line else line.strip()
                 print(f"  {txt}")

    except FileNotFoundError:
        print("[video] ffmpeg not found in PATH. Cannot listing capabilities.")
    except Exception as e:
        print(f"[video] Error querying dshow options: {e}")


def open_video_source(src: str, width: int = None, height: int = None):
    # src:
    #   "0"            default camera
    #   "dshow:0"      DirectShow
    #   "path\to\file"
    if src.startswith("dshow:"):
        idx = int(src.split(":", 1)[1])
        
        # Query options BEFORE opening, to avoid busy device issues
        query_dshow_options_ffmpeg(idx)
        
        if width is not None and height is not None:
            print(f"[video] Opening DSHOW device {idx} with explicit params in constructor...")
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW, [
                cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'),
                cv2.CAP_PROP_FRAME_WIDTH, width,
                cv2.CAP_PROP_FRAME_HEIGHT, height,
                cv2.CAP_PROP_FPS, 30
            ])
        else:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            
        is_device = True
    else:
        try:
            # Check for v4l2 prefix or simple index
            if src.startswith("v4l2:"):
                idx = int(src.split(":", 1)[1])
            else:
                idx = int(src)
            
            if width is not None and height is not None:
                 print(f"[video] Opening device {idx} with explicit params...")
                 cap = cv2.VideoCapture(idx, cv2.CAP_ANY, [
                    cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'),
                    cv2.CAP_PROP_FRAME_WIDTH, width,
                    cv2.CAP_PROP_FRAME_HEIGHT, height,
                    cv2.CAP_PROP_FPS, 30
                ]) 
            else:
                cap = cv2.VideoCapture(idx)
            is_device = True
        except ValueError:
            cap = cv2.VideoCapture(src)
            is_device = False

    if is_device:
        fourcc_now = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec_now = "".join([chr((fourcc_now >> 8 * i) & 0xFF) for i in range(4)])
        
        if codec_now != 'MJPG':
             print(f"[video] Constructor params didn't force MJPG (got {codec_now}). Trying legacy set()...")
             cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
             cap.set(cv2.CAP_PROP_FPS, 30.0)
             if width: cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
             if height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if width is not None and height is not None:
        actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"[debug] Requested {width}x{height}, got {actual_w}x{actual_h}")
        
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {src}")

    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    print(f"[video] Active FOURCC: '{codec}' (Int: {fourcc})")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[video] Active FPS: {fps}")

    return cap


class CameraDevice:
    """Represents a physical video capture device"""
    def __init__(self, config_dict):
        self.id_str = str(config_dict.get("id", 0))
        self.name = config_dict.get("name", f"Cam_{self.id_str}")
        self.res = config_dict.get("resolution", [1280, 720])
        self.width, self.height = self.res[0], self.res[1]
        self.is_dual = (config_dict.get("type", "single") == "dual")
        self.format = config_dict.get("format", "MJPG")
        
        # Initialize Source
        self.cap = open_video_source(self.id_str, self.width, self.height)
        
        # Verify resolution, read one frame
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read initial frame from camera {self.id_str}")
        self.actual_h, self.actual_w = frame.shape[:2]
        
        # OpenGL Texture
        self.tex_id = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex_id)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        # Allocate
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB, self.actual_w, self.actual_h, 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, None)
        
        self.last_frame = frame
        self.new_frame_ready = True

    def update(self):
        ret, frame = self.cap.read()
        if ret:
            self.last_frame = frame
            self.new_frame_ready = True
        else:
             # Loop file sources
             if not self.id_str.startswith("dshow") and not self.id_str.isdigit():
                 self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def upload_texture(self):
        if self.new_frame_ready:
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex_id)
            # BGR to RGB
            rgb = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2RGB)
            GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, self.actual_w, self.actual_h, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, rgb)
            self.new_frame_ready = False
