import os
import glob
import logging
import subprocess
import threading
import time

import cv2
import numpy as np
from OpenGL import GL

log = logging.getLogger(__name__)


def cap_get_fourcc(cap):
    if cap is None or not cap.isOpened():
        return "----"
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    return "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])


def _configure_device(cap, width, height, fourcc_fmt, fps):
    if fourcc_fmt:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc_fmt))
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        cap.set(cv2.CAP_PROP_FPS, float(fps))


def _opencv_has_gstreamer():
    try:
        info = cv2.getBuildInformation()
        return "GStreamer" in info
    except Exception:
        return False


def get_libcamera_name_by_index(idx: int) -> str:
    try:
        proc = subprocess.run(
            ["libcamera-hello", "--list-cameras"],
            check=False,
            capture_output=True,
            text=True,
        )
        lines = proc.stdout.strip().splitlines()
        for line in lines:
            if line.strip().startswith(str(idx)) and ":" in line:
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return str(idx)


def make_libcamera_pipeline(width, height, fps, cam_name, fmt):
    fmt_str = (fmt or "yuv420").lower()
    w_cmd = width or 1280
    h_cmd = height or 720
    fps_cmd = fps or 30
    return (
        f"libcamerasrc camera-name=\"{cam_name}\" ! video/x-raw,format={fmt_str},width={w_cmd},height={h_cmd},framerate={fps_cmd}/1 "
        f"! videoconvert ! appsink drop=1"
    )


def query_dshow_options_ffmpeg(idx: int):
    try:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        log.warning("[video] ffmpeg not found; cannot list dshow devices")


class FFmpegPipeCapture:
    def __init__(self, cmd: str, width: int, height: int, fps: int = None, pix_fmt: str = "bgr24"):
        self.cmd = cmd
        self.width = width
        self.height = height
        self.fps = fps
        self.pix_fmt = pix_fmt
        self.frame_size = width * height * 3  # bgr24
        self.process = None
        self.stdout = None
        self._opened = False
        self._start_process()

    def _start_process(self):
        try:
            self.process = subprocess.Popen(
                self.cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=self.frame_size * 4,
            )
            self.stdout = self.process.stdout
            self._opened = True if self.stdout else False
        except Exception as exc:
            log.error(f"[pipe] Failed to start pipe: {exc}")
            self._opened = False

    def isOpened(self):
        return self._opened and self.process is not None and self.process.poll() is None

    def read(self):
        if not self.isOpened():
            return False, None
        try:
            data = self.stdout.read(self.frame_size)
            if not data or len(data) < self.frame_size:
                return False, None
            frame = np.frombuffer(data, dtype=np.uint8)
            frame = frame.reshape((self.height, self.width, 3))
            return True, frame
        except Exception as exc:
            log.warning(f"[pipe] Read error: {exc}")
            return False, None

    def release(self):
        self._opened = False
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=1.0)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
        if self.stdout:
            try:
                self.stdout.close()
            except Exception:
                pass

    # Mimic minimal VideoCapture API for compatibility
    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        if prop_id == cv2.CAP_PROP_FPS:
            return float(self.fps) if self.fps is not None else 0.0
        if prop_id == cv2.CAP_PROP_FOURCC:
            return float(cv2.VideoWriter_fourcc(*"BGR3"))
        return 0.0

    def set(self, prop_id, value):
        # No-op; return False to match VideoCapture behavior on unsupported props
        return False


def open_video_source(src: str, width: int = None, height: int = None, fmt: str = None, fps: int = None, prefer_yuv_pipe: bool = False):
    fourcc_fmt = None
    fmt_normalized = None if fmt is None else str(fmt).strip()
    if fmt_normalized:
        fmt_upper = fmt_normalized.upper()
        if len(fmt_upper) == 4:
            fourcc_fmt = fmt_upper
        fmt = fmt_upper
    else:
        fmt = None

    fps_to_use = fps if fps is not None else 30
    if src.startswith("/dev/video"):
        devs = sorted(glob.glob("/dev/video*"))
        log.info(f"[video] Detected V4L2 nodes: {devs}")
    log.info(f"[video] Opening source {src} width={width} height={height} fmt={fmt} fps={fps_to_use} fourcc={fourcc_fmt}")

    cap = None
    is_device = False

    if src.startswith("rpicam-vid") or src.startswith("libcamera-vid"):
        idx = 0
        try:
            idx = int(src.split(":", 1)[1])
        except Exception:
            pass
        fps_cmd = fps_to_use or 30
        w_cmd = width or 1280
        h_cmd = height or 720

        if src.startswith("rpicam-vid"):
            cmd_name = "rpicam-vid"
            use_bgr_first = not prefer_yuv_pipe

            if use_bgr_first:
                pipe_cmd_bgr = (
                    f"{cmd_name} --timeout 0 --camera {idx} --width {w_cmd} --height {h_cmd} "
                    f"--framerate {fps_cmd} --codec bgr --libav-format rawvideo --output - --nopreview"
                )
                log.info(f"[video] Using {cmd_name} raw BGR pipe: {pipe_cmd_bgr}")
                cap = FFmpegPipeCapture(pipe_cmd_bgr, w_cmd, h_cmd, fps_cmd)
                is_device = False

                if cap.isOpened():
                    return cap

                log.warning("[video] BGR pipe failed; falling back to yuv420 -> ffmpeg pipeline")

            pipe_cmd = (
                f"{cmd_name} --timeout 0 --camera {idx} --width {w_cmd} --height {h_cmd} "
                f"--framerate {fps_cmd} --codec yuv420 --libav-format rawvideo --output - --nopreview | "
                f"ffmpeg -f rawvideo -pix_fmt yuv420p -s {w_cmd}x{h_cmd} -r {fps_cmd} -i pipe:0 "
                f"-f rawvideo -pix_fmt bgr24 pipe:1"
            )
            log.info(f"[video] Using {cmd_name} pipe: {pipe_cmd}")
            cap = cv2.VideoCapture(pipe_cmd, cv2.CAP_FFMPEG)
            is_device = False

            if not cap.isOpened():
                log.warning("[video] OpenCV FFMPEG backend failed; using raw pipe reader fallback")
                cap = FFmpegPipeCapture(pipe_cmd, w_cmd, h_cmd, fps_cmd)
                if not cap.isOpened():
                    log.error(f"[video] Pipe reader fallback failed to start for {cmd_name}")
                    raise RuntimeError(f"Failed to open video source: {src}")
        else:
            cmd_name = "libcamera-vid"
            pipe_cmd = (
                f"{cmd_name} -t 0 --camera {idx} --width {w_cmd} --height {h_cmd} "
                f"--framerate {fps_cmd} --codec yuv420 --inline --stdout | "
                f"ffmpeg -f rawvideo -pix_fmt yuv420p -s {w_cmd}x{h_cmd} -r {fps_cmd} -i pipe:0 "
                f"-f rawvideo -pix_fmt bgr24 pipe:1"
            )
            log.info(f"[video] Using {cmd_name} pipe: {pipe_cmd}")
            cap = cv2.VideoCapture(pipe_cmd, cv2.CAP_FFMPEG)
            is_device = False

            if not cap.isOpened():
                log.warning("[video] OpenCV FFMPEG backend failed; using raw pipe reader fallback")
                cap = FFmpegPipeCapture(pipe_cmd, w_cmd, h_cmd, fps_cmd)
                if not cap.isOpened():
                    log.error(f"[video] Pipe reader fallback failed to start for {cmd_name}")
                    raise RuntimeError(f"Failed to open video source: {src}")

    elif src.startswith("dshow:"):
        idx = int(src.split(":", 1)[1])
        query_dshow_options_ffmpeg(idx)
        if width is not None and height is not None:
            log.info(f"[video] Opening DSHOW device {idx} with explicit params in constructor...")
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW, [
                cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*((fourcc_fmt or "MJPG"))),
                cv2.CAP_PROP_FRAME_WIDTH, width,
                cv2.CAP_PROP_FRAME_HEIGHT, height,
                cv2.CAP_PROP_FPS, fps_to_use,
            ])
        else:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        is_device = True

    elif src.startswith("libcamera:"):
        log.info("[video] Initializing libcamera (GStreamer)...")
        idx = int(src.split(":", 1)[1])
        cam_name = get_libcamera_name_by_index(idx)
        log.info(f"[video] Resolved libcamera index {idx} to name: {cam_name}")

        if not _opencv_has_gstreamer():
            log.warning("[video] OpenCV build lacks GStreamer; falling back to V4L2 device")
            fallback_src = f"/dev/video{idx}" if os.name != "nt" else str(idx)
            cap = cv2.VideoCapture(fallback_src)
            is_device = True
        else:
            pipeline = make_libcamera_pipeline(width, height, fps_to_use, cam_name, fmt)
            log.info(f"[video] GStreamer Pipeline: {pipeline}")
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if not cap.isOpened():
                log.warning("[video] Failed to open libcamera via GStreamer; falling back to V4L2 device")
                fallback_src = f"/dev/video{idx}" if os.name != "nt" else str(idx)
                cap = cv2.VideoCapture(fallback_src)
                is_device = True
            else:
                is_device = False

    else:
        try:
            if src.startswith("v4l2:"):
                idx = int(src.split(":", 1)[1])
            else:
                idx = int(src)

            cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
            if cap.isOpened():
                _configure_device(cap, width, height, fourcc_fmt, fps_to_use)
            is_device = True
        except ValueError:
            if src.startswith("/dev/"):
                log.info(f"[video] Opening V4L2 device path: {src}")
                cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
                if cap.isOpened():
                    _configure_device(cap, width, height, fourcc_fmt, fps_to_use)
                is_device = True
            else:
                cap = cv2.VideoCapture(src)
                is_device = False

    if cap is None or not cap.isOpened():
        resolved = None
        try:
            if src.startswith("/dev/v4l/by-path") and os.path.islink(src):
                resolved = os.path.realpath(src)
        except Exception:
            resolved = None

        candidates = []
        if resolved and resolved.startswith("/dev/video"):
            candidates.append(resolved)
        if src.startswith("/dev/video"):
            candidates.append(src)

        tried_idx = set()
        for cand in candidates:
            try:
                idx = int(cand.replace("/dev/video", ""))
            except Exception:
                idx = None
            backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
            if idx is not None and idx not in tried_idx:
                tried_idx.add(idx)
                for be in backends:
                    log.info(f"[video] Fallback open numeric index {idx} backend={be} (from {cand})")
                    cap = cv2.VideoCapture(idx, be)
                    if cap.isOpened():
                        _configure_device(cap, width, height, fourcc_fmt, fps_to_use)
                        is_device = True
                        break
                if cap.isOpened():
                    break

    if cap is None or not cap.isOpened():
        log.error(f"[video] Failed to open video source: {src}")
        raise RuntimeError(f"Failed to open video source: {src}")

    if is_device and cap.isOpened():
        fourcc_now = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec_now = "".join([chr((fourcc_now >> 8 * i) & 0xFF) for i in range(4)])

        if fourcc_fmt and codec_now != fourcc_fmt:
            log.info(f"[video] Constructor params didn't force {fourcc_fmt} (got {codec_now}). Trying legacy set()...")
            _configure_device(cap, width, height, fourcc_fmt, fps_to_use)

        if fps_to_use and cap.get(cv2.CAP_PROP_FPS) <= 0:
            cap.set(cv2.CAP_PROP_FPS, float(fps_to_use))

    if width is not None and height is not None:
        actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        log.info(f"[debug] Requested {width}x{height}, got {actual_w}x{actual_h}")

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {src}")

    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    log.info(f"[video] Active FOURCC: '{codec}' (Int: {fourcc})")

    fps_val = cap.get(cv2.CAP_PROP_FPS)
    log.info(f"[video] Active FPS: {fps_val}")

    return cap


class CameraDevice:
    """Represents a physical video capture device"""

    def __init__(self, config_dict):
        self.id_str = str(config_dict.get("id", 0))
        self.name = config_dict.get("name", f"Cam_{self.id_str}")
        self.res = config_dict.get("resolution", [1280, 720])
        self.width, self.height = self.res[0], self.res[1]
        self.is_dual = (config_dict.get("type", "single") == "dual")
        default_fmt = "BGR" if str(self.id_str).startswith("libcamera:") else "MJPG"
        fmt_cfg = config_dict.get("format", default_fmt)
        if fmt_cfg is None or str(fmt_cfg).strip().lower() in ("auto", "", "none"):
            self.format = None
        else:
            self.format = str(fmt_cfg).upper()

        self.upload_format = self.format if self.format in ("RGB", "BGR", "GRAY") else "BGR"
        self.fps = int(config_dict.get("fps", 30)) if config_dict.get("fps") is not None else 30
        self.convert_code = None

        self.yuy_map = {
            "YUYV": cv2.COLOR_YUV2BGR_YUY2,
            "YUY2": cv2.COLOR_YUV2BGR_YUY2,
            "UYVY": cv2.COLOR_YUV2BGR_UYVY,
        }
        if self.format in self.yuy_map:
            self.convert_code = self.yuy_map[self.format]
            self.upload_format = "BGR"

        self.cap = open_video_source(self.id_str, self.width, self.height, self.format, self.fps)
        log.info(f"[video] Device opened id={self.id_str} fmt={self.format} req_res={self.width}x{self.height} fps={self.fps}")

        ret, frame = False, None
        for _ in range(30):
            ret, frame = self.cap.read()
            if ret:
                break
            time.sleep(0.05)

        if ret and self.convert_code is not None:
            try:
                frame = cv2.cvtColor(frame, self.convert_code)
            except Exception as e:
                raise RuntimeError(f"Color convert failed for camera {self.id_str}: {e}")

        if not ret:
            log.warning(
                f"[video] Initial frame read failed for {self.id_str} at {self.width}x{self.height} fmt={self.format}. Current props: "
                f"fourcc={cap_get_fourcc(self.cap)} fps={self.cap.get(cv2.CAP_PROP_FPS)} res={self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}"
            )

            if self.id_str.startswith("rpicam-vid"):
                log.warning("[video] Retrying rpicam-vid with yuv420 -> ffmpeg pipeline after BGR pipe read failure")
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = open_video_source(self.id_str, self.width, self.height, self.format, self.fps, prefer_yuv_pipe=True)

                ret, frame = False, None
                for _ in range(30):
                    ret, frame = self.cap.read()
                    if ret:
                        break
                    time.sleep(0.05)

                if ret and self.convert_code is not None:
                    try:
                        frame = cv2.cvtColor(frame, self.convert_code)
                    except Exception as e:
                        raise RuntimeError(f"Color convert failed for camera {self.id_str}: {e}")

                if not ret:
                    raise RuntimeError(f"Failed to read initial frame from camera {self.id_str} after yuv fallback")

            elif self.id_str.startswith("libcamera-vid"):
                raise RuntimeError(f"Failed to read initial frame from camera {self.id_str}")

            if not ret:
                fallback_formats = ["BGR3", "RGB3", "YUYV", "UYVY", None]
                tried = [self.format]
                reopened = False
                for fmt in fallback_formats:
                    if fmt in tried:
                        continue
                    try:
                        log.info(f"[video] Retrying open of {self.id_str} with format {fmt}...")
                        self.cap.release()
                        self.cap = open_video_source(self.id_str, self.width, self.height, fmt, self.fps)
                        self.format = fmt
                        self.upload_format = fmt if fmt in ("RGB", "BGR", "GRAY") else "BGR"
                        if fmt in self.yuy_map:
                            self.convert_code = self.yuy_map[fmt]
                            self.upload_format = "BGR"

                        for _ in range(30):
                            ok, frame = self.cap.read()
                            if ok:
                                ret = True
                                break
                            time.sleep(0.05)

                        if ret:
                            reopened = True
                            break

                        self.cap.release()
                        log.info(f"[video] Retrying {self.id_str} at 640x480 fmt={fmt}...")
                        self.cap = open_video_source(self.id_str, 640, 480, fmt, self.fps)
                        for _ in range(30):
                            ok, frame = self.cap.read()
                            if ok:
                                ret = True
                                break
                            time.sleep(0.05)

                        if ret:
                            reopened = True
                            break
                    except Exception as e:
                        log.warning(f"[video] Fallback open with format {fmt} failed: {e}")
                        tried.append(fmt)
                        continue

                if not reopened:
                    raise RuntimeError(f"Failed to read initial frame from camera {self.id_str}")

        if not ret or frame is None:
            raise RuntimeError(f"Failed to obtain initial frame for camera {self.id_str} (frame is None)")

        self.actual_h, self.actual_w = frame.shape[:2]
        channels = 1 if frame.ndim == 2 else frame.shape[2]
        self.gl_format = GL.GL_RGB
        if self.upload_format == "BGR" and hasattr(GL, "GL_BGR"):
            self.gl_format = GL.GL_BGR
        elif self.upload_format == "RGB" and hasattr(GL, "GL_RGB"):
            self.gl_format = GL.GL_RGB
        elif self.upload_format == "GRAY":
            self.gl_format = GL.GL_RED if hasattr(GL, "GL_RED") else GL.GL_LUMINANCE
            channels = 1
        internal_format = GL.GL_RGB if channels > 1 else self.gl_format

        self.tex_id = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex_id)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, internal_format, self.actual_w, self.actual_h, 0, self.gl_format, GL.GL_UNSIGNED_BYTE, None)

        self.last_frame = frame
        self.new_frame_ready = True

        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        while self.running:
            if not self.cap.isOpened():
                break

            ret, frame = self.cap.read()
            if ret:
                if self.convert_code is not None:
                    try:
                        frame = cv2.cvtColor(frame, self.convert_code)
                    except Exception:
                        continue
                with self.lock:
                    self.last_frame = frame
                    self.new_frame_ready = True
            else:
                if not self.id_str.startswith("dshow") and not self.id_str.isdigit():
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else:
                    time.sleep(0.01)

    def update(self):
        pass

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()

    def upload_texture(self):
        if self.new_frame_ready:
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex_id)
            GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
            GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, self.actual_w, self.actual_h, self.gl_format, GL.GL_UNSIGNED_BYTE, self.last_frame)
            self.new_frame_ready = False
