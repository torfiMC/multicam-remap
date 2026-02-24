import subprocess
import threading
from typing import Optional


class FFmpegRecorder:
    """Simple ffmpeg pipe recorder for raw BGR frames."""

    def __init__(self, output_path: str, width: int, height: int, fps: float = 30.0, preset: str = "veryfast"):
        self.output_path = output_path
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.preset = preset
        self.proc: Optional[subprocess.Popen] = None
        self.lock = threading.Lock()
        self._start()

    def _start(self) -> None:
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            self.preset,
            "-pix_fmt",
            "yuv420p",
            self.output_path,
        ]
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            raise RuntimeError("ffmpeg not found in PATH; recording unavailable")

    def write_frame(self, frame_bytes: bytes) -> None:
        if not self.proc or not self.proc.stdin:
            return
        with self.lock:
            try:
                self.proc.stdin.write(frame_bytes)
            except Exception:
                self.stop()

    def stop(self) -> None:
        if self.proc:
            try:
                if self.proc.stdin:
                    self.proc.stdin.close()
            finally:
                try:
                    self.proc.terminate()
                except Exception:
                    pass
            self.proc = None
