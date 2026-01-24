import subprocess
import sys
from typing import Dict, List, Optional, Tuple

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None  # fallback when OpenCV is not installed

if cv2 is not None:
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:
        pass


def list_ffmpeg_devices() -> Tuple[List[str], Optional[str]]:
    """Return device names reported by ffmpeg dshow probe, or error text."""
    cmd = [
        "ffmpeg",
        "-list_devices",
        "true",
        "-f",
        "dshow",
        "-i",
        "dummy",
    ]
    try:
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except FileNotFoundError:
        return [], "ffmpeg not found in PATH"
    except Exception as exc:  # pragma: no cover
        return [], f"ffmpeg probe failed: {exc}"

    lines = res.stderr.splitlines()
    devices: List[str] = []
    collecting = False
    for line in lines:
        if "DirectShow video devices" in line:
            collecting = True
            continue
        if collecting and "DirectShow audio devices" in line:
            break
        if collecting:
            # Expected form: [dshow @ ...]  "Device Name"
            if '"' in line:
                parts = line.split('"')
                if len(parts) >= 2:
                    devices.append(parts[1].strip())
    return devices, None


def fourcc_to_str(code: int) -> str:
    chars = [chr((int(code) >> 8 * i) & 0xFF) for i in range(4)]
    text = "".join(chars)
    return text if text.strip("\x00\x01\x02\x03\x04\x05\x06\x07") else "n/a"


def probe_opencv(max_devices: int = 10, names: Optional[List[str]] = None) -> Tuple[List[Dict[str, object]], Optional[str]]:
    """Fallback: attempt to open sequential dshow indices and report details."""
    if cv2 is None:
        return [], "OpenCV not installed; cannot fallback probe"

    found: List[Dict[str, object]] = []
    for idx in range(max_devices):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        try:
            if not cap.isOpened():
                continue

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            fourcc_raw = int(cap.get(cv2.CAP_PROP_FOURCC)) if hasattr(cv2, "CAP_PROP_FOURCC") else 0
            backend = cap.getBackendName() if hasattr(cap, "getBackendName") else "unknown"

            name = names[idx] if names and idx < len(names) else f"Camera {idx}"

            found.append({
                "index": idx,
                "name": name,
                "width": width,
                "height": height,
                "fps": fps,
                "fourcc": fourcc_to_str(fourcc_raw),
                "backend": backend,
            })
        finally:
            cap.release()
    return found, None if found else "no devices opened via OpenCV"


def main() -> int:
    devices, err = list_ffmpeg_devices()
    if devices:
        print("Discovered cameras (ffmpeg dshow):")
        for i, name in enumerate(devices):
            print(f"  dshow:{i} - {name}")
    else:
        print("ffmpeg discovery unavailable", end="")
        if err:
            print(f" ({err})")
        else:
            print()

    fallback, fb_err = probe_opencv(names=devices if devices else None)
    if fallback:
        print("Discovered cameras (OpenCV fallback):")
        for info in fallback:
            print(
                f"  dshow:{info['index']} - {info['name']} | "
                f"backend={info['backend']} | "
                f"size={info['width']}x{info['height']} | "
                f"fps={info['fps']:.2f} | "
                f"fourcc={info['fourcc']}"
            )
        return 0

    print(f"No cameras found. Last error: {fb_err or 'unknown'}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
