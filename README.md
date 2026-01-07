# Multicam Remap Viewer

A high-performance Python OpenGL viewer for dual-lens fisheye cameras. It remaps raw fisheye video feeds into a corrected equirectangular projection in real-time using cached UV lookup textures.

## Features

- **High-Quality Remapping**: Uses 32-bit floating point UV lookups (RG16F) for smooth linear interpolation and precise unwarping.
- **Hardware Acceleration**: OpenGL-based rendering pipeline handles the heavy lifting of mapping fisheye textures onto a virtual sphere.
- **Dual-Lens Support**: Designed for side-by-side dual fisheye streams (e.g., 2560x720 split into two 1280x720 views).
- **Lookup Caching**: Automatically generates and caches lookup tables as .npy or .png files to speed up startup.
- **Cross-Platform**:
    - **Windows**: Optimized for DirectShow (`cv2.CAP_DSHOW`) to enforce MJPG codec for high-bandwidth USB streaming.
    - **Raspberry Pi 5**: Supports standard V4L2 capture (`cv2.CAP_ANY` or `cv2.CAP_V4L2`), verified with libcamera hardware.
- **Interactive View**:
    - **Mouse Drag**: Pan (Yaw) and Tilt (Pitch).
    - **Scroll**: Zoom (Field of View).
    - **Keys**: Q/E to Roll, R to Reset view, ESC to quit.

## How It Works

1.  **Video Capture**:
    -   Connects to the camera using OpenCV.
    -   **Windows**: Uses a specialized construction method to force **MJPG** compression *before* the stream opens. This is crucial for obtaining 30fps at high resolutions (e.g., 2560x720) over USB, preventing fallback to uncompressed YUY2 (which caps at ~5-10fps).
    -   **Linux/RPi**: Uses standard V4L2 backend.

2.  **Lookup Generation**:
    -   Calculates a mapping from the output equirectangular pixel coordinates back to the input fisheye coordinates.
    -   **Float Mode (Default)**: Creates a high-precision (H, W, 2) RGFloat16 OpenGL texture where channel R is U and G is V. This allows the GPU to linearly interpolate coordinates for the smoothest image.
    -   **Packed Mode (Legacy)**: Compresses 16-bit UV coordinates into an RGBA8 texture. Useful for older GL versions but requires manual unpacking in the shader.

3.  **Rendering**:
    -   The script creates an "inside-out" sphere mesh.
    -   A specialized Fragment Shader samples the Lookup Texture to find which pixel of the raw camera frame to display.

## Usage

### Dependencies
Install the required Python packages:

```bash
pip install -r requirements.txt
```

You will need `opencv-python`, `numpy`, `glfw`, `PyOpenGL`, and `Pillow`.

### Command Line Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--source` | **Required**. Video source. Use `0` for default cam, `dshow:1` for DirectShow index, `v4l2:0` for Linux, or a file path. | N/A |
| `--width` | Requested capture width (e.g., 2560 for dual 1280x720). | None |
| `--height` | Requested capture height (e.g., 720). | None |
| `--hfov` | Horizontal Field of View (in degrees) covered by the width of *one* lens. | 160.0 |
| `--out_w` | Width of the generated lookup texture (pano width). | 1024 |
| `--out_h` | Height of the generated lookup texture (pano height). | 512 |
| `--left_yaw` | Yaw offset for the left lens sphere. | 0.0 |
| `--right_yaw` | Yaw offset for the right lens sphere. | 180.0 |
| `--use_packed` | Use legacy RGBA8 packed lookup instead of RGFloat16. | False |
| `--lookup_path`| Base filename for saved lookup cache. | lookup_autogen.png |

### Examples

**Windows (DirectShow - Force MJPG)**
```bash
python combined_viewer.py --source dshow:1 --width 2560 --height 720
```

**Raspberry Pi 5 / Linux**
```bash
python combined_viewer.py --source 0 --width 2560 --height 720
```

## Troubleshooting

-   **Low FPS / YUY2 Codec**: On Windows, the viewer specifically attempts to force MJPG during initialization. If your camera falls back to YUY2 at high resolutions (like 2560x720), USB bandwidth limits will drop framerate to ~5fps.
-   **DirectShow Device Index**: If `dshow:0` is your webcam, your USB camera might be `dshow:1` or `dshow:2`.
-   **Black Screen**: Check if the `--hfov` matches your lens. If the lookup generation assumes a FOV larger than what the lens projects, pixels may be discarded.
