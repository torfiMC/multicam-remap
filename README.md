# Multicam Remap Viewer

A high-performance Python OpenGL viewer for fusing multiple camera streams into one world view. Supports regular webcams and dual-lens fisheye cameras. It remaps raw fisheye video feeds into a sphere around a virtual camera in real-time using cached UV lookup textures. 

## Written by Torfi Frans Olafsson

I have a background in games graphics programming am an FPV drone hobbyist. Recently I have been putting multiple cameras on my drones using hardware switches to flip between them. However, I have wanted to be able to see the feed from many of them at the same time, but in a way that isn't confusing. So I thought it could be cool to take their feed and fuse it like many VR and 360 cameras do, building a full spherical vision field around a virtual camera and output that to the video transmitter. This does introduce latency and is not suited for agressive acrobatic flight, but for slower platforms, it can work.

Transforming lens data using OpenCV and the CPU is very slow, but graphics hardware is perfect for it. So I thought about pre-computing the lens distortion correction into a lookup table for every pixel, which could be run on a fragment shader. The computation results in a texture that is never show, but contains two 16 bit channels, which represent the remapped U and V coordinates for each pixel. 

## Purpose

This application is being built with the aim of running on a Raspberry Pi 5 mounted on a mobile platform, such as a drone or rover. It is designed to take input from multiple cameras and transmit a single fused video stream back to the operator.

## Future Plans
- **Mobile Platform Camera Fusion**: The goal is for a remote operator to have a single fused view of their surroundings. The operator can then orient the virtual camera via mapped channels on their remote, or other commands, providing a virtual gimbal using multiple fixed cameras. This eliminates the need for gimbal motors or servos, resulting in a more rigid mobile platform with fewer moving parts.

- **Real-time Stabilization**: An aspirational goal is to connect an IMU to the Raspberry Pi and perform real-time stabilization on the feed, shifting the virtual camera to accommodate for movement of the camera platform. This will likely require its own IMU rather than piggybacking on a flight controller, due to latency and bandwidth issues.

- **Pi Camera Support**: Currently, the system uses USB cameras with MJPG compressed feeds, but USB bandwidth saturates quickly when running multiple inputs. Using the two MIPI CSI/DSI ports for cameras would significantly reduce the load on the USB bus.



## Features

- **High-Quality Remapping**: Uses 32-bit floating point UV lookups (RG16F) for smooth linear interpolation and precise unwarping.
- **Hardware Acceleration**: OpenGL-based rendering pipeline handles the heavy lifting of mapping fisheye textures onto a virtual sphere.
- **Dual-Lens Support**: Designed for side-by-side dual fisheye streams (e.g., 2560x720 split into two 1280x720 views).
- **Lookup Caching**: Automatically generates and caches lookup tables as .npy files to speed up startup.
- **Cross-Platform**:
    - **Windows**: Optimized for DirectShow (`cv2.CAP_DSHOW`) to enforce MJPG codec for high-bandwidth USB streaming.
    - **Raspberry Pi 5**: Supports standard V4L2 capture (`cv2.CAP_ANY` or `cv2.CAP_V4L2`), verified with libcamera hardware.
- **Interactive View**:
    - **Mouse Drag**: Pan (Yaw) and Tilt (Pitch).
    - **Scroll**: Zoom (Field of View).
    - **Keys**: Q to Roll, R to Reset view, V for modes, E for Edit Mode, ESC to quit.

## How It Works

1.  **Video Capture**:
    -   Connects to the camera using OpenCV.
    -   **Windows**: Uses a specialized construction method to force **MJPG** compression *before* the stream opens. This is crucial for obtaining 30fps at high resolutions (e.g., 2560x720) over USB, preventing fallback to uncompressed YUY2 (which caps at ~5-10fps).
    -   **Linux/RPi**: Uses standard V4L2 backend.

2.  **Lookup Generation**:
    -   Calculates a mapping from the output equirectangular pixel coordinates back to the input fisheye coordinates.
    -   **Float Mode**: Creates a high-precision (H, W, 2) RGFloat16 OpenGL texture where channel R is U and G is V. This allows the GPU to linearly interpolate coordinates for the smoothest image.

3.  **Rendering**:
    -   The script creates an "inside-out" sphere mesh.
    -   A specialized Fragment Shader samples the Lookup Texture to find which pixel of the raw camera frame to display.
    -   Multiple cameras can be composited into the same sphere view.

## Usage

### 1. Dependencies
Install the required Python packages:

```bash
pip install -r requirements.txt
```

You will need `opencv-python`, `numpy`, `glfw`, `PyOpenGL`, `pyyaml`, and `pillow`.

**Windows Note:** To debug camera capabilities, `ffmpeg` is recommended on your system PATH. The application uses it to list supported resolutions and codecs on startup.

### 2. Configuration (`cameras.yaml`)
Create or edit `cameras.yaml` to define your camera setup. The application supports multiple cameras, including single fisheye lenses or dual-lens setups.

**Example `cameras.yaml`:**
```yaml
cameras:
  - id: dshow:0          # Source ID (dshow:N, v4l2:N, integer index, or file path)
    name: Front Camera   # Display name
    resolution:          # Capture resolution [width, height]
      - 1280
      - 720
    type: single         # 'single', 'dual_left', or 'dual_right'
    fov: 160.0           # Lens Field of View in degrees
    yaw: 0.0             # World Yaw (horizontal rotation)
    pitch: 0.0           # World Pitch (vertical rotation)
    roll: 0.0            # World Roll
    orientation: 0.0     # Sensor rotation (0, 90, 180, 270)
```

**Field Key:**
- `id`: The device source. 
    - Windows DirectShow: `dshow:0`, `dshow:1` (Explicitly uses MJPG forcing logic).
    - Windows/Linux Default: `0`, `1` (Uses `cv2.CAP_ANY`, less control over codec).
    - Linux V4L2: `v4l2:0` (Explicit V4L2 backend).
    - Raspberry Pi Libcamera: `libcamera:0` (Uses GStreamer `libcamerasrc` to access MIPI CSI cameras).
    - File: Absolute or relative path to a video file.
- `resolution`: The list `[width, height]` to request from the camera driver.
- `type`:
    - `single`: Uses the entire frame for the fisheye project.
    - `dual_left`: Takes the left 50% of the frame (common for dual-lens cameras).
    - `dual_right`: Takes the right 50% of the frame.
- `fov`: The field of view of the lens itself.
- `yaw`/`pitch`/`roll`: Position of the camera in the virtual world.
- `orientation`: Rotation of the sensor image itself (0, 90, 180, 270).

**Raspberry Pi Camera Note: (untested)**
To use the native MIPI CSI ports on a Raspberry Pi (e.g., Pi Camera Module 3), use `id: libcamera:0`.
*   **Requirements**: Your OpenCV installation must support GStreamer. The standard `python3-opencv` on Raspberry Pi OS usually works.
*   **Multiple Cameras**: If you have two cameras connected, use `libcamera:0` and `libcamera:1`. The app attempts to automatically resolve the correct camera name.
*   **Console**: You may need to have `rpicam-hello` or `libcamera-hello` installed for the app to detect camera names.

### 3. Running
Start the viewer:
```bash
python main.py --config cameras.yaml
```

**Fullscreen Mode (Raspberry Pi Console / Headless):**
Use the `--fullscreen` flag. If running from a console without a desktop environment (e.g., Raspberry Pi Lite), you can use `startx` to launch a minimal X session:

```bash
# Install minimal X11 requirements
sudo apt-get install xserver-xorg xinit

# Run directly fullscreen
startx python main.py --fullscreen --config cameras.yaml
```

## Controls

### View Mode
- **Mouse Drag**: Pan (Yaw) and Tilt (Pitch) the view.
- **Scroll**: Zoom In/Out (Field of View).
- **R**: Reset view to default.
- **V**: Toggle viewing mode (Sphere vs Equirectangular/Debug).
- **Q**: Roll view.
- **E**: Enter Edit Mode.
- **ESC**: Quit.

### Edit Mode
Press **E** to toggle Edit Mode. This allows you to calibrate camera positions in real-time. The console will display the currently selected camera and attribute.
- **C**: Cycle through selected cameras (if multiple are defined).
- **A**: Cycle through attributes to adjust (Yaw, Pitch, Roll, Orientation).
- **+ / -**: Adjust the selected attribute.
    - **Shift + (+/-)**: Fine adjustment (smaller steps).
- **E**: Exit Edit Mode and **save changes** to `cameras.yaml` automatically.

## Troubleshooting

-   **Low FPS / YUY2 Codec**: On Windows, use `dshow:N` for the id in `cameras.yaml`. The viewer specifically attempts to force MJPG during initialization for `dshow` targets. If your camera falls back to YUY2 at high resolutions (like 2560x720), USB bandwidth limits will drop framerate to ~5fps.
-   **DirectShow Device Index**: If `dshow:0` is your webcam, your USB camera might be `dshow:1` or `dshow:2`. Use the `ffmpeg -list_options true -f dshow -i video="...` command or trial-and-error to find the right index.
-   **Black Screen**: Check if the `fov` in config matches your lens. If the lookup generation assumes a FOV larger than what the lens projects, pixels may be discarded.

## Todo

-   **Raspberry Pi**: The project hasn't been fully tested on Raspberry Pi, so it is assumed that there is functionality still missing or broken.