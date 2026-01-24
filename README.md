# Multicam Remap Viewer

Python + OpenGL viewer that fuses multiple camera feeds (single or dual fisheye) onto an inside-out sphere in real time. Lookups and edge masks are cached so remapping stays GPU-bound and low-latency.

## Current Capabilities
- Multi-camera fusion with per-lens yaw/pitch/roll/orientation and edit-at-runtime controls.
- Distortion models: `fisheye` (equidistant, default) and `corrected` (rectilinear pinhole) per camera.
- Cached resources: float UV lookups (`.npy`) and optional edge masks (`.png`), keyed by camera name, FOV, projection FOV, and distortion model.
- Soft border blending: optional alpha masks for smooth seams; fast discard path when disabled.
- Render modes: inside-sphere view, manual orbit view with grid overlay, and equirect debug view.
- Render modes: inside-sphere view, manual orbit view with grid overlay, equirect debug view, and an "All" grid that shows raw camera feeds.
- Robust startup: cameras that fail to open are skipped with warnings; remaining cameras continue.
- Cross-platform capture: DirectShow MJPG forcing on Windows, V4L2/libcamera/GStreamer on Linux/RPi, plus file playback.

## How It Works
1) **Capture**: OpenCV grabs frames (forcing MJPG on DirectShow when possible) and uploads BGR data directly to an OpenGL texture. Capture runs on a background thread per device.
2) **Lookup + Mask Generation**: CPU builds analytic remap textures from equirect to lens space. Distortion type controls projection math. Optional supersampled edge mask encodes distance to valid region for alpha blending.
3) **Rendering**: A float lookup texture is sampled in the fragment shader to fetch from the source frame. Lenses are drawn back-to-front with optional blending. Orbit mode renders a ground grid first and draws the sphere with depth writes but no depth test for clean layering.

## Configuration

### `config.yaml` (optional, viewer defaults)
```yaml
softborder: false   # true enables alpha blending with edge masks
cache_lookup: true  # reuse lookup/mask caches; false regenerates every run
maskblur: 0         # Gaussian blur kernel for mask (0/1 = none, 3+ softens)
view:
  yaw: 0.0
  pitch: 0.0
  roll: 0.0
  fov: 70.0
mesh:
  sphere_lat_steps: 96
  sphere_lon_steps: 192
  sphere_radius: 10.0
```

### `cameras.yaml`
```yaml
cameras:
  - id: dshow:0            # dshow:N, v4l2:N, integer index, libcamera:N, or file path
    enabled: true          # false skips initialization for this entry
    name: Front Camera
    resolution: [1280, 720]
    type: single           # single | dual_left | dual_right
    fov: 160.0             # lens HFOV in degrees
    distortion: fisheye    # fisheye (default) or corrected (rectilinear)
    mask_mindistance: 0.5  # 0=center ramp, 1=edge-only ramp (optional)
    yaw: 0.0
    pitch: 0.0
    roll: 0.0
    orientation: 0.0       # sensor rotation (0/90/180/270)
    # dshow_query_options: true   # optional ffmpeg probe (slow)
    # dshow_device_name: "Exact DirectShow name"
```

**Field notes:**
- `enabled: false` skips camera initialization while keeping the config entry around.
- `distortion` omitted ⇒ fisheye. Use `corrected` for already-rectified lenses (pinhole projection).
- `type` controls UV slice: `dual_left/right` take half the width of side-by-side stereo feeds.
- `mask_mindistance` adjusts how far from the edge the alpha ramp starts when `softborder` is true.

## Controls
- Mouse drag: Yaw/Pitch (orbit mode drags the outside camera instead).
- Scroll: Zoom (FOV clamp 20–180).
- R: Reset view. Q: Roll. V: Toggle inside ↔ equirect. S: Toggle orbit view. ESC: Quit.
- F: Toggle All view (raw camera grid) on/off. R: Reset view. Q: Roll. V: Toggle inside ↔ equirect. S: Toggle orbit view. ESC: Quit.
- Edit mode (E): C cycle camera, A cycle attribute (Yaw/Pitch/Roll/Orientation), +/- adjust (Shift for fine). Exiting edit saves back to `cameras.yaml`.

## Running
```bash
python main.py --config cameras.yaml        # windowed
python main.py --config cameras.yaml --fullscreen
```

## Troubleshooting
- On Windows use `dshow:N` to force MJPG; YUY2 fallback will tank FPS at high resolutions.
- Black or clipped image: check `fov` and `distortion`; overly small FOV or wrong model will discard pixels.
- Missing cameras: failures are logged and skipped so the app can continue with remaining devices.

## Future Plans
- IMU-driven stabilization and gimbal-less virtual camera control.
- Broader Raspberry Pi testing and CSI/libcamera tuning.
- Optional ffmpeg/recording pipeline.