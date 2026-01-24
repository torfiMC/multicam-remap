import asyncio
import threading
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn


class CameraUpdate(BaseModel):
  enabled: Optional[bool] = None
  yaw: Optional[float] = None
  pitch: Optional[float] = None
  roll: Optional[float] = None
  orientation: Optional[float] = None
  fov: Optional[float] = None
  mask_mindistance: Optional[float] = None
  distortion: Optional[str] = None
  rebuild_lookup: bool = False
  save: bool = True


class ViewUpdate(BaseModel):
  yaw: Optional[float] = None
  pitch: Optional[float] = None
  roll: Optional[float] = None
  fov: Optional[float] = None
  reset: bool = False


class ConnectionManager:
    def __init__(self) -> None:
        self.active: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self.active.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self.active.discard(websocket)

    async def broadcast(self, message: dict) -> None:
        async with self._lock:
            targets = list(self.active)
        for ws in targets:
            try:
                await ws.send_json(message)
            except Exception:
                try:
                    await ws.close()
                finally:
                    async with self._lock:
                        self.active.discard(ws)

    def has_clients(self) -> bool:
        return bool(self.active)


class ControlServer:
    def __init__(self, renderer_app, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.renderer_app = renderer_app
        self.host = host
        self.port = port
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._server: Optional[uvicorn.Server] = None
        self.manager = ConnectionManager()

        self.fastapi_app = FastAPI(title="Multicam Control", version="0.1.0")
        self.fastapi_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self._setup_routes()

    def _setup_routes(self) -> None:
        @self.fastapi_app.get("/")
        async def root():
            return HTMLResponse(content=self._html_page(), media_type="text/html")

        @self.fastapi_app.get("/api/status")
        async def status():
            return self.renderer_app.renderer_status()

        @self.fastapi_app.get("/api/cameras")
        async def cameras():
            return self.renderer_app.describe_cameras()

        @self.fastapi_app.get("/api/view")
        async def view_state():
          return self.renderer_app.describe_view()

        @self.fastapi_app.post("/api/view")
        async def update_view(payload: ViewUpdate):
          data = payload.dict(exclude_unset=True)
          try:
            result = self.renderer_app.run_on_main(
              lambda: self.renderer_app.apply_view_update(data)
            )
          except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

          self.queue_broadcast()
          return result

        @self.fastapi_app.get("/api/snapshot")
        async def snapshot():
            try:
                return self.renderer_app.run_on_main(lambda: self.renderer_app.snapshot_cameras())
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        @self.fastapi_app.post("/api/cameras/{idx}")
        async def update_camera(idx: int, payload: CameraUpdate):
            data = payload.dict(exclude_unset=True)
            try:
                result = self.renderer_app.run_on_main(
                    lambda: self.renderer_app.apply_camera_update(idx, data, save=payload.save)
                )
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

            self.queue_broadcast()
            return result

        @self.fastapi_app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.manager.connect(websocket)
            try:
                await websocket.send_json(self._full_state())
                while True:
                    await websocket.receive_text()
            except WebSocketDisconnect:
                await self.manager.disconnect(websocket)
            except Exception:
                await self.manager.disconnect(websocket)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        config = uvicorn.Config(self.fastapi_app, host=self.host, port=self.port, log_level="info")
        self._server = uvicorn.Server(config)
        self._loop.run_until_complete(self._server.serve())

    def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(lambda: None)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _full_state(self) -> dict:
        return {
            "status": self.renderer_app.renderer_status(),
            "cameras": self.renderer_app.describe_cameras(),
        "view": self.renderer_app.describe_view(),
        }

    def queue_broadcast(self) -> None:
        if self._loop and self._loop.is_running() and self.manager.has_clients():
            asyncio.run_coroutine_threadsafe(self.manager.broadcast(self._full_state()), self._loop)

    def _html_page(self) -> str:
        return """<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>Multicam Control</title>
  <link rel=\"preconnect\" href=\"https://fonts.googleapis.com\">
  <link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin>
  <link href=\"https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&display=swap\" rel=\"stylesheet\">
    <button id="snapshotBtn" class="secondary">Snapshot</button>
  <style>
    :root {
      --bg: #0f172a;
      --panel: #111827;
      --accent: #0ea5e9;
      --accent-2: #f59e0b;
      --text: #e5e7eb;
      --muted: #9ca3af;
      --border: #1f2937;
      --shadow: 0 10px 35px rgba(0,0,0,0.35);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: 'Space Grotesk', 'Segoe UI', sans-serif;
      background: radial-gradient(circle at 20% 20%, rgba(14,165,233,0.15), transparent 25%),
                  radial-gradient(circle at 80% 0%, rgba(245,158,11,0.15), transparent 20%),
                  var(--bg);
      color: var(--text);
      min-height: 100vh;
      padding: 1rem;
    }
    header {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 0.5rem;
      margin-bottom: 1rem;
    }
    .title {
      font-size: 1.4rem;
      font-weight: 600;
      letter-spacing: 0.01em;
    }
    .status {
      display: flex;
      gap: 0.5rem;
      flex-wrap: wrap;
    }
    .pill {
      padding: 0.4rem 0.75rem;
      border-radius: 999px;
      background: rgba(255,255,255,0.05);
      border: 1px solid var(--border);
      color: var(--muted);
      font-size: 0.9rem;
    }
    .pill strong { color: var(--text); }
    #cams {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 1rem;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 1rem;
      box-shadow: var(--shadow);
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }
    .card h3 {
      margin: 0;
      font-size: 1.05rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.5rem;
    }
    .badge {
      background: rgba(14,165,233,0.1);
      color: var(--text);
      padding: 0.2rem 0.6rem;
      border-radius: 10px;
      border: 1px solid rgba(14,165,233,0.4);
      font-size: 0.8rem;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 0.5rem;
    }
    label { font-size: 0.85rem; color: var(--muted); }
    input[type=number] {
      width: 100%;
      padding: 0.5rem 0.6rem;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: #0b1221;
      color: var(--text);
      font-size: 0.95rem;
    }
    .row { display: flex; align-items: center; gap: 0.5rem; justify-content: space-between; }
    .toggle { width: 20px; height: 20px; }
    .actions { display: flex; flex-wrap: wrap; gap: 0.5rem; }
    button {
      border: none;
      border-radius: 10px;
      padding: 0.55rem 0.8rem;
      font-weight: 600;
      cursor: pointer;
      color: var(--text);
      background: var(--accent);
      box-shadow: 0 6px 20px rgba(14,165,233,0.35);
      transition: transform 120ms ease, box-shadow 120ms ease;
    }
    button.secondary {
      background: rgba(255,255,255,0.05);
      border: 1px solid var(--border);
      box-shadow: none;
    }
    button:active { transform: translateY(1px); }
    .muted { color: var(--muted); font-size: 0.85rem; }
    .error { color: #f87171; font-size: 0.9rem; }
    .snap {
      width: 100%;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: #0b1221;
      object-fit: contain;
      max-height: 220px;
    }
  </style>
</head>
<body>
  <header>
    <div class=\"title\">Multicam Control Panel</div>
    <div class=\"status\">
      <div class=\"pill\">Mode: <strong id=\"mode\">--</strong></div>
      <div class=\"pill\">Active: <strong id=\"active\">0</strong></div>
      <div class=\"pill\">Configured: <strong id=\"configured\">0</strong></div>
    </div>
  </header>

  <div id=\"message\" class=\"error\"></div>
  <div id=\"cams\"></div>

  <script>
    let cameras = [];
    let status = {};
    let view = {};
    let ws;
    const debounceTimers = {};
    let pingTimer = null;
    let snapshots = {};

    function statusText(label, value) {
      const el = document.getElementById(label);
      if (el) el.textContent = value;
    }

    function render() {
      statusText('mode', status.view_mode || 'inside');
      statusText('active', status.active_cameras || 0);
      statusText('configured', status.configured_cameras || 0);
      updateViewUI();
      const wrap = document.getElementById('cams');
      wrap.innerHTML = cameras.map(cam => {
        const res = cam.actual_resolution && cam.actual_resolution[0] ? `${cam.actual_resolution[0]}x${cam.actual_resolution[1]}` : (cam.resolution || []).join('x') || '—';
        const active = cam.active ? 'Live' : 'Inactive';
        const badgeColor = cam.active ? 'var(--accent)' : 'var(--accent-2)';
        const snap = snapshots[cam.index];
        return `
          <div class="card" data-cam="${cam.index}" data-fov="${cam.fov}">
            <h3>
              <span>${cam.name}</span>
              <span class="badge" style="border-color:${badgeColor}; color:${badgeColor}">${active}</span>
            </h3>
            <div class="muted">${cam.id} · ${cam.type} · ${res}</div>
            ${snap ? `<img class="snap" src="${snap}" alt="snapshot" />` : ''}
            <div class="row">
              <label>Enabled</label>
              <input class="toggle" type="checkbox" ${cam.enabled ? 'checked' : ''} onchange="applyCamera(${cam.index}, false)" />
            </div>
            <div class="grid">
              <div><label>Yaw</label><input class="cam-input" type="number" step="0.1" value="${cam.yaw}" /></div>
              <div><label>Pitch</label><input class="cam-input" type="number" step="0.1" value="${cam.pitch}" /></div>
              <div><label>Roll</label><input class="cam-input" type="number" step="0.1" value="${cam.roll}" /></div>
              <div><label>Orientation</label><input class="cam-input" type="number" step="0.1" value="${cam.orientation}" /></div>
              <div><label>FOV</label><input class="cam-input" type="number" step="0.5" value="${cam.fov}" /></div>
              <div><label>Mask Edge</label><input class="cam-input" type="number" step="0.05" min="0" max="1" value="${cam.mask_mindistance}" /></div>
            </div>
            <div class="actions">
              <button onclick="applyCamera(${cam.index}, false)">Save</button>
              <button class="secondary" onclick="applyCamera(${cam.index}, true)">Rebuild Map</button>
            </div>
          </div>`;
      }).join('');

      wireInputs();
    }

    function updateViewUI() {
      const yaw = document.getElementById('viewYaw');
      const pitch = document.getElementById('viewPitch');
      const roll = document.getElementById('viewRoll');
      const fov = document.getElementById('viewFov');
      if (yaw) yaw.value = view.yaw ?? 0;
      if (pitch) pitch.value = view.pitch ?? 0;
      if (roll) roll.value = view.roll ?? 0;
      if (fov) fov.value = view.fov ?? 70;
    }

    function collectPayload(card) {
      const nums = [...card.querySelectorAll('input[type=number]')].map(inp => parseFloat(inp.value));
      const [yaw, pitch, roll, orientation, fov, mask] = nums;
      return {
        enabled: card.querySelector('.toggle').checked,
        yaw, pitch, roll, orientation, fov, mask_mindistance: mask,
      };
    }

    function wireInputs() {
      const wrap = document.getElementById('cams');
      if (!wrap) return;

      wrap.querySelectorAll('.toggle').forEach(toggle => {
        toggle.onchange = () => {
          const card = toggle.closest('.card');
          if (!card) return;
          const idx = parseInt(card.dataset.cam, 10);
          scheduleApply(idx, false, 0);
        };
      });

      wrap.querySelectorAll('.cam-input').forEach(inp => {
        inp.oninput = () => {
          const card = inp.closest('.card');
          if (!card) return;
          const idx = parseInt(card.dataset.cam, 10);
          scheduleApply(idx, false, 200);
        };
      });
    }

    function scheduleApply(idx, forceRebuild, delayMs) {
      clearTimeout(debounceTimers[idx]);
      debounceTimers[idx] = setTimeout(() => applyCamera(idx, forceRebuild), delayMs);
    }

    async function applyCamera(idx, forceRebuild) {
      const card = document.querySelector(`[data-cam="${idx}"]`);
      if (!card) return;
      const payload = collectPayload(card);
      const prevFov = parseFloat(card.dataset.fov || payload.fov);
      payload.rebuild_lookup = forceRebuild || Math.abs(prevFov - payload.fov) > 1e-3;
      payload.save = true;
      try {
        const res = await fetch(`/api/cameras/${idx}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        if (!res.ok) {
          const detail = await res.json().catch(() => ({}));
          throw new Error(detail.detail || 'Update failed');
        }
        const updated = await res.json();
        cameras = cameras.map(c => c.index === idx ? updated : c);
        render();
      } catch (err) {
        const msg = document.getElementById('message');
        msg.textContent = err.message || err;
        setTimeout(() => msg.textContent = '', 3000);
      }
    }

    async function fetchState() {
      const [cRes, sRes, vRes] = await Promise.all([
        fetch('/api/cameras'),
        fetch('/api/status'),
        fetch('/api/view'),
      ]);
      const [cData, sData, vData] = await Promise.all([cRes.json(), sRes.json(), vRes.json()]);
      cameras = cData;
      status = sData;
      view = vData;
      render();
    }

    function wireViewInputs() {
      const fields = ['viewYaw','viewPitch','viewRoll','viewFov'];
      fields.forEach(id => {
        const el = document.getElementById(id);
        if (!el) return;
        el.oninput = () => scheduleViewApply(200);
      });
      const resetBtn = document.getElementById('resetView');
      if (resetBtn) resetBtn.onclick = () => applyView(true);
    }

    function collectViewPayload() {
      const yaw = parseFloat(document.getElementById('viewYaw')?.value || '0');
      const pitch = parseFloat(document.getElementById('viewPitch')?.value || '0');
      const roll = parseFloat(document.getElementById('viewRoll')?.value || '0');
      const fov = parseFloat(document.getElementById('viewFov')?.value || '70');
      return { yaw, pitch, roll, fov };
    }

    function scheduleViewApply(delayMs) {
      clearTimeout(debounceTimers['view']);
      debounceTimers['view'] = setTimeout(() => applyView(false), delayMs);
    }

    async function applyView(reset) {
      const payload = reset ? { reset: true } : collectViewPayload();
      try {
        const res = await fetch('/api/view', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        if (!res.ok) {
          const detail = await res.json().catch(() => ({}));
          throw new Error(detail.detail || 'Update failed');
        }
        view = await res.json();
        render();
      } catch (err) {
        const msg = document.getElementById('message');
        msg.textContent = err.message || err;
        setTimeout(() => msg.textContent = '', 3000);
      }
    }

    async function takeSnapshot() {
      try {
        const res = await fetch('/api/snapshot');
        if (!res.ok) throw new Error('Snapshot failed');
        const data = await res.json();
        snapshots = {};
        (data.snapshots || []).forEach(s => {
          snapshots[s.index] = s.image;
        });
        render();
      } catch (e) {
        const msg = document.getElementById('message');
        msg.textContent = e.message || e;
        setTimeout(() => msg.textContent = '', 3000);
      }
    }

    function connectSocket() {
      try {
        ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
        if (pingTimer) clearInterval(pingTimer);
        ws.onopen = () => {
          pingTimer = setInterval(() => {
            try { ws.send('ping'); } catch (e) { /* ignore */ }
          }, 15000);
        };
        ws.onmessage = evt => {
          const data = JSON.parse(evt.data || '{}');
          if (data.cameras) cameras = data.cameras;
          if (data.status) status = data.status;
          if (data.view) view = data.view;
          render();
        };
        ws.onclose = () => {
          if (pingTimer) clearInterval(pingTimer);
          setTimeout(connectSocket, 3000);
        };
      } catch (e) {
        console.warn('WS unavailable, falling back to polling', e);
        setInterval(fetchState, 2000);
      }
    }

    fetchState();
    connectSocket();
    document.getElementById('snapshotBtn').onclick = () => takeSnapshot();
    wireViewInputs();
  </script>
</body>
</html>
"""
