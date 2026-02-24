import asyncio
import os
import threading
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
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
    pano_yaw: Optional[float] = None
    pano_pitch: Optional[float] = None
    pano_zoom: Optional[float] = None
    reset: bool = False


class RecordUpdate(BaseModel):
    enabled: bool


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


class StreamManager(ConnectionManager):
    async def broadcast_bytes(self, payload: bytes) -> None:
        if not payload:
            return
        async with self._lock:
            targets = list(self.active)
        for ws in targets:
            try:
                await ws.send_bytes(payload)
            except Exception:
                try:
                    await ws.close()
                finally:
                    async with self._lock:
                        self.active.discard(ws)


class ControlServer:
    def __init__(self, renderer_app, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.renderer_app = renderer_app
        self.host = host
        self.port = port
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._server: Optional[uvicorn.Server] = None
        self.manager = ConnectionManager()
        self.stream_manager = StreamManager()
        self.ui_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "webui"))
        self.ui_index = os.path.join(self.ui_dir, "index.html")

        self.fastapi_app = FastAPI(title="Multicam Control", version="0.1.0")
        self.fastapi_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        if os.path.isdir(self.ui_dir):
            self.fastapi_app.mount("/static", StaticFiles(directory=self.ui_dir), name="static")
        self._setup_routes()

    def _setup_routes(self) -> None:
        @self.fastapi_app.get("/")
        async def root():
            if os.path.exists(self.ui_index):
                return FileResponse(self.ui_index)
            return {"message": "UI assets missing", "path": self.ui_index}

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

        @self.fastapi_app.post("/api/record")
        async def update_record(payload: RecordUpdate):
            try:
                result = self.renderer_app.run_on_main(
                    lambda: self.renderer_app.set_recording(payload.enabled)
                )
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

            self.queue_broadcast()
            return result

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

        @self.fastapi_app.websocket("/ws/stream")
        async def stream_endpoint(websocket: WebSocket):
            await self.stream_manager.connect(websocket)
            try:
                await websocket.send_json({"type": "stream-ready"})
                while True:
                    await websocket.receive_text()
            except WebSocketDisconnect:
                await self.stream_manager.disconnect(websocket)
            except Exception:
                await self.stream_manager.disconnect(websocket)

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

    def has_stream_clients(self) -> bool:
        return self.stream_manager.has_clients()

    def broadcast_frame(self, frame: bytes) -> None:
        if not frame:
            return
        if self._loop and self._loop.is_running() and self.stream_manager.has_clients():
            asyncio.run_coroutine_threadsafe(self.stream_manager.broadcast_bytes(frame), self._loop)
