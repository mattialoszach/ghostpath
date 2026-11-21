"""Local FastAPI application and WebSocket workspace controller."""

import argparse
import asyncio
import json
import os
from contextlib import asynccontextmanager, suppress
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from serial.tools.list_ports import comports
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .cameras import jpeg
from .dataset import load_dataset_backend
from .models import Command, Phase
from .motion import JOINTS
from .preview import render_camera
from .session import Session, WorkflowError, workspace_path

LOCAL = ("127.0.0.1", "localhost")


@lru_cache(maxsize=4)
def demo_preview(index: int) -> bytes:
    pose = {joint: (65.0 if joint == "gripper" else 0.0) for joint in JOINTS}
    return jpeg(render_camera(pose, 640, 480, index))


def create_app(data: Path | None = None) -> FastAPI:
    session = Session(data or workspace_path())

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        session.start()
        try:
            yield
        finally:
            await asyncio.to_thread(session.close)

    app = FastAPI(title="GhostPath", lifespan=lifespan)
    app.state.session = session
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(LOCAL))
    owner: WebSocket | None = None

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/devices")
    def devices():
        ports = [
            {"port": port.device, "label": port.description}
            for port in sorted(comports())
            if port.vid is not None or "usb" in port.device.lower()
        ]
        cache = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache"))
        hf = Path(os.getenv("HF_HOME", cache / "huggingface"))
        lerobot = Path(os.getenv("HF_LEROBOT_HOME", hf / "lerobot"))
        calibration_root = (
            Path(os.getenv("HF_LEROBOT_CALIBRATION", lerobot / "calibration")).expanduser()
            / "robots"
        )
        calibrations = sorted(
            {
                file.stem
                for kind in ("so101_follower", "so_follower")
                for file in (calibration_root / kind).glob("*.json")
            }
        )
        return {"ports": ports, "calibrations": calibrations}

    @app.get("/api/preview/{name}")
    def preview(name: str, review: int | None = None):
        cameras, datasets = session.cameras, session.datasets
        if review is not None:
            frame = (
                datasets.review(name, review)
                if datasets and session.phase == Phase.REVIEW
                else None
            )
        elif cameras is not None:
            frame = cameras.preview(name)
        else:
            frame = (
                demo_preview(1 if name == "wrist" else 0)
                if name in {"workspace", "wrist"}
                else None
            )
        if not frame:
            raise HTTPException(404, "Camera frame unavailable")
        return Response(frame, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.websocket("/ws")
    async def websocket(socket: WebSocket):
        nonlocal owner
        origin = socket.headers.get("origin")
        if origin and (
            urlsplit(origin).scheme not in {"http", "https"}
            or urlsplit(origin).netloc != socket.headers.get("host")
        ):
            await socket.close(code=1008)
            return
        await socket.accept()
        if owner is not None:
            await socket.send_json(
                {"type": "error", "message": "Close the other GhostPath tab to take control."}
            )
            await socket.close(code=1008)
            return
        owner = socket
        session.heartbeat()
        outgoing: asyncio.Queue = asyncio.Queue(maxsize=32)
        command_tasks: set[asyncio.Task] = set()

        async def emit(message: dict):
            await outgoing.put(message)

        async def respond(command: Command, ident: int | str | None):
            try:
                await asyncio.wrap_future(session.submit(command))
            except WorkflowError as exc:
                await emit({"type": "error", "id": ident, "message": str(exc)})
            else:
                if command.action not in {"jog", "demo_move", "grip_adjust"}:
                    await emit({"type": "ack", "id": ident})

        async def receive():
            while True:
                payload = await socket.receive_text()
                if len(payload) > 8192:
                    await socket.close(code=1009)
                    return
                try:
                    raw = json.loads(payload)
                    if not isinstance(raw, dict):
                        raise ValueError("Expected a command object.")
                    ident = raw.pop("id", None)
                    command = Command.model_validate(raw)
                except (ValidationError, ValueError) as exc:
                    message = (
                        exc.errors()[0]["msg"] if isinstance(exc, ValidationError) else str(exc)
                    )
                    await emit({"type": "error", "message": message})
                    continue
                if command.action == "heartbeat":
                    session.heartbeat()
                    continue
                if len(command_tasks) >= 20 and command.action != "stop":
                    continue
                task = asyncio.create_task(respond(command, ident))
                command_tasks.add(task)
                task.add_done_callback(command_tasks.discard)

        async def send():
            while True:
                try:
                    message = await asyncio.wait_for(outgoing.get(), timeout=0.10)
                except TimeoutError:
                    message = {"type": "state", "state": session.state}
                await socket.send_json(message)

        tasks = [asyncio.create_task(receive()), asyncio.create_task(send())]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        except (WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
            pass
        finally:
            session.lost_controller()
            owner = None
            for task in (*tasks, *command_tasks):
                task.cancel()
            with suppress(asyncio.CancelledError):
                await asyncio.gather(*tasks, *command_tasks, return_exceptions=True)

    ui = Path("frontend/dist").resolve()
    if ui.is_dir():
        app.mount("/", StaticFiles(directory=ui, html=True), name="workspace")
    else:

        @app.get("/")
        async def build_ui():
            return {"message": "Build the workspace with npm ci and npm run build in frontend/."}

    return app


def main():
    parser = argparse.ArgumentParser(description="Serve the local GhostPath workspace")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data", type=Path, help="Directory for datasets, start poses and paths")
    options = parser.parse_args()
    with suppress(RuntimeError):
        load_dataset_backend()
    uvicorn.run(
        create_app(options.data.resolve() if options.data else None),
        host="127.0.0.1",
        port=options.port,
        access_log=False,
    )
