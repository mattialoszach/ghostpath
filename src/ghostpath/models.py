from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_MAC_CAMERA_BUFFER_MS = 60
type SavedKind = Literal["rest", "recording"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Phase(StrEnum):
    SETUP = "setup"
    READY = "ready"
    PRACTICE = "practice"
    TEACHING = "teaching"
    TEACH_PAUSED = "teach_paused"
    RESET = "reset"
    RESET_RELEASED = "reset_released"
    RETURNING = "returning"
    REPLAY = "replay"
    MANUAL = "manual"
    PROCESSING = "processing"
    REVIEW = "review"
    SAVING = "saving"
    NEXT = "next"
    FINISHED = "finished"
    FAULT = "fault"


class Mode(StrEnum):
    TEACH = "teach"
    REPEAT = "repeat"
    HYBRID = "hybrid"


class ReplayProfile(StrEnum):
    STANDARD = "standard"
    FAST = "fast"


class JogSpeed(StrEnum):
    PRECISION = "precision"
    NORMAL = "normal"
    FAST = "fast"


class CameraSpec(StrictModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,23}$")
    source: int | str
    width: int = Field(default=640, ge=160, le=1920)
    height: int = Field(default=480, ge=120, le=1080)
    capture_fps: int = Field(default=30, ge=10, le=120)
    record: bool = True


class Configuration(StrictModel):
    device: Literal["simulation", "so101"] = "simulation"
    port: str = Field(default="", max_length=240)
    robot_id: str = Field(default="my_awesome_follower_arm", pattern=r"^[\w.-]{1,80}$")
    calibration_dir: Path | None = None
    task: str = Field(default="Pick up the orange block and place it in the tray", max_length=500)
    dataset: str = Field(default="block-placement", pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    fps: int = Field(default=20, ge=10, le=30)
    mac_camera_buffer_ms: Literal[75, 68, 60, 0] = DEFAULT_MAC_CAMERA_BUFFER_MS
    speed: float = Field(default=1.0, ge=0.25, le=2)
    replay_profile: ReplayProfile = ReplayProfile.STANDARD
    mode: Mode = Mode.TEACH
    automatic_reset: bool = False
    cameras: list[CameraSpec] = Field(
        default_factory=lambda: [
            CameraSpec(name="workspace", source=0),
            CameraSpec(name="wrist", source=1),
        ],
        max_length=4,
    )

    @model_validator(mode="after")
    def validate_setup(self):
        if not self.task.strip():
            raise ValueError("Enter a task description.")
        if self.robot_id in {".", ".."}:
            raise ValueError("Choose a calibration ID.")
        if self.device == "so101" and not self.port.strip():
            raise ValueError("Select the SO-101 serial port.")
        if len({c.name for c in self.cameras}) != len(self.cameras):
            raise ValueError("Camera names must be unique.")
        if len({str(c.source) for c in self.cameras}) != len(self.cameras):
            raise ValueError("Select a different source for each camera.")
        if any(c.record and c.capture_fps < self.fps for c in self.cameras):
            raise ValueError("Recorded cameras must capture at least as fast as the dataset FPS.")
        return self


class Command(StrictModel):
    action: Literal[
        "configure",
        "connect",
        "disconnect",
        "capture_start",
        "select_start",
        "save_start",
        "delete_start",
        "capture_rest",
        "clear_rest",
        "refresh_saved",
        "delete_saved",
        "park",
        "release_torque",
        "hold_torque",
        "teach",
        "continue_teach",
        "discard_teaching",
        "discard_episode",
        "stop_teach",
        "release_reset",
        "hold_reset",
        "auto_reset",
        "reverse_reset",
        "grip_adjust",
        "replay",
        "practice",
        "finish_practice",
        "take_over",
        "set_jog_speed",
        "jog",
        "finish_manual",
        "stop_recording",
        "save",
        "retry",
        "discard",
        "next",
        "finish",
        "stop",
        "recover",
        "heartbeat",
        "load_path",
        "demo_move",
        "demo_path",
    ]
    config: Configuration | None = None
    confirmed: bool = False
    allow_auto_reset_once: bool = False
    firm_carry_grip: bool | None = None
    grip_assist: bool = False
    grip_direction: Literal[-1, 0, 1] = 0
    jog_speed: JogSpeed | None = None
    axes: dict[str, float] = Field(default_factory=dict, max_length=6)
    immediate_stop: bool = False
    path_id: str = ""
    pose_id: str = ""
    pose_label: str = Field(default="", max_length=60)
    saved_kind: SavedKind | None = None
    saved_id: str = Field(default="", max_length=160)
    reuse: bool | None = None
    manual_finish: bool | None = None
