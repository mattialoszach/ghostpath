"""Single-owner motion loop and explicit workspace transitions."""

import json
import logging
import os
import re
import time
from concurrent.futures import Future
from dataclasses import replace
from datetime import UTC, datetime
from math import cos, pi, sin
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
from uuid import uuid4

from .cameras import CameraNotReady, Cameras, source_index
from .dataset import Datasets, Record
from .grip import CarryGripPath, GripAssist, GrippedPath
from .models import Command, Configuration, Mode, Phase, ReplayProfile
from .motion import JOINTS, Motion, ParkTrajectory, Pose, SafetyFault, Settings
from .poses import SavedStart, write_json
from .robot import Follower, Robot, SimulatedFollower
from .torque import TorqueManager
from .trajectory import (
    JOINT_MARGIN,
    LOAD_AXES,
    MAX_TEACH_SECONDS,
    START_TOLERANCE,
    AlignedPath,
    LoadTrim,
    NativePlayer,
    Player,
    ReplayPath,
    ReversePath,
    Sample,
    ServoPlayer,
    Teaching,
    automatic_pose,
    tolerance_for,
)
from .trajectory import (
    Path as MotionPath,
)

LOG = logging.getLogger(__name__)
CONTROL = Settings(
    speed=50,
    base_speed_scale=1,
    acceleration=170,
    gripper_speed=48,
    gripper_acceleration=170,
    park_speed=22,
    park_acceleration=75,
    tolerance=1.4,
    max_lead=6,
    gripper_max_lead=9,
)
FAST_REPLAY = replace(
    CONTROL, speed=120, acceleration=360, gripper_speed=100, gripper_acceleration=400
)
REPLAY_SETTINGS = {ReplayProfile.STANDARD: CONTROL, ReplayProfile.FAST: FAST_REPLAY}
ACTIVE = {Phase.TEACHING, Phase.RESET_RELEASED, Phase.RETURNING, Phase.REPLAY, Phase.MANUAL}
IDLE = {Phase.READY, Phase.TEACH_PAUSED, Phase.RESET, Phase.REVIEW, Phase.NEXT, Phase.FINISHED}
REST_PHASES = (Phase.READY, Phase.RESET, Phase.NEXT, Phase.FINISHED)


class WorkflowError(ValueError):
    """A rejected operator command that does not indicate a hardware fault."""


class Session:
    def __init__(self, data: Path):
        self.data = data
        self.config = Configuration(dataset=self._available_dataset("block-placement"))
        self.phase = Phase.SETUP
        self.robot: Robot | None = None
        self.cameras: Cameras | None = None
        self.datasets: Datasets | None = None
        self.motion: Motion | None = None
        self.start_pose: Pose | None = None
        self.start_pose_id: str | None = None
        self.saved_starts: dict[str, SavedStart] = {}
        self.rest_pose: Pose | None = None
        self.pose: Pose | None = None
        self.power: TorqueManager | None = None
        self.teacher: Teaching | None = None
        self.teaching_origin: Pose | None = None
        self.path: ReplayPath | GrippedPath | None = None
        self.path_id: str | None = None
        self.chart: list[list[float]] = []
        self.taught_grip: list[float] = []
        self.paths: list[dict] = []
        self.player: Player | None = None
        self.reset_action: str | None = None
        self.return_phase: Phase | None = None
        self.stopped_early = False
        self.scene_reversed = False
        self.firm_carry_grip = False
        self.replay_firm_carries: int | None = None
        self.grip_assist_enabled = False
        self.grip_assist: GripAssist | None = None
        self.quality: dict | None = None
        self.operation: tuple[str, Future] | None = None
        self.progress = self.elapsed = 0.0
        self.notice = ""
        self.fault = ""
        self.demo: float | None = None
        self.axes: Pose = {}
        self.jog_at = 0.0
        self.jog_active = False
        self.record_origin: float | None = None
        self.record_index = 0
        self.record_candidate: Record | None = None
        self.control_trace: list[Record] = []
        self.last_recorded_at = float("-inf")
        self.replay_path_started: float | None = None
        self.replay_timing: dict[str, float] | None = None
        self.manual_started = 0.0
        self.manual_finish = False
        self.manual_goal: Pose | None = None
        self.manual_trim: LoadTrim | None = None
        self.heartbeat_at = time.monotonic()
        self.last_tick = self.heartbeat_at
        self.pending: Queue[tuple[Command, Future]] = Queue(maxsize=24)
        self.stop_requested = Event()
        self.closed = Event()
        self.last_torque_check = 0.0
        self.last_goal_check = 0.0
        self.state = self._snapshot()
        self.thread = Thread(target=self._run, daemon=True, name="ghostpath-control")

    def start(self):
        self.thread.start()

    def close(self):
        self.closed.set()
        self.stop_requested.set()
        if self.thread.is_alive():
            self.thread.join()

    def heartbeat(self):
        self.heartbeat_at = time.monotonic()

    def lost_controller(self):
        # A reconnect cannot resume autonomous movement from a stale browser.
        self.heartbeat_at = float("-inf")
        if self.phase in ACTIVE or self.operation and self.operation[0] == "prepare":
            self.stop_requested.set()

    def submit(self, command: Command) -> Future:
        future = Future()
        if command.action == "stop":
            self.stop_requested.set()
            future.set_result(None)
        else:
            try:
                self.pending.put_nowait((command, future))
            except Full:
                future.set_exception(WorkflowError("Controls are busy. Try again."))
        return future

    def _run(self):
        period = 1 / CONTROL.fps
        deadline = time.monotonic()
        try:
            while not self.closed.is_set():
                try:
                    if self.stop_requested.is_set():
                        self.stop_requested.clear()
                        if (
                            self.phase in ACTIVE
                            or self.phase in {Phase.READY, Phase.RESET}
                            or (self.operation and self.operation[0] == "prepare")
                        ):
                            reason = (
                                "The browser control link was lost."
                                if time.monotonic() - self.heartbeat_at > 0.65
                                else "Stopped by the operator."
                            )
                            self._fail(reason, save_teaching=True)
                    self._check_operation()
                    self._commands()
                    self._tick(time.monotonic())
                except Exception as exc:
                    self._fail(str(exc) or type(exc).__name__, save_teaching=True)
                self.state = self._snapshot()
                deadline += period
                now = time.monotonic()
                if deadline < now:
                    deadline = now + period
                self.closed.wait(max(0.0, deadline - now))
        finally:
            if self.robot is not None and self.phase in ACTIVE:
                self._fail("Controller shut down.", save_teaching=True)
            self._close_devices()
            if self.datasets is not None:
                try:
                    self.datasets.close()
                except Exception:
                    LOG.exception("Dataset finalization failed on shutdown")

    def _commands(self):
        for _ in range(4):
            try:
                command, reply = self.pending.get_nowait()
            except Empty:
                return
            if not reply.set_running_or_notify_cancel():
                continue
            try:
                self._command(command)
            except WorkflowError as exc:
                reply.set_exception(exc)
            except Exception as exc:
                self._fail(str(exc) or type(exc).__name__, save_teaching=True)
                reply.set_exception(WorkflowError(self.fault))
            else:
                reply.set_result(None)

    def _command(self, cmd: Command):
        action = cmd.action
        if self.fault and self.phase != Phase.FAULT and action not in {
            "save",
            "discard",
            "disconnect",
        }:
            raise WorkflowError("Robot control is stopped. Save or discard the take first.")
        if action in {"configure", "connect"}:
            self._require(Phase.SETUP)
            if cmd.config:
                self.config = cmd.config
            if action == "connect":
                self._connect()
        elif action == "disconnect":
            self._require(
                Phase.READY,
                Phase.TEACH_PAUSED,
                Phase.RESET,
                Phase.NEXT,
                Phase.FINISHED,
                Phase.FAULT,
            )
            if (
                self.datasets
                and self.datasets.accepted
                and self.phase not in {Phase.FINISHED, Phase.FAULT}
            ):
                raise WorkflowError("Finish the dataset before disconnecting.")
            if self.operation:
                raise WorkflowError("Wait for the current dataset operation.")
            if self.datasets and not self.datasets.broken:
                self._begin_operation("disconnect", self.datasets.finalize())
            else:
                self._disconnect()
        elif action in {"capture_rest", "clear_rest"}:
            self._require(*REST_PHASES)
            if self.rest_pose is not None and not cmd.confirmed:
                raise WorkflowError("Confirm changing the saved rest pose.")
            file = self._pose_file(rest=True)
            if action == "capture_rest":
                self.pose = self.robot.read()
                position = automatic_pose(self.pose, self.motion, self._joint_margin())
            try:
                if action == "capture_rest":
                    write_json(file, {"version": 1, "position": position})
                    self.rest_pose = position
                    self.notice = "Rest pose saved."
                else:
                    file.unlink(missing_ok=True)
                    self.rest_pose = None
                    self.notice = "Rest pose cleared."
            except OSError as exc:
                raise WorkflowError(f"Could not change the saved rest pose: {exc}") from exc
        elif action in {"capture_start", "select_start"}:
            self._require(Phase.READY)
            self.pose = self.robot.read()
            if action == "capture_start":
                self.start_pose_id, self.start_pose = None, self.motion.clamp(self.pose)
            else:
                selected = self.saved_starts.get(cmd.pose_id)
                if selected is None:
                    raise WorkflowError("Choose a saved start pose for this arm.")
                self.start_pose_id, self.start_pose = selected.id, selected.position.copy()
            self.notice = (
                "Start selected. You can begin teaching."
                if self._near(self._reset_goal())
                else "Start selected. Return to it to begin teaching."
            )
        elif action == "save_start":
            self._require(Phase.READY)
            if self.start_pose is None:
                raise WorkflowError("Choose a start pose before saving it.")
            label = cmd.pose_label.strip()
            if not label:
                raise WorkflowError("Enter a name for the saved start pose.")
            self.start_pose_id = self._save_start(label)
            self.notice = "Start pose saved."
        elif action == "delete_start":
            self._require(Phase.READY)
            self._delete_start(cmd)
        elif action in {"release_torque", "hold_torque"}:
            self._require(*IDLE)
            self._confirmed(cmd, "Support the arm and gripper before changing torque.")
            self.power.refresh()
            if action == "release_torque":
                if self.power.state != "off":
                    self.power.release()
                self.notice = "Torque off. Support the arm while moving it by hand."
            else:
                if self.power.state != "on":
                    self.pose = self.power.engage(self._assert_controller)
                self.notice = "Torque on. The arm is holding its current position."
        elif action in {"teach", "demo_path", "continue_teach"}:
            self._begin_teaching(cmd)
        elif action == "stop_teach":
            self._require(Phase.TEACHING)
            self._stop_teaching()
        elif action == "discard_teaching":
            self._require(Phase.TEACH_PAUSED)
            self._discard_episode()
        elif action == "discard_episode":
            self._require(
                Phase.TEACHING,
                Phase.TEACH_PAUSED,
                Phase.RESET,
                Phase.RESET_RELEASED,
                Phase.RETURNING,
            )
            if not cmd.confirmed:
                raise WorkflowError("Confirm that you want to discard the current episode.")
            self._discard_episode()
        elif action == "load_path":
            self._require(Phase.READY, Phase.TEACH_PAUSED, Phase.RESET, Phase.NEXT)
            self._load_path(cmd.path_id)
        elif action == "release_reset":
            self._require(Phase.READY, Phase.TEACH_PAUSED, Phase.RESET)
            if self._reset_goal() is None:
                next_phase = Phase.READY
            elif self.phase == Phase.TEACH_PAUSED and self.path is None:
                raise WorkflowError("Continue teaching until the path is ready to reset.")
            else:
                next_phase = Phase.RESET_RELEASED
            self._confirmed(cmd, "Support the arm before releasing torque.")
            self.power.release()
            self._clear_teaching()
            self.scene_reversed = False
            self.phase = next_phase
            self.last_tick = time.monotonic()
            self.notice = (
                "Guide the arm to the start pose, then select Hold & verify."
                if next_phase == Phase.RESET_RELEASED
                else "Position the supported arm and use its current position as start."
            )
        elif action == "hold_reset":
            self._require(Phase.RESET_RELEASED)
            self._assert_controller()
            self.pose = self.power.engage(self._assert_controller)
            self.phase = Phase.RESET if self.path else Phase.READY
            self.notice = (
                "Start pose verified. Restore the scene before replaying."
                if self._near(self._reset_goal())
                else "Pose held. The arm is outside the start tolerance."
            )
        elif action in {"auto_reset", "reverse_reset", "park"}:
            self._auto_reset(cmd, reverse=action == "reverse_reset", park=action == "park")
        elif action == "grip_adjust":
            if self.grip_assist and self.phase == Phase.RETURNING:
                self.grip_assist.input(cmd.grip_direction, time.monotonic())
        elif action == "replay":
            self._require(Phase.TEACH_PAUSED, Phase.RESET)
            if not self.path:
                raise WorkflowError("Teach or load a path first.")
            self._confirmed(cmd, "Restore the scene and keep the full path and camera views clear.")
            self._check_start()
            self._verify_holding()
            try:
                self.cameras.check_recording(self.config.fps)
            except ValueError as exc:
                raise WorkflowError(str(exc)) from exc
            if self.datasets is None:
                self.datasets = Datasets(
                    self.data / "datasets" / self.config.dataset, self.config, self.cameras
                )
            self.manual_finish = (
                cmd.manual_finish
                if cmd.manual_finish is not None
                else self.config.mode == Mode.HYBRID
            )
            if cmd.firm_carry_grip is not None:
                self.firm_carry_grip = cmd.firm_carry_grip
            self._begin_operation("prepare", self.datasets.prepare())
            self.notice = (
                "Starting recording. The arm holds until the cameras and writer are ready."
            )
        elif action == "jog":
            self._require(Phase.MANUAL)
            if any(j not in JOINTS or abs(v) > 1 for j, v in cmd.axes.items()):
                raise WorkflowError("Invalid manual joint input.")
            self.axes = cmd.axes
            self.jog_at = time.monotonic()
        elif action in {"finish_manual", "stop_recording"}:
            phases = (Phase.MANUAL,) if action == "finish_manual" else (Phase.REPLAY, Phase.MANUAL)
            self._require(*phases)
            self.robot.hold()
            self.robot.stream_profile(CONTROL)
            if self.record_index < 3:
                self._begin_operation("stop_empty", self.datasets.discard())
                self.notice = "Recording stopped before three frames were captured."
            else:
                self._seal(stopped=self.phase == Phase.REPLAY)
        elif action == "save":
            self._require(Phase.REVIEW)
            self._begin_operation(
                "save",
                self.datasets.save(
                    self.path_id,
                    self.config.mode,
                    manual_finish=self.manual_finish,
                    firm_carry_grip=bool(self.replay_firm_carries),
                ),
            )
        elif action in {"retry", "discard"}:
            self._require(Phase.REVIEW)
            self._begin_operation(action, self.datasets.discard())
        elif action == "next":
            self._require(Phase.NEXT)
            self._next(cmd.reuse)
        elif action == "finish":
            self._require(Phase.NEXT, Phase.READY, Phase.TEACH_PAUSED, Phase.RESET, Phase.FAULT)
            if self.operation:
                raise WorkflowError("Wait for the current dataset operation.")
            if not self.datasets or not self.datasets.accepted:
                raise WorkflowError("Save at least one episode first.")
            self._begin_operation("finish", self.datasets.finalize())
        elif action == "recover":
            self._require(Phase.FAULT)
            if self.datasets and self.datasets.broken:
                raise WorkflowError("Disconnect and start a new session after the dataset error.")
            if self.operation:
                raise WorkflowError("Wait for the capture cleanup to finish.")
            if not self.robot:
                self._disconnect()
                return
            self.pose = self.robot.read()
            self.power.refresh()
            if self.power.state == "mixed":
                self._confirmed(cmd, "Support the arm until every joint is holding.")
                self.pose = self.power.engage(self._assert_controller)
            self.phase = Phase.RESET if self.path else Phase.READY
            self.fault = ""
            self.notice = "Recovered. Verify the start pose before moving."
        elif action == "demo_move":
            self._require(Phase.TEACHING, Phase.RESET_RELEASED, Phase.READY)
            if not isinstance(self.robot, SimulatedFollower):
                raise WorkflowError("Virtual joints are available in simulation.")
            pose = self.robot.read()
            for j, v in cmd.axes.items():
                if j not in JOINTS or abs(v) > 100:
                    raise WorkflowError("Invalid virtual joint position.")
                pose[j] = v
            self.robot.teach(pose)
        else:
            raise WorkflowError("Invalid workspace command.")

    def _require(self, *phases: Phase):
        if self.phase not in phases:
            raise WorkflowError("This action is unavailable at the current workflow step.")

    def _confirmed(self, cmd: Command, message: str):
        if self.config.device == "so101" and not cmd.confirmed:
            raise WorkflowError(message)
        if time.monotonic() - self.heartbeat_at > 0.65 or self.stop_requested.is_set():
            raise WorkflowError("The workspace controller is disconnected.")

    def _available_dataset(self, name: str) -> str:
        base = re.sub(r"-[0-9]{3,}$", "", name)[:58]
        candidate, index = name, 1
        while (self.data / "datasets" / candidate).exists():
            index += 1
            candidate = f"{base}-{index:03}"
        return candidate

    def _connect(self):
        if (self.data / "datasets" / self.config.dataset).exists():
            raise WorkflowError("This dataset name exists. Enter a fresh name in setup.")
        robot = None
        try:
            if self.config.device == "simulation":
                robot = SimulatedFollower(CONTROL)
            else:
                for spec in self.config.cameras:
                    source_index(spec.source)
                # Only an enumerated local serial device is eligible for control.
                from serial.tools.list_ports import comports

                if self.config.port not in {p.device for p in comports()}:
                    raise ValueError("Select a connected serial device and scan again.")
                robot = Follower(
                    self.config.port,
                    self.config.robot_id,
                    CONTROL,
                    self.config.calibration_dir.expanduser()
                    if self.config.calibration_dir
                    else None,
                )
            robot.connect()
            pose, power = robot.read(), TorqueManager(robot)
            motion = Motion(pose, robot.limits, CONTROL)
            if power.state == "mixed":
                raise ValueError("Motors report mixed torque. Support the arm and check power.")
        except Exception as exc:
            if robot is not None:
                robot.close()
            raise WorkflowError(f"Could not connect: {exc}") from exc
        self.robot, self.pose, self.power, self.motion = robot, pose, power, motion
        self.rest_pose = self._read_saved_pose(rest=True)
        self._refresh_paths()
        self.cameras = Cameras(self.config.cameras, self.config.device == "simulation")
        self.cameras.demo_pose = pose
        self.cameras.open()
        self.phase = Phase.READY
        self.notice = "Connected. Choose a start pose."

    def _read_saved_pose(self, *, rest: bool) -> Pose | None:
        try:
            saved = json.loads(self._pose_file(rest=rest).read_text())
            self.motion.check_pose(saved["position"])
            if saved["version"] == 1:
                return saved["position"]
        except (OSError, ValueError, KeyError, TypeError, SafetyFault):
            pass
        return None

    def _start_library(self) -> Path:
        return self.data / "start-poses" / self._pose_file().stem

    def _save_start(self, label: str) -> str:
        position = self.start_pose.copy()
        for saved in self.saved_starts.values():
            if not saved.legacy and saved.label == label and saved.position == position:
                return saved.id
        ident = uuid4().hex[:12]
        created = datetime.now(UTC).isoformat()
        try:
            write_json(
                self._start_library() / f"{ident}.json",
                {"version": 1, "label": label, "position": position, "created": created},
            )
        except OSError as exc:
            raise WorkflowError(f"Could not save the start pose: {exc}") from exc
        saved = SavedStart(ident, label, created, position)
        self.saved_starts = {ident: saved, **self.saved_starts}
        return ident

    def _delete_start(self, cmd: Command):
        ident = cmd.pose_id
        if not cmd.confirmed or ident not in self.saved_starts:
            raise WorkflowError("Confirm which saved start pose to delete.")
        file = (
            self._pose_file() if ident == "previous-start"
            else self._start_library() / f"{ident}.json"
        )
        try:
            file.unlink(missing_ok=True)
        except OSError as exc:
            raise WorkflowError(f"Could not delete the saved start pose: {exc}") from exc
        del self.saved_starts[ident]
        if self.start_pose_id == ident:
            self.start_pose_id = None
        self.notice = "Saved start deleted. The session's selected position is unchanged."

    def _saved_start_options(self) -> dict[str, SavedStart]:
        candidates = []
        for file in self._start_library().glob("*.json"):
            if not re.fullmatch(r"[a-f0-9]{12}", file.stem):
                continue
            try:
                data = json.loads(file.read_text())
                self.motion.check_pose(data["position"])
                if data["version"] == 1 and isinstance(data["created"], str):
                    label = data.get("label")
                    named = isinstance(label, str) and bool(label.strip())
                    candidates.append(
                        SavedStart(
                            file.stem, label if named else "Earlier start",
                            data["created"], data["position"], legacy=not named,
                        )
                    )
            except (OSError, ValueError, TypeError, KeyError, SafetyFault):
                continue
        candidates.sort(key=lambda saved: saved.created, reverse=True)
        position = self._read_saved_pose(rest=False)
        if position:
            try:
                created = datetime.fromtimestamp(self._pose_file().stat().st_mtime, UTC).isoformat()
            except OSError:
                pass
            else:
                candidates.append(
                    SavedStart("previous-start", "Previous start", created, position, legacy=True)
                )
        return {saved.id: saved for saved in candidates}

    def _close_devices(self):
        if self.robot is not None:
            try:
                self.robot.close()
            except Exception:
                LOG.exception("Could not close the robot link")
            self.robot = None
        if self.cameras is not None:
            self.cameras.close()
            self.cameras = None

    def _disconnect(self):
        self._close_devices()
        if self.datasets is not None:
            self.datasets.executor.shutdown(wait=True)
            self.datasets = None
        self.phase = Phase.SETUP
        self.config = self.config.model_copy(
            update={"dataset": self._available_dataset(self.config.dataset)}
        )
        self.pose = self.start_pose = self.path = self.path_id = self.motion = None
        self.start_pose_id = None
        self.saved_starts = {}
        self.rest_pose = None
        self._clear_teaching()
        self.reset_action = None
        self.return_phase = None
        self.firm_carry_grip = False
        self.grip_assist_enabled = False
        self.grip_assist = None
        self.stopped_early = False
        self.scene_reversed = False
        self.power = None
        self.paths, self.chart = [], []
        self.taught_grip = []
        self.quality, self.fault = None, ""
        self.progress = self.elapsed = 0.0
        self.notice = "Disconnected."

    def _pose_file(self, *, rest: bool = False) -> Path:
        suffix = "-rest" if rest else ""
        return (
            self.data
            / "poses"
            / f"{self.robot.robot_id}-{self.robot.calibration_hash}{suffix}.json"
        )

    def _reset_goal(self) -> Pose | None:
        if self.path:
            return self.path.start
        if self.start_pose:
            return automatic_pose(self.start_pose, self.motion, self._joint_margin())
        return None

    def _joint_margin(self) -> float:
        return JOINT_MARGIN if self.config.device == "so101" else 0.0

    def _park_goal(self) -> Pose | None:
        return (
            automatic_pose(self.rest_pose, self.motion, self._joint_margin())
            if self.rest_pose is not None else None
        )

    def _near(self, goal: Pose | None, pose: Pose | None = None) -> bool:
        pose = self.pose if pose is None else pose
        return bool(
            goal and pose and all(abs(pose[j] - goal[j]) <= tolerance_for(j) for j in JOINTS)
        )

    def _check_start(self):
        target = self._reset_goal()
        if target is None:
            raise WorkflowError("Choose a start pose first.")
        self.pose = self.robot.read()
        if not self._near(target):
            raise WorkflowError("The arm is outside start tolerance. Reset and verify the pose.")

    def _verify_holding(self):
        if self.config.device == "so101":
            self.power.refresh()
            if self.power.state != "on":
                raise WorkflowError(
                    "Turn on torque and let the arm hold before starting powered motion."
                )

    def _powered_start(self) -> Pose:
        self._assert_controller()
        self._verify_holding()
        if self.config.device == "simulation" and self.power.state != "on":
            self.power.engage(self._assert_controller)
        self.pose, latched = self.robot.motion_start()
        self._assert_controller()
        return latched

    def _auto_reset(self, cmd: Command, *, reverse: bool = False, park: bool = False):
        phases = REST_PHASES if park else (Phase.READY, Phase.TEACH_PAUSED, Phase.RESET)
        self._require(*phases)
        goal = self._park_goal() if park else self._reset_goal()
        if goal is None:
            raise WorkflowError("Save a rest pose first." if park else "Choose a start pose first.")
        if self.config.device == "so101" and not (
            self.config.automatic_reset or cmd.allow_auto_reset_once
        ):
            raise WorkflowError("Opt in to automatic motion before returning.")
        self._verify_holding()
        if reverse:
            self._require(Phase.TEACH_PAUSED, Phase.RESET)
            if self.path is None:
                raise WorkflowError("Teach or load a path before starting Boomerang.")
            self._confirmed(
                cmd, "Confirm the object is at its final placement and both routes are clear."
            )
            self.pose = self.robot.read()
            if not any(
                self._near(pose) for pose in (self.path.end, self._reset_goal(), self._park_goal())
            ):
                raise WorkflowError("Start Boomerang from the start pose or taught endpoint.")
        else:
            self._confirmed(cmd, "Empty the gripper and clear the entire return path.")
        self.motion.check_pose(goal)
        latched = self._powered_start()
        self.scene_reversed = False
        if self.phase == Phase.TEACH_PAUSED and self.path is None:
            raise WorkflowError("Continue teaching until the path is ready to reset.")
        self.return_phase = Phase.RESET if self.phase == Phase.TEACH_PAUSED else self.phase
        if reverse:
            at_endpoint = self._near(self.path.end)
            if not at_endpoint and not (
                self._near(self._reset_goal()) or self._near(self._park_goal())
            ):
                raise WorkflowError("The arm moved away from the start. Verify its position.")
            if cmd.firm_carry_grip is not None:
                self.firm_carry_grip = cmd.firm_carry_grip
            self.grip_assist_enabled = cmd.grip_assist
            if at_endpoint:
                self._start_reverse(latched)
                self._clear_teaching()
                return
            self.reset_action = "reverse_approach"
            self._start_player(ParkTrajectory(latched, self.path.end, CONTROL), Phase.RETURNING)
            self.notice = "Moving to the taught endpoint before Boomerang."
        else:
            self.reset_action = "park" if park else "direct"
            self._start_player(ParkTrajectory(latched, goal, CONTROL), Phase.RETURNING)
            self.notice = "Returning to rest." if park else "Returning to the task start."
        self._clear_teaching()

    def _start_reverse(self, latched: Pose):
        self.reset_action = "reverse"
        origin = self._servo_origin(latched)
        carry_grip = CarryGripPath(self.path, reverse=True) if self.firm_carry_grip else None
        reverse = carry_grip or ReversePath(self.path, time_scale=1.0)
        path = AlignedPath(origin, reverse, CONTROL)
        if self.grip_assist_enabled:
            self.grip_assist = GripAssist(self.path, path, path.prefix, carry_grip)
        self._start_player(
            self.grip_assist or path, Phase.RETURNING, held=latched, settings=self.path.settings
        )
        self.notice = "Verify the scene before recording."
        if carry_grip is not None:
            count = len(carry_grip.grasps)
            segment = "likely carry" if count == 1 else "likely carries"
            message = (
                f"Firm grip is on for {count} {segment}."
                if count
                else "No carry qualified for extra grip."
            )
            self.notice = f"{message} {self.notice}"

    def _path_file(self, ident: str) -> Path:
        return self.data / "paths" / f"{ident}.json"

    def _save_raw(self, samples: list[Sample], complete: bool) -> str:
        ident = f"{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid4().hex[:8]}"
        t0 = samples[0].t
        write_json(
            self._path_file(ident),
            {
                "version": 1,
                "robot_id": self.robot.robot_id,
                "calibration": self.robot.calibration_hash,
                "units": "degrees; gripper=percent",
                "start_pose": self.teaching_origin,
                "exact_start": True,
                "task": self.config.task,
                "created": datetime.now(UTC).isoformat(),
                "complete": complete,
                "samples": [{"t": sample.t - t0, "pose": sample.pose} for sample in samples],
            },
        )
        return ident

    def _refresh_paths(self):
        paths = []
        if self.robot:
            for file in sorted((self.data / "paths").glob("*.json"), reverse=True)[:200]:
                try:
                    item = json.loads(file.read_text())
                    if item["complete"] and self._same_robot(item):
                        paths.append(
                            {
                                "id": file.stem,
                                "task": item["task"],
                                "created": item["created"],
                                "duration": item["samples"][-1]["t"],
                                "grip_edited": "grip_edits" in item,
                            }
                        )
                except (OSError, ValueError, TypeError, KeyError, IndexError, SafetyFault):
                    continue
        self.paths = paths
        self.saved_starts = self._saved_start_options()

    def _same_robot(self, item: dict):
        return (
            item["version"] == 1
            and item["robot_id"] == self.robot.robot_id
            and item["calibration"] == self.robot.calibration_hash
        )

    def _load_path(self, ident: str):
        if not re.fullmatch(r"\d{8}T\d{6}-[a-f0-9]{8}", ident):
            raise WorkflowError("Select a saved trajectory.")
        try:
            item = json.loads(self._path_file(ident).read_text())
            if not item["complete"] or not self._same_robot(item):
                raise ValueError("The trajectory belongs to a different calibration.")
            origin = item["start_pose"]
            self.motion.check_pose(origin)
            samples = [Sample(**sample) for sample in item["samples"]]
            edited = "grip_edits" in item
            # Grip offsets are indexed by fitted time and must use their original profile.
            profile = ReplayProfile(item.get("grip_profile", "standard")) if edited else None
            base = self._compile_path(
                samples,
                origin,
                exact_start=item.get("exact_start") is True,
                legacy_timing=edited and item.get("grip_timing", 1) == 1,
                profile=profile,
            )
            path = (
                GrippedPath(base, [tuple(point) for point in item["grip_edits"]], base.settings)
                if edited
                else base
            )
            self._set_path(path, ident)
            self._clear_teaching()
        except (OSError, ValueError, TypeError, KeyError, SafetyFault) as exc:
            raise WorkflowError(f"Could not load path: {exc}") from exc

    def _compile_path(
        self,
        samples: list[Sample],
        origin: Pose,
        *,
        exact_start: bool = False,
        legacy_timing: bool = False,
        profile: ReplayProfile | None = None,
    ) -> ReplayPath:
        if samples and not self._near(origin, samples[0].pose):
            raise ValueError("The path begins outside its saved start tolerance.")
        path = ReplayPath(
            samples,
            self.motion,
            REPLAY_SETTINGS[profile or self.config.replay_profile],
            self.config.speed,
            joint_margin=self._joint_margin(),
            start_pose=origin if exact_start else None,
            legacy_timing=legacy_timing,
        )
        if not self._near(origin, path.start):
            raise ValueError("The path begins outside its saved start tolerance.")
        return path

    def _update_path(self, path: ReplayPath | GrippedPath, ident: str):
        self.path, self.path_id = path, ident
        self.chart = [list(path.at(path.duration * i / 89).values()) for i in range(90)]
        self.taught_grip = (
            [path.base.at(path.base.duration * i / 89)["gripper"] for i in range(90)]
            if isinstance(path, GrippedPath)
            else [row[-1] for row in self.chart]
        )

    def _set_path(self, path: ReplayPath | GrippedPath, ident: str):
        self._update_path(path, ident)
        self.firm_carry_grip = False
        self.reset_action = None
        self.scene_reversed = False
        self.phase = Phase.RESET
        self.progress = self.elapsed = 0.0
        self.notice = "Path ready. Reset the arm and restore the objects."

    def _finish_grip_assist(self):
        assist, self.grip_assist = self.grip_assist, None
        self.grip_assist_enabled = False
        if assist is None:
            return
        try:
            edited = assist.result(self.path.settings)
            if edited is None:
                return
            source = json.loads(self._path_file(self.path_id).read_text())
            ident = f"{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid4().hex[:8]}"
            source.update(
                grip_edits=edited.knots,
                grip_timing=1 if edited.base.legacy_timing else 2,
                grip_profile=(
                    ReplayProfile.FAST if edited.settings == FAST_REPLAY else ReplayProfile.STANDARD
                ),
                source_path=self.path_id,
                created=datetime.now(UTC).isoformat(),
            )
            write_json(self._path_file(ident), source)
        except (OSError, ValueError, TypeError) as exc:
            LOG.warning("Could not save grip edits: %s", exc)
            self.notice = "Boomerang finished. Grip edits could not be saved. Verify the scene."
        else:
            self._update_path(edited, ident)
            self._refresh_paths()
            self.notice = "Grip edits saved for the next replay. Verify the scene before recording."

    def _begin_teaching(self, cmd: Command):
        continuing = cmd.action == "continue_teach"
        self._require(Phase.TEACH_PAUSED if continuing else Phase.READY)
        if cmd.action == "demo_path" and not isinstance(self.robot, SimulatedFollower):
            raise WorkflowError("The example path is available in simulation.")
        self.power.refresh()
        if self.power.state == "off":
            self._assert_controller()
        else:
            self._confirmed(cmd, "Support the arm and gripper before releasing torque.")
        anchor = self.teacher.samples[-1].pose if continuing else self.start_pose
        if anchor is None:
            raise WorkflowError("Choose a start pose before teaching.")
        self.pose = self.robot.read()
        if not self._near(anchor):
            raise WorkflowError(
                "Return to the paused endpoint to continue teaching."
                if continuing
                else "Return to the selected start pose to begin teaching."
            )
        if continuing and self.teacher.samples[-1].t >= MAX_TEACH_SECONDS - 0.15:
            raise WorkflowError("The taught path has reached the recording time limit.")
        if not continuing:
            self.teacher = Teaching()
            self.teaching_origin = anchor.copy()
        if self.power.state != "off":
            self.power.release()
        self.pose = self.robot.read()
        if not self._near(anchor):
            raise SafetyFault("The arm moved while torque was releasing. Keep it supported.")
        now = time.monotonic()
        # Keep a short bridge for any measured change as the motors release.
        # The wall-clock time spent considering a paused take never enters the path.
        self.teacher.resume(now, self.pose, transition=0.08)
        self.path_id, self.path, self.chart = None, None, []
        self.phase = Phase.TEACHING
        self.demo = now if cmd.action == "demo_path" else None
        self.progress, self.elapsed = 0, self.teacher.samples[-1].t
        self.last_tick = now
        self.notice = "Torque off. Guide the arm and gripper through the task."

    def _stop_teaching(self):
        self.pose = self.robot.read()
        self.teacher.append(time.monotonic(), self.pose)
        self.pose = self.power.engage(self._assert_controller)
        self.demo = None
        samples = self.teacher.samples
        self.phase = Phase.TEACH_PAUSED
        self.elapsed = samples[-1].t
        try:
            path = self._compile_path(samples, self.teaching_origin, exact_start=True)
            ident = self._save_raw(samples, complete=True)
        except (OSError, ValueError, SafetyFault) as exc:
            self.notice = str(exc)
            return
        self._set_path(path, ident)
        self.elapsed = samples[-1].t
        self.phase = Phase.TEACH_PAUSED
        self.notice = "Teaching paused. Continue from this pose or use Boomerang."
        self._refresh_paths()

    def _can_continue(self) -> bool:
        return bool(
            self.phase == Phase.TEACH_PAUSED
            and self.teacher
            and self.teacher.samples
            and self.teacher.samples[-1].t < MAX_TEACH_SECONDS - 0.15
            and self._near(self.teacher.samples[-1].pose)
        )

    def _clear_teaching(self):
        self.teacher = None
        self.teaching_origin = None

    def _discard_episode(self):
        if self.teacher is None and self.path is None:
            raise WorkflowError("There is no episode in progress.")
        if self.phase == Phase.RETURNING:
            self.power.stop()
        self._clear_teaching()
        self.demo = self.player = self.path = self.path_id = None
        self.chart, self.taught_grip = [], []
        self.return_phase = self.reset_action = None
        self.grip_assist = None
        self.grip_assist_enabled = self.scene_reversed = self.firm_carry_grip = False
        self.replay_firm_carries = None
        self.progress = self.elapsed = 0.0
        self.quality = None
        self.phase = Phase.READY
        self.notice = "Episode discarded. Return to your start pose to teach again."

    def _next(self, reuse: bool | None):
        self.quality = None
        self.elapsed = self.progress = 0.0
        if self.path and (reuse if reuse is not None else self.config.mode == Mode.REPEAT):
            self.phase = Phase.RESET
            self.notice = "Path reused. Restore the scene for the next take."
        else:
            self.path = self.path_id = None
            self.chart = []
            self.taught_grip = []
            self.phase = Phase.READY
            self.notice = "Return to the selected start pose before teaching again."

    def _servo_origin(self, latched: Pose) -> Pose:
        if isinstance(self.robot, Follower):
            actual = self.motion.clamp(self.pose)
            return {**latched, **{j: actual[j] for j in LOAD_AXES}}
        return latched

    def _start_player(
        self,
        path: MotionPath,
        phase: Phase,
        *,
        held: Pose | None = None,
        settings: Settings = CONTROL,
    ):
        motion = Motion(self.pose, self.robot.limits, settings)
        if isinstance(self.robot, Follower) and isinstance(path, ParkTrajectory):
            self.robot.native_profile(path)
            self._assert_controller()
            self.last_tick = time.monotonic()
            self.player = NativePlayer(path, motion, self.last_tick, START_TOLERANCE)
        else:
            self.robot.stream_profile(settings)
            self._assert_controller()
            self.last_tick = time.monotonic()
            motion.target = path.start
            if isinstance(self.robot, Follower):
                self.player = ServoPlayer(
                    path,
                    motion,
                    self.last_tick,
                    START_TOLERANCE,
                    held=held or path.start,
                    adjust_grip=isinstance(path, GripAssist),
                )
            else:
                self.player = Player(path, motion, self.last_tick, START_TOLERANCE)
        self.last_goal_check = self.last_tick
        self.phase = phase
        self.elapsed = self.progress = 0.0

    def _begin_operation(self, name: str, future: Future):
        self.operation = (name, future)
        self.phase = Phase.SAVING if name in {"save", "finish", "disconnect"} else Phase.PROCESSING

    def _check_operation(self):
        if not self.operation or not self.operation[1].done():
            return
        name, future = self.operation
        self.operation = None
        try:
            result = future.result()
            if name == "prepare" and self.phase != Phase.FAULT:
                self.cameras.check_recording(self.config.fps)
        except CameraNotReady as exc:
            self.phase = Phase.RESET
            self.notice = f"Recording could not start. {exc}"
            return
        except Exception as exc:
            self._fail(str(exc) or type(exc).__name__)
            return
        if self.phase == Phase.FAULT:
            return
        if name == "prepare":
            self._assert_controller()
            self._check_start()
            latched = self._powered_start()
            origin = self._servo_origin(latched)
            if not self._near(self.path.start) or not self._near(self.path.start, origin):
                raise SafetyFault(
                    "The measured pose or holding goal moved away from the replay start"
                )
            self._assert_controller()
            self.record_origin = None
            self.record_index = 0
            self.record_candidate = None
            self.control_trace = []
            self.last_recorded_at = float("-inf")
            self.stopped_early = False
            self.scene_reversed = False
            self.reset_action = None
            carry_grip = CarryGripPath(self.path, reverse=False) if self.firm_carry_grip else None
            self.replay_firm_carries = len(carry_grip.grasps) if carry_grip else None
            path = AlignedPath(origin, carry_grip or self.path, CONTROL)
            self._start_player(path, Phase.REPLAY, held=latched, settings=self.path.settings)
            self.replay_path_started = None if path.prefix else self.player.started
            self.replay_timing = None
            self._clear_teaching()
            self.notice = "Replaying the taught path and recording synchronized data."
        elif name == "seal":
            self.quality = result
            self.phase = Phase.REVIEW
            self.notice = (
                "Recording stopped early. Review the captured portion."
                if self.stopped_early
                else "Capture passed validation. Review and save the episode."
            )
        elif name == "save":
            self.phase = Phase.FAULT if self.fault else Phase.NEXT
            self.notice = "Episode saved."
        elif name in {"retry", "discard"}:
            self.quality = None
            self.progress = self.elapsed = 0.0
            self.phase = Phase.RESET if name == "retry" else Phase.NEXT
            self.notice = "Take discarded. Restore the scene to try again."
            if self.fault:
                self.phase = Phase.FAULT
                self.notice = "Take discarded."
        elif name == "stop_empty":
            self.quality = None
            self.progress = self.elapsed = 0.0
            self.phase = Phase.RESET
            self.notice = "Recording stopped before three frames were captured."
        elif name == "finish":
            self.phase = Phase.FINISHED
            self.notice = "LeRobot dataset finalized."
        elif name == "disconnect":
            self._disconnect()

    def _assert_controller(self):
        if self.stop_requested.is_set() or self.closed.is_set():
            raise SafetyFault("Stopped by the operator.")
        if time.monotonic() - self.heartbeat_at > 0.65:
            raise SafetyFault("The browser control link was lost.")

    def _tick(self, now: float):
        dt, self.last_tick = now - self.last_tick, now
        if self.robot is None or self.fault or self.phase == Phase.SETUP:
            return
        if self.phase in {Phase.REPLAY, Phase.MANUAL} and self.datasets.error:
            raise SafetyFault(self.datasets.error)
        active = self.phase in ACTIVE
        if active:
            self._assert_controller()
            if dt > CONTROL.watchdog:
                raise SafetyFault("The motion loop missed its deadline.")
        # Only drive active states at control rate. Idle telemetry requires less serial traffic.
        if not active and now - getattr(self, "last_idle_read", 0.0) < 0.10:
            return
        self.last_idle_read = now
        if self.phase == Phase.TEACHING and self.demo is not None:
            self._demo_pose(now)
        read_started = time.monotonic()
        self.pose = self.robot.read()
        read_ended = time.monotonic()
        encoder_read = (read_started, read_ended)
        stamp = (read_started + read_ended) / 2
        if self.cameras:
            self.cameras.demo_pose = self.pose
        if now - self.last_torque_check > 0.27:
            self.last_torque_check = now
            self.power.refresh(active=active)
        if self.phase == Phase.TEACHING:
            self.teacher.append(stamp, self.pose)
            self.elapsed = self.teacher.samples[-1].t
            if self.elapsed >= MAX_TEACH_SECONDS or (
                self.demo is not None and now - self.demo >= 4.35
            ):
                self._stop_teaching()
        elif self.phase in {Phase.REPLAY, Phase.RETURNING}:
            if now - self.last_goal_check > 1:
                self.robot.verify_goals()
                self.last_goal_check = now
            assist = self.grip_assist
            if assist:
                assist.advance(stamp, dt, self.player.cursor)
            target, done = self.player.step(stamp, self.pose)
            self._assert_controller()
            send_started = time.monotonic()
            sent = self.robot.send(target)
            command_send = (send_started, time.monotonic())
            if assist:
                cursor = (
                    self.player.goal_cursor
                    if isinstance(self.player, ServoPlayer)
                    else self.player.cursor
                )
                assist.capture(cursor, target["gripper"])
            self.elapsed = max(0, stamp - self.player.started)
            self.progress = min(1.0, self.player.cursor / self.player.path.duration)
            if self.phase == Phase.REPLAY:
                on_path = self.player.cursor >= self.player.path.prefix
                if self.replay_path_started is None and on_path:
                    self.replay_path_started = stamp
                self.progress = max(
                    0.0,
                    min(1.0, (self.player.cursor - self.player.path.prefix) / self.path.duration),
                )
                stage = "path" if on_path else "aligning"
                if self.player.ended is not None:
                    stage = "settling"
                self._record(
                    stamp, sent, reference=self.player.motion.target,
                    encoder_read=encoder_read, command_send=command_send, stage=stage,
                    path_cursor=self.player.cursor, replay_rate=self.player.rate,
                )
                if done:
                    path_start, settle_start = self.replay_path_started, self.player.ended
                    if path_start is not None and settle_start is not None:
                        self.replay_timing = {
                            "alignment_seconds": round(path_start - self.player.started, 3),
                            "motion_seconds": round(settle_start - path_start, 3),
                            "settle_seconds": round(stamp - settle_start, 3),
                        }
                    if self.manual_finish:
                        self._start_manual(stamp, sent)
                    else:
                        self._seal()
            elif done and self.reset_action == "reverse_approach":
                self._start_reverse(sent)
            elif done:
                self.phase = self.return_phase or (Phase.RESET if self.path else Phase.READY)
                self.return_phase = None
                self.scene_reversed = self.reset_action == "reverse"
                if self.reset_action == "park":
                    self.notice = "At rest. Boomerang can reset a reversible scene from here."
                elif self.scene_reversed:
                    self.robot.stream_profile(CONTROL)
                    self.notice = "Reverse motion complete. Verify the scene before recording."
                    self._finish_grip_assist()
                else:
                    self.notice = (
                        "Start pose verified. Restore the objects before replaying."
                        if self.path
                        else "Start pose verified. Ready to teach."
                    )
        elif self.phase == Phase.MANUAL:
            self._manual_tick(now, dt, stamp, encoder_read)

    def _start_manual(self, stamp: float, sent: Pose):
        self.robot.stream_profile(CONTROL)
        self._assert_controller()
        self.phase = Phase.MANUAL
        self.manual_started = stamp
        nominal = self.motion.hold(self._servo_origin(sent))
        self.manual_trim = LoadTrim(nominal, sent) if isinstance(self.robot, Follower) else None
        self.manual_goal = sent
        self.axes, self.jog_active = {}, False
        self.jog_at = 0.0
        self.notice = "Ghost path complete. Keyboard control is ready. Recording continues."

    def _manual_tick(self, now: float, dt: float, stamp: float, encoder_read: tuple[float, float]):
        self.motion.check_tracking(self.pose)
        enabled = any(self.axes.values()) and now - self.jog_at < 0.25
        self._assert_controller()
        command_send = None
        reference = None
        if enabled or self.jog_active:
            target = (
                self.motion.step(self.axes, max(0.0001, dt), self.pose, precision=True)
                if enabled
                else self.motion.hold(self.pose)
            )
            reference = target
            if self.manual_trim:
                if enabled:
                    self.manual_trim.update(min(dt, 1 / CONTROL.fps), target, self.pose)
                target = self.manual_trim.goal(target, self.motion)
            send_started = time.monotonic()
            self.manual_goal = self.robot.send(target)
            command_send = (send_started, time.monotonic())
        self.jog_active = enabled
        self._record(
            stamp, self.manual_goal, manual=True, reference=reference,
            encoder_read=encoder_read, command_send=command_send, stage="manual",
        )
        self.elapsed = stamp - self.record_origin
        if stamp - self.manual_started > 180:
            self._seal()

    def _demo_pose(self, now: float):
        t = min(1.0, (now - self.demo) / 4.2)
        pose = self.teaching_origin.copy()
        pose["shoulder_pan"] += 22 * (1 - cos(pi * t)) / 2
        pose["shoulder_lift"] -= 19 * sin(pi * t)
        pose["elbow_flex"] += 23 * sin(pi * t)
        pose["wrist_flex"] += 10 * sin(2 * pi * t)
        pose["gripper"] -= 47 * sin(pi * t) ** 4
        self.robot.teach(self.motion.clamp(pose))

    def _record(
        self, stamp: float, sent: Pose, manual: bool = False, *, reference: Pose | None = None,
        encoder_read: tuple[float, float] | None = None,
        command_send: tuple[float, float] | None = None, stage: str = "",
        path_cursor: float | None = None, replay_rate: float | None = None,
    ):
        if self.datasets.error:
            raise SafetyFault(self.datasets.error)
        if self.record_origin is None:
            self.record_origin = stamp
        due = self.record_origin + self.record_index / self.config.fps
        prior = self.record_candidate
        current = Record(
            stamp, self.pose.copy(), sent.copy(), manual,
            reference.copy() if reference is not None else None,
            encoder_read, command_send, stage, path_cursor, replay_rate,
        )
        self.control_trace.append(current)
        self.record_candidate = current
        if stamp < due:
            return
        # Choose the nearer measured tick on either side of the sample time.
        # Keep each command and encoder reading together and in original order.
        if (
            prior is not None
            and prior.timestamp > self.last_recorded_at
            and prior.manual == manual
            and due - prior.timestamp < stamp - due
        ):
            current = prior
        self.datasets.offer(current)
        self.last_recorded_at = current.timestamp
        self.record_index += 1

    def _seal(self, *, stopped: bool = False):
        self.stopped_early = stopped
        trace, self.control_trace = self.control_trace, []
        self._begin_operation(
            "seal", self.datasets.seal(
                stopped_early=stopped, replay_timing=self.replay_timing,
                control_trace=trace, control_fps=CONTROL.fps,
            )
        )
        self.robot.stream_profile(CONTROL)

    def _fail(self, message: str, save_teaching: bool = False):
        if self.phase == Phase.FAULT:
            return
        LOG.warning("GhostPath stopped during %s: %s", self.phase, message)
        teaching_was_active = self.phase == Phase.TEACHING
        operation = self.operation[0] if self.operation else None
        # Keep a completed capture available while the arm stays faulted. A
        # writer failure clears its operation before entering this handler.
        keep_take = self.phase == Phase.REVIEW or operation in {"seal", "save", "retry", "discard"}
        if not keep_take:
            self.phase = Phase.FAULT
        self.fault = message
        self.demo = None
        self.firm_carry_grip = False
        self.grip_assist_enabled = False
        self.grip_assist = None
        if self.robot:
            try:
                self.power.stop()
            except Exception:
                self.fault += " Torque/hold could not be verified. Support the arm."
        if self.teacher:
            if save_teaching and teaching_was_active and self.teacher.samples:
                try:
                    self._save_raw(self.teacher.samples, complete=False)
                except Exception:
                    LOG.exception("Could not save interrupted teaching samples")
            self._clear_teaching()
        self.player = None
        self.control_trace.clear()
        if not keep_take:
            if self.datasets and not self.datasets.broken:
                self.operation = ("cleanup", self.datasets.discard())
            self.quality = None

    def _snapshot(self):
        datasets = self.datasets
        assist = self.grip_assist
        grip = None
        if self.path and (self.grip_assist_enabled or isinstance(self.path, GrippedPath)):
            grip = {
                "enabled": self.grip_assist_enabled,
                "ready": bool(assist and assist.prefix <= self.player.cursor < assist.duration),
                "direction": assist.direction if assist else 0,
                "offset": round(assist.offset, 1) if assist else 0,
                "cursor": assist.fraction
                if assist
                else self.progress
                if self.phase == Phase.REPLAY
                else None,
                "taught": self.taught_grip,
                "edited": [
                    min(100, max(0, row[-1] + assist.preview(i / 89))) if assist else row[-1]
                    for i, row in enumerate(self.chart)
                ],
                "saved": isinstance(self.path, GrippedPath),
            }
        return {
            "phase": self.phase,
            "config": self.config.model_dump(mode="json"),
            "connected": self.robot is not None,
            "torque": self.power.state if self.power else "unknown",
            "pose": self.pose,
            "start_pose": self.start_pose,
            "start_pose_id": self.start_pose_id,
            "start_poses": [saved.summary() for saved in self.saved_starts.values()],
            "reset_target": self._reset_goal(),
            "rest_target": self._park_goal(),
            "custom_rest": self.rest_pose is not None,
            "can_continue": self._can_continue(),
            "manual_finish": self.manual_finish,
            "firm_carry_grip": self.firm_carry_grip,
            "replay_firm_carries": self.replay_firm_carries if self.phase == Phase.REPLAY else None,
            "replay_stage": (
                "aligning"
                if self.player.cursor < self.player.path.prefix
                else "settling"
                if self.player.ended is not None
                else "path"
            )
            if self.phase == Phase.REPLAY
            else None,
            "at_rest": self._park_goal() is not None and self._near(self._park_goal()),
            "stopped_early": self.stopped_early,
            "at_start": self._near(self._reset_goal()),
            "at_path_end": bool(self.path and self._near(self.path.end)),
            "reset_action": self.reset_action,
            "scene_reversed": self.scene_reversed,
            "grip": grip,
            "tracking_joint": self.player.tracking_joint if self.player else None,
            "limits": self.robot.limits if self.robot else None,
            "cameras": self.cameras.status() if self.cameras else {},
            "episode": 1 + (datasets.accepted if datasets else 0),
            "accepted": datasets.accepted if datasets else 0,
            "frames": datasets.frames if datasets else 0,
            "recording": self.phase in {Phase.REPLAY, Phase.MANUAL} and self.record_index > 0,
            "quality": self.quality,
            "elapsed": round(self.elapsed, 2),
            "progress": self.progress,
            "fault": self.fault,
            "notice": self.notice,
            "operation": self.operation[0] if self.operation else None,
            "path": {
                "id": self.path_id,
                "duration": round(self.path.duration, 2),
                "teaching_duration": round(self.path.teaching_duration, 2),
                "replay_profile": (
                    ReplayProfile.FAST
                    if self.path.settings == FAST_REPLAY
                    else ReplayProfile.STANDARD
                ),
                "keyframes": self.path.keyframes,
                "chart": self.chart,
            }
            if self.path
            else None,
            "paths": self.paths,
            "dataset_path": str(datasets.root) if datasets else None,
            "repo_id": datasets.repo_id if datasets else None,
        }


def workspace_path() -> Path:
    configured = os.environ.get("GHOSTPATH_DATA")
    return Path(configured).expanduser().resolve() if configured else Path("data").resolve()
