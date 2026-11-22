# GhostPath

**Teach by hand. Replay without a trace.**

GhostPath records a hand-guided trajectory from one SO-101 arm, then replays it while collecting hand-free imitation-learning demonstrations. The local workspace includes a simulator, selectable RGB cameras, joint telemetry, and episode review. Accepted episodes are written directly to LeRobotDataset v3.0.

## Quick start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and [Node.js](https://nodejs.org/) 22.12 or newer. From the repository root, run:

```sh
./start.sh
```

Open [GhostPath on localhost](http://127.0.0.1:8765/). Choose **Connect simulator → Use current position → Example path**. When teaching pauses, choose **Boomerang scene reset**, then **Replay & record**. Turn on **Episode options → Keyboard finish** to try the handoff. Review the captured views, save the episode, and select **Session → Finish dataset**. The virtual cameras and simulated joint lag let you try the workflow without a robot.

The first launch installs Python 3.12, the LeRobot recording dependencies, and the frontend dependencies as needed. Pass `--data /path/to/workspace` or `--port 8765` to change storage or the server port. Use one browser tab for control.

## Connect an SO-101

```sh
./start.sh --hardware
```

1. Set up the motor IDs and calibrate the arm with LeRobot 0.7.0. In GhostPath, select **SO-101**, scan the USB serial ports, and select the device. Enter the same calibration ID. Set a custom calibration directory in **Capture preferences** when needed.
2. Give the session a task description and choose a collection mode. Enter a dataset name in setup. GhostPath stores each session at `data/datasets/<name>` and requires a fresh directory.
3. Edit each camera row to set its local OpenCV device index, resolution, and capture FPS. Linux also supports a `/dev/videoN` source. Use the **Record** switches to include specific cameras in the dataset. Switches left off preserve live previews. Remove all camera rows for joint-only collection. On macOS, grant camera access to the app launching GhostPath if prompted.
4. Connect and inspect the camera views and joint feedback. The dataset FPS defaults to 20 when switching from the simulator to the SO-101. Leave capture-rate headroom for independent camera clocks. Lower dataset FPS or resolution if capture validation reports missing frames or excessive skew. To look up local camera indices, run `uv run --extra hardware lerobot-find-cameras opencv` while GhostPath is disconnected.

On macOS, each camera runs in a separate process. The camera monitor shows **RECONNECTING** if a stream fails, and GhostPath starts a fresh capture process for the same physical device. Check that both recorded views are live before starting another take.

Follow the official hardware instructions for the serial connection and calibration. [LeRobot SO-101 setup](https://huggingface.co/docs/lerobot/v0.7.0/so101)

## Collect demonstrations

1. **Choose a start pose.** Support the arm and release torque if you need to position it by hand. **Use current position** saves and selects the measured pose. You can also pick from **Saved start poses**. For a saved pose elsewhere, move the arm by hand to match it or turn on torque and use **Return to start**. **Start teaching** becomes available when the joint readings match the selected pose. Saved poses are tied to the robot ID and calibration fingerprint.
2. **Teach and pause.** Support the arm and any object before releasing torque. Guide all six joints, including the gripper. Hold still while selecting **Stop teaching** so GhostPath can engage torque at the measured pose. **Continue teaching** appends to the same path from that endpoint. Time spent paused is excluded.
3. **Reset the scene.** **Boomerang scene reset** retraces the arm and gripper motions from the taught endpoint to the selected start. Use it for reversible tasks and verify the object positions afterward. **Return to start** or **Manual reset → Hold & verify** lets you move the arm and reset the objects by hand. From the start, Boomerang can approach the taught endpoint first. Each saved path retains the start it was taught from.
   **Firm grip** applies to Boomerang and replay and is off by default. Enable it for rigid objects that slip. GhostPath looks for a close, arm movement with a steady gripper position, and a release. It adds a small closure margin inside that interval and keeps the arm path's timing. Carry detection uses joint positions alone.
   Enable **Episode options → Keyboard grip adjustment** to hold **C** to close more or **O** to open more. Release the key to ease back to the current grip profile. The timeline compares the taught grip with the adjustment, and a completed Boomerang saves your adjustments for replay.
4. **Replay & record.** Clear the arm’s path and camera views. Capture starts on the first replay tick and includes the initial alignment. The ghost path progress bar follows the arm and shows when it is settling. With **Episode options → Keyboard finish** enabled, GhostPath displays the shortcuts during replay and unlocks them after the arm settles. Keyboard actions continue in the same recording.
   In **Setup → Capture preferences**, **Target replay pace** sets the requested pace relative to the teaching recording. At **1× · Teaching pace**, replay follows the taught cadence wherever the joint speed and acceleration limits allow it. The trajectory shows the taught duration and planned path duration. Alignment, tracking slowdowns, and endpoint settling can add to the recording time.
   **Replay speed and acceleration limits** apply to replay and Boomerang. The arm-joint caps are 50°/s and 170°/s² with **Standard limits** and 120°/s and 360°/s² with **Higher limits**. Parking and keyboard control use the standard limits. Saved grip edits use the limits they were recorded with.
   Use **Discard episode** during teaching or reset to end the draft. Saved episodes and reusable paths stay available.
5. **Review.** Scrub the captured views and inspect timing and tracking metrics. **Retry** discards the take and returns to reset with the current path. **Discard** lets you decide what to do next. **Save episode** commits the take.
6. **Continue or finish.** Use **Next episode** to collect another take or **Session → Finish dataset** to close the dataset. There is no session episode-count cap. Individual teaching paths are limited to three minutes, and bounded replay paths to ten minutes. Saved paths can be loaded in a later session with the same arm calibration.

| Collection mode | Behavior |
| --- | --- |
| Teach every episode | The next episode starts at teaching. You can explicitly reuse the current path. |
| Repeat path | The next episode starts at reset with the same path. You can explicitly teach a replacement. |
| Hybrid / manual finish | Preselects **Episode options → Keyboard finish**. You can change the option before each take. Hold **Space** plus a displayed shortcut, or use the on-screen jog controls. Select **Finish recording** to review. |

During keyboard teleoperation, hold **Space** together with a direction key. Releasing **Space** holds the current position.

| Joint | First direction | Second direction |
| --- | --- | --- |
| Shoulder pan | **A** left | **D** right |
| Shoulder lift | **W** up | **S** down |
| Elbow | **↑** bend | **↓** straighten |
| Wrist flex | **I** up | **K** down |
| Wrist roll | **J** left | **L** right |
| Gripper | **C** close | **O** open |

**Stop recording** holds the arm and sends the captured portion to review. Takes with fewer than three frames are discarded. Saved early stops are marked in the manifest. **Stop motion** or **Esc** aborts active motion and discards the take. Outside motion and capture, **Turn torque off** frees the supported arm to move by hand. **Turn torque on** holds the measured position.

Keep the control tab active during motion. The backend stops active operations if the control link times out. During a fault it attempts a measured hold on energized motors and leaves disabled motors off. Torque can remain on after disconnect. A software stop depends on the computer, serial link, and servos. Keep a physical way to cut motor power and support the arm in case torque drops. Encoder checks and speed caps cannot verify obstacle clearance.

On the SO-101, direct moves use the servos' acceleration and velocity profiles and synchronize their arrival using per-joint limits. Taught-path playback reads encoders on a 60 Hz control loop and sends bounded lookahead goals at up to about 30 Hz. The shoulder and elbow use slow, limited load correction based on encoder feedback. Persistent stalls stop the arm and identify the joint, encoder reading, and goal. GhostPath requires stable encoder feedback within 2° on the arm and 2.5 percentage points on the gripper before completing automatic motion. Servo gains, friction, backlash, and load affect achievable tracking accuracy.

**Dataset FPS** sets the sampling rate for the exported episode independently of arm control. Use **20 FPS** to start with two 30 FPS cameras. Use **30 FPS** when all sources sustain synchronized frames without drops, or **15 FPS** for a lower-bandwidth recording of slower tasks. LeRobot’s SO-101 recording example uses 30 FPS cameras. [LeRobot recording guide](https://huggingface.co/docs/lerobot/main/cheat-sheet)

## Dataset and timing

GhostPath uses the official LeRobot 0.7.0 writer and v3.0 metadata, Parquet, and H.264 videos. Every accepted episode is finalized as a checkpoint before the next one begins. **Session → Finish dataset** closes the workspace’s collection lifecycle. Everything stays on disk locally. [LeRobotDataset API](https://huggingface.co/docs/lerobot/v0.7.0/api/datasets)

| Field | Contents |
| --- | --- |
| `observation.state` | Measured joint positions from the encoders. |
| `action` | Position commands sent after bounds and hardware quantization. |
| `observation.images.<name>` | RGB video for each selected camera. |
| `timestamp` | Official per-episode, constant-FPS LeRobot time. |
| `ghostpath.actual_timestamp` | Actual monotonic observation time relative to the first sampled robot observation. |
| `ghostpath.camera_timestamps` | Per-camera host capture times relative to that same origin, in feature-name order. Present when recording video. |
| `ghostpath.manual` | True for powered manual continuation frames. |
| Episode, frame and task indices | Standard LeRobot indices and task metadata. |

Joints are ordered `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`. Feature names use LeRobot’s `.pos` suffix. Arm joints use degrees and the gripper uses percent of calibrated travel. Hardware camera timestamps reflect host arrival. GhostPath selects a set of unused frames together, within 90% of one dataset frame interval of the robot sample and of each other. This is software synchronization across independent devices. Raw teaching lives in `data/paths/` as passive encoder samples and is kept out of active action labels. A `ghostpath.json` manifest lists the raw path and capture quality for each accepted episode.

During recording, GhostPath looks one robot sample ahead to avoid using a camera frame that the next sample needs. It can omit one isolated camera gap every two seconds. Review and the episode manifest report the omitted samples. The LeRobot video uses the configured constant FPS, and the actual timestamp fields retain the gaps.

Review compares the teaching and planned path times with the measured alignment, traversal, and settling times for a completed replay. Path error compares encoder readings with the replay reference at the recorded sample time, after alignment. Command-to-encoder gaps also include any servo lookahead and load compensation. Both summaries are stored in the episode manifest.

Open a saved dataset with LeRobot directly:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    "local/block-placement",
    root="data/datasets/block-placement",
    video_backend="pyav",
)
print(dataset.num_episodes, dataset[0]["observation.state"])
```

Use that same local root and repo ID in your LeRobot training configuration. Adjust policy image inputs to the selected camera names. Simulator datasets carry the `so101_simulated` robot type.

## Development

```sh
uv sync --extra dev --extra record
uv run --extra dev --extra record pytest -q
uv run --extra dev ruff check src tests
uv run --extra dev ruff format --check src tests
npm --prefix frontend ci
npm --prefix frontend run build
```

For frontend hot reload, run `uv run --extra record ghostpath` and `npm --prefix frontend run dev` in separate terminals. Vite proxies the local API and WebSocket during development.

- `robot.py` and `torque.py` adapt the LeRobot follower and manage verified torque transitions. The measured-goal seeding, joint units, and bounded jog logic build on the existing `lerobot-tools` SO-101 implementation.
- `trajectory.py` preserves raw samples and builds a monotone quintic path through significant points, including gripper reversals. It retimes individual segments against velocity and acceleration bounds and paces playback from measured tracking error.
- `cameras.py` and `dataset.py` capture and write outside the control loop. Reported stream failures, repeated camera gaps, and a backed-up writer abort the take.
- `session.py` owns the robot and workflow state on one control thread. `api.py` provides the local FastAPI and WebSocket entry points. The React and TypeScript workspace lives in `frontend/`.

Licensed under the [MIT License](LICENSE).
