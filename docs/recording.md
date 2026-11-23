# Recording guide

Start with the [simulator walkthrough](../README.md#try-it-without-a-robot), then connect a calibrated SO-101 follower if you have one. GhostPath pins LeRobot to 0.7.0. You don’t need to set up a leader for this workflow.

## Set up the cameras

- Expand **Cameras** to set each camera’s source, resolution, and capture FPS. On Linux, a `/dev/videoN` source also works. On macOS, use the `device_id` reported by the check below as the source to bind a view to the same device instead of relying on its index.
- Use the **Record** switches to include or exclude views. A camera with the switch off is preview-only. Remove the camera rows if you don’t need live video either.
- Default to **640 × 480 at 30 capture FPS** for both views and **20 dataset FPS**. Dataset FPS controls the export rate, not how often the arm is controlled. This lets the cameras supply distinct frames even if they run slightly below nominal FPS. Use 720p or 1080p only if VGA is missing task-relevant detail after you have checked framing and focus.
- On macOS, grant camera access to the app you launch GhostPath from. If a camera says **Reconnecting**, GhostPath is trying to reopen it. Check the camera view and its reported rate before the next take.

With GhostPath disconnected, measure both cameras at once without opening the robot connection:

```sh
uv run --locked --extra hardware python -m ghostpath.camera_check 0 1 --seconds 30 --dataset-fps 20
```

This checks both streams at once and writes and reads back a temporary LeRobot episode using synthetic joint telemetry. It doesn't move the arm or add anything to your datasets. Omit `--dataset-fps` to check camera delivery alone. GhostPath uses native sizes and does not silently resize a different camera mode. On macOS the report uses AVFoundation presentation timestamps; they are mapped to the host clock but are not a physical exposure measurement.

## Collect an ACT baseline

- Keep the same camera names, mounts, resolution, and lighting for the dataset. In the workspace view, check that the full pickup area and destination are visible; in the wrist view, check that the jaws and target are sharp through approach and release. If two identical blocks are visible, remove the distractor or make the task instruction and demonstrations unambiguous.
- Use **Standard limits** for the first supervised take. Review grasp, carry, and release, and keep only successful, continuous takes. Start with a few successful pilots before collecting a larger set. Vary the block's starting positions within the area you plan to use at deployment instead of repeating one identical reset; reserve whole episodes or positions for a held-out evaluation.
- Keep your VGA recordings. Start ACT with both full frames. If training or inference is the bottleneck, compare an aspect-preserving **320 × 240** input on the same episodes and held-out rollouts, using the same preprocessing at deployment. Don't crop away a pickup or placement area to fit a model.
- PPO doesn't prescribe an image size. If you use it later, pick a size for the simulator or environment and measure rollout speed and success. Standard PPO needs policy rollouts and rewards; these demonstration datasets are directly useful for imitation.

## Teach and reset

1. **Set a start pose.** Use **Use current position** to save the measured pose, or select a saved one. If the arm is away from the selected start, move it by hand with torque off or use **Return to start** with torque on. Teaching starts only when the pose matches. Saved poses and paths are tied to the robot ID and calibration.
2. **Teach the motion.** Support the arm and gripper when you release torque, then guide all six joints, including the gripper. Hold the arm still when you stop teaching so it can engage torque at the measured position. **Continue teaching** picks up at that endpoint without adding the pause to the path.
3. **Restore the scene.** **Boomerang scene reset** retraces the taught motion and can restore object placement for reversible tasks. From the start pose it may move to the taught endpoint first. Clear the full route and check the result yourself. Otherwise, **Return to start** moves the arm but not the objects. **Manual reset → Hold & verify** lets you position both by hand.

If the gripper slips on a rigid object, try **Episode options → Firm grip**. It adds a small extra closure to detected carries without changing the arm path. Carry detection uses joint readings, not images or contact sensing. You can also enable **Keyboard grip adjustment** for Boomerang: hold **C** to close or **O** to open, then release the key to ease back. A completed Boomerang saves those adjustments for replay without overwriting the raw teaching samples.

## Record and review

**Replay & record** captures from the first replay tick, including any initial alignment. If the arm falls behind the taught path, playback slows to let it catch up. **Target replay pace** requests a speed relative to the teaching pass; the actual take can be longer because of speed and acceleration limits, alignment, and settling. Start with **Standard limits**. **Higher limits** let replay and Boomerang move faster, but still apply bounds. Saved grip edits retain the limits they were recorded with.

| Collection mode | Default |
| --- | --- |
| Teach every episode | **Teach a new path** is the suggested next step. |
| Repeat path | **Reuse this path** is suggested, so you can reset the scene and record again. |
| Hybrid / manual finish | **Keyboard finish** is selected. After replay settles, you can take over while the same recording continues. |

For keyboard finish, hold **Space** and a direction key shown on screen, or hold one of the jog buttons. Releasing Space or the button stops jogging. Select **Finish recording** to review. The keyboard pairs are **A/D** (shoulder pan), **W/S** (shoulder lift), **↑/↓** (elbow), **I/K** (wrist flex), **J/L** (wrist roll), and **C/O** (gripper).

In review, scrub the camera previews and check **Capture and tracking details**. The scrubber shows sampled stills, not every exported frame. Path error compares encoder readings to the replay reference at that instant. The command-to-encoder gap also reflects servo lookahead and load compensation, so the two numbers don’t have to match. **Retry this path** discards the take and returns to reset. **Save episode** keeps it. When you’re done, **Finish dataset** closes the session and displays the local export path.

**Stop recording** holds the arm and sends any valid partial take to review. Takes shorter than three frames are discarded; saved early stops are marked in the manifest. **Stop motion** or **Esc** aborts active motion and discards the in-progress take. Keep the control tab active during motion. The backend stops active operations if the control link times out. Torque can remain on after a fault or disconnect, and it can drop when power is removed. Support the arm, clear and supervise its path, and keep a physical power cutoff within reach. Encoder checks can’t tell whether a path is clear.

## Read the export

GhostPath uses the LeRobot v3.0 writer (Parquet, H.264 video, and metadata). Video encoding runs during capture; a save is rejected if the encoded frame count does not match the telemetry rows for every view. The `ghostpath.json` manifest links each accepted episode to its taught path and reports measured FPS at preparation, native modes, device IDs on macOS, and signed camera timing percentiles. By default raw teaching traces and saved poses stay in the local workspace; none of this is uploaded.

```python
import json
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

root = Path("data/datasets/block-placement")  # Choose a newly recorded (manifest v2) dataset.
manifest = json.loads((root / "ghostpath.json").read_text())
assert manifest["version"] == 2
episodes = [
    entry["episode"]
    for entry in manifest["episodes"]
    if not entry["skipped_camera_samples"] and not entry["stopped_early"]
]
if not episodes:
    raise ValueError("No continuous episodes selected for training.")
dataset = LeRobotDataset(
    f"local/{root.name}",
    root=root,
    episodes=episodes,
    video_backend="pyav",
)
print(dataset.num_episodes, dataset[0]["observation.state"])
```

Use the same local root and repo ID in your LeRobot training configuration, and match the policy image inputs to your recorded camera names. The simulator writes `so101_simulated` as its robot type.

| Field | Contents |
| --- | --- |
| `observation.state` | Measured encoder positions. |
| `action` | Absolute servo goals after bounds and hardware quantization, including any applicable lookahead and load compensation. |
| `observation.images.<name>` | RGB video from a recorded camera. |
| `timestamp` | LeRobot’s constant-FPS episode time. |
| `ghostpath.actual_timestamp` | Encoder read midpoint relative to the first sampled robot observation. |
| `ghostpath.encoder_read` | Host start and end times of that encoder read, relative to the same origin. |
| `ghostpath.camera_timestamps` | Camera PTS mapped to the monotonic host clock on macOS; host read time elsewhere. Relative to the same origin. |
| `ghostpath.camera_arrival_timestamps` | Host times at the capture callback or read completion, in camera feature order. |
| `ghostpath.camera_available_timestamps` | Host times pixels became available to GhostPath, in camera feature order. |
| `ghostpath.sample_index` / `ghostpath.segment_index` | Original sample number and continuous segment number. A new segment begins after any internal gap. |
| `ghostpath.manual` | Whether a frame was recorded during powered manual continuation. |

Joint order is `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`. LeRobot feature names use the `.pos` suffix. The arm uses degrees and the gripper uses percent of calibrated travel. The policy needs a matching bounded servo-goal interface at the export cadence; do not send `action` back through a controller that adds the same lookahead and trim again.

GhostPath selects unused frames that were available by the end of the encoder read. On macOS it targets frames 105 ms behind the nominal dataset grid to absorb uneven USB delivery and encoder sampling jitter; reproduce this input selection at deployment. This intentionally trades some visual freshness for steadier sampling. A simultaneous flash visible to both cameras, and a visible arm marker in a clear, supervised test, are still needed to calibrate physical camera-to-camera and camera-to-encoder timing.

An isolated missing sample can be skipped, while repeated gaps abort the take. `discontinuities` in the manifest contains the exported frame indices where a new segment begins. Filter those episodes out as above, or use a loader that refuses observation and action windows spanning two `ghostpath.segment_index` values. Stock LeRobot uses the constant-FPS timeline, not these segment markers. Older version 1 manifests use host camera arrival timestamps and may contain frames delivered after the robot sample; do not silently mix the two timing conventions.

Each accepted replay links to a `control/episode-*.json` trace. It preserves every replay and manual control tick (including intermediate ticks), its stage, raw measured state, reference when available, resulting servo goal, and the host encoder and send-call intervals. An unchanged goal may not cause a bus write; the send interval does not measure when the motors act. Use the trace around the reported peak to see how long a joint stayed off its reference.

## Code map

- [`robot.py`](../src/ghostpath/robot.py) and [`torque.py`](../src/ghostpath/torque.py) adapt the LeRobot follower and verify torque transitions. Measured-goal seeding, joint units, and bounded jog logic build on the `lerobot-tools` SO-101 implementation.
- [`trajectory.py`](../src/ghostpath/trajectory.py) builds a bounded replay path from the teaching samples without replacing them.
- [`cameras.py`](../src/ghostpath/cameras.py) and [`dataset.py`](../src/ghostpath/dataset.py) handle camera matching and dataset writing outside the control loop.
- [`session.py`](../src/ghostpath/session.py) owns the workflow and robot control. [`api.py`](../src/ghostpath/api.py) serves the local API and workspace.
