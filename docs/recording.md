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

To compare pairing delays on the same two-camera trace:

```sh
uv run --locked --extra hardware python -m ghostpath.camera_check 0 1 --seconds 60 --dataset-fps 20 --compare --timing-trace /tmp/ghostpath-camera-timing.json
```

The trace contains sequence numbers, presentation, arrival, and availability times for every delivered frame, including warm-up, plus the nominal ticks and synthetic observation cutoffs. It contains no images. The comparison reports image age, camera skew, and missed rows for 105, 90, 75, 68, and 60 ms targets and for newest-available causal matching (0 ms). Use `--buffer-ms 68`, `--buffer-ms 60`, or `--buffer-ms 0` to verify a temporary export with that setting. Before collecting in bulk, choose the lowest-delay setting that has no missed rows in repeated camera checks, then confirm on several supervised pilot takes with the final lighting, resolution, and USB setup. Compare p95 image age and camera skew, and check for skips and writer backlog. Synthetic cutoffs don't include encoder scheduling jitter.

## Collect an ACT baseline

- Keep the same camera names, mounts, resolution, and lighting for the dataset. In the workspace view, check that the full pickup area and destination are visible; in the wrist view, check that the jaws and target are sharp through approach and release. If two identical blocks are visible, remove the distractor or make the task instruction and demonstrations unambiguous.
- Use **Standard limits** for the first supervised take. Review grasp, carry, and release, and keep only successful, continuous takes. Start with a few successful pilots before collecting a larger set. Vary the block's starting positions within the area you plan to use at deployment instead of repeating one identical reset; reserve whole episodes or positions for a held-out evaluation.
- Keep your VGA recordings. Start ACT with both full frames. If training or inference is the bottleneck, compare an aspect-preserving **320 × 240** input on the same episodes and held-out rollouts, using the same preprocessing at deployment. Don't crop away a pickup or placement area to fit a model.
- PPO doesn't prescribe an image size. If you use it later, pick a size for the simulator or environment and measure rollout speed and success. Standard PPO needs policy rollouts and rewards; these demonstration datasets are directly useful for imitation.

## Teach and reset

1. **Set a start pose.** **Use current position** sets a temporary start for this session, or you can select a saved start. Choose **Save this start for later** and give it a name only if you want to keep it. Earlier captures are grouped separately and can be deleted individually. If the arm is away from the selected start, move it by hand with torque off or use **Return to start** with torque on. Teaching starts only when the pose matches.
2. **Teach the motion.** Support the arm and gripper when you release torque, then guide all six joints, including the gripper. Hold the arm still when you stop teaching so it can engage torque at the measured position. **Continue teaching** picks up at that endpoint without adding the pause to the path.
3. **Restore the scene.** **Boomerang scene reset** retraces the taught motion and can restore object placement for reversible tasks. From the start pose it may move to the taught endpoint first. Clear the full route and check the result yourself. Otherwise, **Return to start** moves the arm but not the objects. **Manual reset → Hold & verify** lets you position both by hand.

Under **Rest position**, use **Save current as rest** to keep a separate parking pose. It is never used as the task start. Saving or clearing rest does not move the arm, and GhostPath never parks automatically; use **Return to rest** only after emptying the gripper and clearing the full route. Start and rest saves are scoped to the robot ID and calibration. Taught paths still keep their own start and raw samples, without creating start-library entries.

To remove old parking poses or recordings, finish any active dataset, disconnect, and open **Clean up saved data** on the setup screen. It lists saved parking poses across calibrations and local LeRobot v3 recording datasets. Deleting a recording permanently removes the whole dataset, not an individual episode. Deleting a parking pose leaves previous recordings unchanged. Neither action removes raw taught paths or saved task starts.

If the gripper slips on a rigid object, try **Episode options → Firm grip**. It adds a small extra closure to detected carries without changing the arm path. Carry detection uses joint readings, not images or contact sensing. You can also enable **Keyboard grip adjustment** for Boomerang: hold **C** to close or **O** to open, then release the key to ease back. A completed Boomerang saves those adjustments for replay without overwriting the raw teaching samples.

## Record and review

**Replay & record** captures from the first replay tick, including any initial alignment. If the arm falls behind the taught path, playback slows to let it catch up. **Target replay pace** requests a speed relative to the teaching pass; the actual take can be longer because of speed and acceleration limits, alignment, and settling. Start with **Standard limits**. **Higher limits** let replay and Boomerang move faster, but still apply bounds. Saved grip edits retain the limits they were recorded with.

| Collection mode | Default |
| --- | --- |
| Teach every episode | **Teach a new path** is the suggested next step. |
| Repeat path | **Reuse this path** is suggested, so you can reset the scene and record again. |
| Hybrid / manual finish | **Manual finish** is selected. Override individual joints during replay, then continue manually once replay settles. |

Use **Practice controls** before teaching to try the same controls without recording or saving poses. **Done** holds the current position and exits practice. **Return to rest** exits practice and parks at the explicitly saved rest pose; clear and supervise the route first.

With **Episode options → Manual finish** selected, hold **Space** and a direction key in either order, or hold an on-screen jog button, to override only that joint's replay command. The other joints keep following the taught path. A keyboard chord held during alignment takes effect only once the arm starts following the path. Use **Focus keyboard controls** if you clicked elsewhere. Release the key or button to ease the joint back toward the live replay command. At the end of the path, recording continues under full manual control from the current goals; any active direction you are still holding continues without a second press. Clear and supervise space for corrections as well as the taught path. Use **Take over** instead if you want to stop the taught path immediately and jog all joints while recording continues.

Once in manual control, hold a direction key in the focused panel (no Space required) or hold an on-screen jog button. Loss of focus or stale movement input cancels the jog immediately. The keyboard pairs are **A/D** (shoulder pan), **W/S** (shoulder lift), **↑/↓** (elbow), **I/K** (wrist flex), **J/L** (wrist roll), and **C/O** (gripper). Use **Finish recording** to review the full take.

A held jog that reaches its tracking limit without encoder progress stops motion and names the affected arm joint. Clear the route and check the load before retrying. This check does not apply to the gripper, which can stop against an object during a grasp.

Use **Jog speed** to select **Precision** (1×, the default), **Normal** (2×), or **Fast** (3×). Release the controls and let the jog settle or rejoin before switching. The arm jog target limits are 12.5, 25, and 37.5°/s; the gripper limits are 12%, 24%, and 36% of calibrated travel per second. During mixed replay these limit how fast manual targets change; the servos retain the replay profile. The same limits apply to keyboard and on-screen jogging. Allow more stopping distance at higher speeds. The setting carries from practice into recording until you disconnect. It does not change replay speed.

In review, scrub the camera previews and check **Capture and tracking details**. The scrubber shows sampled stills, not every exported frame. Path error compares encoder readings to the replay reference on ticks without manual input or rejoining. The command-to-encoder gap also reflects servo lookahead and load compensation, so the two numbers don’t have to match. **Retry this path** discards the take and returns to reset. **Save episode** keeps it. When you’re done, **Finish dataset** closes the session and displays the local export path.

**Stop recording** holds the arm and sends any valid partial take to review. Takes shorter than three frames are discarded; saved early stops are marked in the manifest. **Stop motion** or **Esc** aborts active motion and discards the in-progress take. Keep the control tab active during motion. The backend stops active operations if the control link times out. Torque can remain on after a fault or disconnect, and it can drop when power is removed. Support the arm, clear and supervise its path, and keep a physical power cutoff within reach. Encoder checks can’t tell whether a path is clear.

## Read the export

GhostPath uses the LeRobot v3.0 writer (Parquet, H.264 video, and metadata). Video encoding runs during capture; a save is rejected unless the Parquet row count and every configured video's frame count and dimensions match the episode. The `ghostpath.json` manifest links each accepted episode to its taught path and reports native modes, device IDs on macOS, and timing percentiles. **Capture and tracking details** shows image age at encoder read end, camera timestamp to callback, callback to app availability, and how far ahead of encoder read end the camera callback arrived. Encoder read duration and writer queue time are reported separately. Saved episodes also keep the full control trace for comparing the measured position, reference, command, and replay rate during tracking excursions. By default raw teaching traces and saved poses stay in the local workspace; none of this is uploaded.

Manual override and continuation samples are labeled `ghostpath.manual` in the same episode. During mixed replay the control trace lists `override_joints` and retains the replay cursor. For an explicit full takeover, `manual_takeover_seconds` in `ghostpath.json` marks when replay yielded control, relative to the start of recording.

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
| `ghostpath.manual` | Whether one or more joints were under manual override (including rejoining), or replay had handed off to full manual control. |

Joint order is `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`. LeRobot feature names use the `.pos` suffix. The arm uses degrees and the gripper uses percent of calibrated travel. The policy needs a matching bounded servo-goal interface at the export cadence; do not send `action` back through a controller that adds the same lookahead and trim again.

GhostPath selects unused frames that were available by the end of the encoder read, before handing the selected pixels off to the writer. **Capture preferences → Camera timing (macOS)** defaults to 60 ms. If a pilot has camera gaps or excessive skew, try 68 ms, then 75 ms if needed. Newest-available matching (0 ms) is experimental: it can use a newer frame now and leave no usable pair for the next tick. Nonzero settings require a frame timestamp at least that far behind the nominal tick; the buffer is not the measured image age. Check `camera_timing.<view>.image_age_ms` and `skipped_camera_samples` from pilot takes before a large collection run. No buffer setting removes latency from exposure, USB transport, or the camera driver.

At deployment, reproduce the dataset's `observation_policy.avfoundation_buffer_seconds` using `Cameras.frames`: pass the encoder read end as `cutoff`, the nominal policy tick as `scheduled`, and the last accepted sequence per view as `after`. Use the same camera modes and dataset cadence, and do not silently pool takes with different buffer settings for a timing-sensitive baseline. A flash visible to both cameras can test relative optical timing, but it is not timestamped by the encoder. Use a shared timing reference if you need to measure absolute exposure-to-encoder latency.

An isolated missing sample can be skipped, while repeated gaps abort the take. `discontinuities` in the manifest contains the exported frame indices where a new segment begins. Filter those episodes out as above, or use a loader that refuses observation and action windows spanning two `ghostpath.segment_index` values. Stock LeRobot uses the constant-FPS timeline, not these segment markers. Older version 1 manifests use host camera arrival timestamps and may contain frames delivered after the robot sample; do not silently mix the two timing conventions.

## Take a policy live

- **Train from the exported inputs.** `ghostpath.json` → `policy_io` lists the dataset FPS, RGB camera keys, dimensions, capture FPS, joint order, units, and action semantics. For ACT use `observation.state` and the recorded `observation.images.*` views to predict `action`. Keep the trained checkpoint's normalization and image preprocessing with it; if you train at a smaller image size, apply the same resize or crop live.
- **Match the live timing and views.** Keep the same camera names, placement, and capture modes, and apply each output action at the dataset cadence (20 FPS by default), not the camera cadence (30 FPS by default). Reuse the `observation_policy` pairing rules, including the encoder cutoff and no frame reuse. `episodes[].cameras` identifies the physical sources and negotiated modes when available. Don't silently swap views or proceed on a missing frame.
- **Match the robot and reset separately.** `policy_io.robot` records the calibration fingerprint and position limits, not calibration files or the local serial port. Use the same calibration and degrees/percent units. A live controller must independently enforce motion, tracking, and stop constraints on predicted servo goals; do not apply the replay lookahead or load trim again. `episodes[].poses.task_start` is the reset target for that take; `poses.rest` is the optional parking target (or `null`). Save rest before recording to include it; later changes don't rewrite old takes. Neither pose is a policy input or permission to move automatically. Before a supervised rollout, clear the route, reset the task scene, and start the arm near the episode start, not the parking pose.

Each accepted replay links to a `control/episode-*.json` trace. It preserves every replay and manual control tick (including intermediate ticks), its stage, raw measured state, reference when available, resulting servo goal, and the host encoder and send-call intervals. An unchanged goal may not cause a bus write; the send interval does not measure when the motors act. Use the trace around the reported peak to see how long a joint stayed off its reference.

## Code map

- [`robot.py`](../src/ghostpath/robot.py) and [`torque.py`](../src/ghostpath/torque.py) adapt the LeRobot follower and verify torque transitions. Measured-goal seeding, joint units, and bounded jog logic build on the `lerobot-tools` SO-101 implementation.
- [`trajectory.py`](../src/ghostpath/trajectory.py) builds a bounded replay path from the teaching samples without replacing them.
- [`cameras.py`](../src/ghostpath/cameras.py) separates frame reception from preview JPEG encoding. [`dataset.py`](../src/ghostpath/dataset.py) pins the causal camera frames without waiting for I/O, and writes datasets on a separate, bounded queue.
- [`session.py`](../src/ghostpath/session.py) owns the workflow and robot control. [`api.py`](../src/ghostpath/api.py) serves the local API and workspace.
