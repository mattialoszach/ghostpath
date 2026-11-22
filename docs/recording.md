# Recording guide

Start with the [simulator walkthrough](../README.md#try-it-without-a-robot), then connect a calibrated SO-101 follower if you have one. GhostPath pins LeRobot to 0.7.0. You don’t need to set up a leader for this workflow.

## Set up the cameras

- Expand **Cameras** to set each camera’s OpenCV device index, resolution, and capture FPS. On Linux, a `/dev/videoN` source also works. To find the indices, run `uv run --locked --extra hardware lerobot-find-cameras opencv` with GhostPath disconnected.
- Use the **Record** switches to include or exclude views. A camera with the switch off is preview-only. Remove the camera rows if you don’t need live video either.
- Start with **20 dataset FPS** for two **30 FPS** cameras. Dataset FPS controls the export rate, not how often the arm is controlled. If frames go missing or the cameras can’t keep up, try a lower dataset FPS or smaller camera resolution.
- On macOS, grant camera access to the app you launch GhostPath from. If a camera says **Reconnecting**, GhostPath is trying to reopen it. Check the camera view and its reported rate before the next take.

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

GhostPath uses the LeRobot v3.0 writer (Parquet, H.264 video, and metadata). Accepted episodes are checkpointed locally as you save them. The `ghostpath.json` manifest links each episode to its taught path and reports capture quality. By default raw teaching traces and saved poses stay in the local workspace; none of this is uploaded.

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    "local/block-placement",
    root="data/datasets/block-placement",
    video_backend="pyav",
)
print(dataset.num_episodes, dataset[0]["observation.state"])
```

Use the same local root and repo ID in your LeRobot training configuration, and match the policy image inputs to your recorded camera names. The simulator writes `so101_simulated` as its robot type.

| Field | Contents |
| --- | --- |
| `observation.state` | Measured encoder positions. |
| `action` | Position commands sent after bounds and hardware quantization. |
| `observation.images.<name>` | RGB video from a recorded camera. |
| `timestamp` | LeRobot’s constant-FPS episode time. |
| `ghostpath.actual_timestamp` | Monotonic observation time since the first sampled robot observation. |
| `ghostpath.camera_timestamps` | Per-camera host arrival times relative to that same origin, in camera feature order. Present with video. |
| `ghostpath.manual` | Whether a frame was recorded during powered manual continuation. |

Joint order is `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`. LeRobot feature names use the `.pos` suffix. The arm uses degrees and the gripper uses percent of calibrated travel. Camera matching uses host timestamps, not a hardware trigger. An isolated missing sample can be skipped, while repeated gaps abort the take. The actual timestamp fields retain any gaps hidden by the constant-FPS export.

## Code map

- [`robot.py`](../src/ghostpath/robot.py) and [`torque.py`](../src/ghostpath/torque.py) adapt the LeRobot follower and verify torque transitions. Measured-goal seeding, joint units, and bounded jog logic build on the `lerobot-tools` SO-101 implementation.
- [`trajectory.py`](../src/ghostpath/trajectory.py) builds a bounded replay path from the teaching samples without replacing them.
- [`cameras.py`](../src/ghostpath/cameras.py) and [`dataset.py`](../src/ghostpath/dataset.py) handle camera matching and dataset writing outside the control loop.
- [`session.py`](../src/ghostpath/session.py) owns the workflow and robot control. [`api.py`](../src/ghostpath/api.py) serves the local API and workspace.
