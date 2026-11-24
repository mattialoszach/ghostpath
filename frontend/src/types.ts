export const joints = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper'] as const
export type Joint = typeof joints[number]
export type Pose = Record<Joint, number>
export type Phase = 'setup' | 'ready' | 'practice' | 'teaching' | 'teach_paused' | 'reset' | 'reset_released' | 'returning' |
  'replay' | 'manual' | 'processing' | 'review' | 'saving' | 'next' | 'finished' | 'fault'
export type Mode = 'teach' | 'repeat' | 'hybrid'
export type ReplayProfile = 'standard' | 'fast'
export type JogSpeed = 'precision' | 'normal' | 'fast'

export const replayProfileLabels: Record<ReplayProfile, string> = {
  standard: 'Standard limits',
  fast: 'Higher limits',
}

export interface CameraSpec {
  name: string
  source: number | string
  width: number
  height: number
  capture_fps: number
  record: boolean
}

export interface Config {
  device: 'simulation' | 'so101'
  port: string
  robot_id: string
  calibration_dir: string | null
  task: string
  dataset: string
  fps: number
  mac_camera_buffer_ms: 75 | 68 | 60 | 0
  speed: number
  replay_profile: ReplayProfile
  mode: Mode
  automatic_reset: boolean
  cameras: CameraSpec[]
}

export interface Quality {
  frames: number
  seconds: number
  capture_span_seconds?: number
  max_gap_ms: number
  camera_skew_ms: number
  skipped_camera_samples?: number
  discontinuities?: number[]
  sensor_offset_ms: number
  encoder_read_ms?: TimingStats | null
  writer_queue_ms?: TimingStats
  camera_offset_ms?: Record<string, { p1: number; p50: number; p95: number; p99: number }>
  camera_timing?: Record<string, {
    fps: number | null
    image_age_ms: TimingStats
    delivery_ms: TimingStats
    camera_callback_ms?: TimingStats
    app_delivery_ms?: TimingStats
    arrival_lead_ms?: TimingStats
    wait_ms: TimingStats
  }>
  max_arm_error_deg: number
  max_gripper_error_pct: number
  max_arm_path_error_deg?: number | null
  arm_path_peak?: { joint: Joint; seconds: number; excursion_deg?: number; excursion_ms?: number } | null
  max_gripper_path_error_pct?: number | null
  replay_timing?: {
    alignment_seconds: number
    motion_seconds: number
    settle_seconds: number
  } | null
  review_frames: number
}

type TimingStats = { p50: number; p95: number; p99: number; max: number }

export interface CameraMode {
  device_id?: string
  label: string
  width: number
  height: number
  nominal_fps?: number | null
  format?: string
  timestamp_source: 'avfoundation_pts' | 'host_read'
}

export interface GripState {
  enabled: boolean
  ready: boolean
  direction: -1 | 0 | 1
  offset: number
  cursor: number | null
  taught: number[]
  edited: number[]
  saved: boolean
}

export interface State {
  phase: Phase
  config: Config
  connected: boolean
  recording: boolean
  torque: 'on' | 'off' | 'mixed' | 'unknown'
  pose: Pose | null
  start_pose: Pose | null
  start_pose_id: string | null
  start_poses: { id: string; label: string; created: string; legacy: boolean }[]
  reset_target: Pose | null
  rest_target: Pose | null
  at_rest: boolean
  can_continue: boolean
  manual_finish: boolean
  jog_speed: JogSpeed
  can_set_jog_speed: boolean
  firm_carry_grip: boolean
  replay_firm_carries: number | null
  replay_stage: 'aligning' | 'path' | 'settling' | null
  stopped_early: boolean
  at_start: boolean
  at_path_end: boolean
  reset_action: 'direct' | 'park' | 'reverse_approach' | 'reverse' | null
  scene_reversed: boolean
  grip: GripState | null
  tracking_joint: Joint | null
  limits: Record<Joint, [number, number]> | null
  cameras: Record<string, { ready: boolean; retrying?: boolean; problem: string; fps: number; mode?: CameraMode | null }>
  episode: number
  accepted: number
  frames: number
  quality: Quality | null
  elapsed: number
  progress: number
  fault: string
  notice: string
  operation: string | null
  path: {
    id: string
    duration: number
    teaching_duration: number
    replay_profile: ReplayProfile
    keyframes: number
    chart: number[][]
  } | null
  paths: { id: string; task: string; created: string; duration: number; grip_edited: boolean }[]
  dataset_path: string | null
  repo_id: string | null
}

export const camera = (name: string, source: number): CameraSpec => ({
  name, source, width: 640, height: 480, capture_fps: 30, record: true,
})

export const defaults: Config = {
  device: 'simulation', port: '', robot_id: 'my_awesome_follower_arm', calibration_dir: null,
  task: 'Pick up the orange block and place it in the tray', dataset: 'block-placement',
  fps: 20, mac_camera_buffer_ms: 60, speed: 1, replay_profile: 'standard', mode: 'teach', automatic_reset: false,
  cameras: [camera('workspace', 0), camera('wrist', 1)],
}

export const jointLabels: Record<Joint, string> = {
  shoulder_pan: 'Shoulder pan', shoulder_lift: 'Shoulder lift', elbow_flex: 'Elbow',
  wrist_flex: 'Wrist flex', wrist_roll: 'Wrist roll', gripper: 'Gripper',
}

export type Send = (action: string, extra?: Record<string, unknown>) => void

export const elapsed = (seconds: number) => {
  const minutes = Math.floor(seconds / 60).toString().padStart(2, '0')
  return `${minutes}:${(seconds % 60).toFixed(2).padStart(5, '0')}`
}

export const humanize = (name: string) => name.replaceAll('_', ' ').replace(/^./, match => match.toUpperCase())
