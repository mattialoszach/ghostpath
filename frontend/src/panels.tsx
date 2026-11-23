import {
  Activity, Camera, ChevronDown, Cpu, Crosshair, Eye, EyeOff, Film, FolderOpen,
  Monitor, Plus, Radio, ScanLine, SlidersHorizontal, Trash2, Video,
} from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { jogKeys, keyLabel } from './controls'
import { camera, elapsed, humanize, jointLabels, joints, replayProfileLabels } from './types'
import type { CameraSpec, Config, Joint, Send, State } from './types'

const modes = [
  { value: 'teach', label: 'Teach every episode' },
  { value: 'repeat', label: 'Repeat path' },
  { value: 'hybrid', label: 'Hybrid / manual finish' },
] as const

interface Devices {
  ports: { port: string; label: string }[]
  calibrations: string[]
}

export function SetupPanel({ config, change, connect, available }: {
  config: Config
  change: (config: Config) => void
  connect: () => void
  available: boolean
}) {
  const [devices, setDevices] = useState<Devices>({ ports: [], calibrations: [] })
  const [scan, setScan] = useState('')
  const [scanning, setScanning] = useState(false)
  const [expanded, setExpanded] = useState<number | null>(null)
  const set = <K extends keyof Config>(key: K, value: Config[K]) => change({ ...config, [key]: value })
  const updateCamera = (index: number, patch: Partial<CameraSpec>) => {
    set('cameras', config.cameras.map((item, i) => i === index ? { ...item, ...patch } : item))
  }
  const scanPorts = async () => {
    setScanning(true)
    setScan('')
    try {
      const response = await fetch('/api/devices')
      if (!response.ok) throw Error()
      const result: Devices = await response.json()
      setDevices(result)
      setScan(result.ports.length ? 'Select the port for your follower.' : 'No USB serial device found.')
    } catch {
      setScan('Device scan unavailable.')
    } finally {
      setScanning(false)
    }
  }

  const recorded = config.cameras.filter(camera => camera.record).length
  const addCamera = () => {
    const source = [0, 1, 2, 3, 4, 5, 6, 7].find(i =>
      !config.cameras.some(spec => String(spec.source) === String(i) || spec.name === `camera_${i}`))
    if (source === undefined) return
    setExpanded(config.cameras.length)
    set('cameras', [...config.cameras, camera(`camera_${source}`, source)])
  }
  return <section className="panel setup-panel" aria-label="Collection setup">
    <form onSubmit={event => { event.preventDefault(); connect() }} className="setup-form">
      <div className="setup-basics">
        <fieldset className="setup-fields">
          <legend>Robot</legend>
          <div className="segmented" role="group" aria-label="Robot source">
            <button type="button" aria-pressed={config.device === 'simulation'} className={config.device === 'simulation' ? 'selected' : ''} onClick={() => set('device', 'simulation')}><Monitor size={14} /> Simulator</button>
            <button type="button" aria-pressed={config.device === 'so101'} className={config.device === 'so101' ? 'selected' : ''} onClick={() => change({ ...config, device: 'so101', fps: config.device === 'simulation' && config.fps === 30 ? 20 : config.fps })}><Cpu size={14} /> SO-101</button>
          </div>
          {config.device === 'so101' && <div className="hardware-fields">
            <div className="field">
              <label htmlFor="serial-port">Serial port</label>
              <div className="row">
                <select id="serial-port" value={config.port} onChange={e => set('port', e.target.value)} required>
                  <option value="">Select a device</option>
                  {config.port && !devices.ports.some(p => p.port === config.port) && <option value={config.port}>{config.port}</option>}
                  {devices.ports.map(p => <option key={p.port} value={p.port}>{p.port}</option>)}
                </select>
                <button type="button" className="button" disabled={!available || scanning} onClick={scanPorts}><ScanLine size={16} /> Scan ports</button>
              </div>
            </div>
            {(scanning || scan) && <p className="hint" role="status">{scanning ? 'Scanning…' : scan}</p>}
            <label className="field">LeRobot calibration ID
              <input value={config.robot_id} onChange={e => set('robot_id', e.target.value)} list="calibrations" required />
              <datalist id="calibrations">{devices.calibrations.map(id => <option key={id} value={id} />)}</datalist>
            </label>
          </div>}
        </fieldset>
        <fieldset className="setup-fields">
          <legend>Dataset</legend>
          <label className="field">Dataset name
            <input value={config.dataset} onChange={e => set('dataset', e.target.value)} pattern={String.raw`[a-z0-9][a-z0-9_\-]{0,63}`} required />
          </label>
          <label className="field">Task description
            <textarea value={config.task} maxLength={500} rows={2} onChange={event => set('task', event.target.value)} required />
          </label>
        </fieldset>
      </div>
      <details className="setup-section">
        <summary><Video size={16} /> Cameras <span className="disclosure-value">{recorded} recorded</span><ChevronDown size={14} /></summary>
        <div className="setup-section-content">
        <div className="camera-settings">
          {config.cameras.map((spec, index) => <div className="camera-setting" key={index}>
            <div className="camera-option">
              <button type="button" className="camera-configure" onClick={() => setExpanded(expanded === index ? null : index)} aria-expanded={expanded === index}>
                <Camera size={14} /><span>{humanize(spec.name || 'camera')}</span><span className="source-id">{String(spec.source)}</span><ChevronDown size={12} className={expanded === index ? 'rotate' : ''} />
              </button>
              <button type="button" className={'switch' + (spec.record ? ' switched' : '')} role="switch" aria-label={`Record ${spec.name}`} aria-checked={spec.record} onClick={() => updateCamera(index, { record: !spec.record })}><span /></button>
            </div>
            {expanded === index && <div className="camera-fields">
              <div className="split">
                <label className="field">Name<input value={spec.name} onChange={e => updateCamera(index, { name: e.target.value })} pattern="[a-z][a-z0-9_]{0,23}" required /></label>
                <label className="field">Source index or ID<input value={spec.source} onChange={e => updateCamera(index, { source: e.target.value })} required /></label>
              </div>
              <div className="split">
                <label className="field">Resolution<select value={`${spec.width}x${spec.height}`} onChange={e => { const [width, height] = e.target.value.split('x').map(Number); updateCamera(index, { width, height }) }}>
                  <option value="320x240">320 × 240</option><option value="640x480">640 × 480</option><option value="1280x720">1280 × 720</option><option value="1920x1080">1920 × 1080</option>
                </select></label>
                <label className="field">Capture FPS<select value={spec.capture_fps} onChange={e => updateCamera(index, { capture_fps: Number(e.target.value) })}><option>15</option><option>30</option><option>60</option></select></label>
              </div>
              <button type="button" className="text-button" onClick={() => { set('cameras', config.cameras.filter((_item, i) => i !== index)); setExpanded(null) }}><Trash2 size={13} /> Remove camera</button>
            </div>}
          </div>)}
          {!config.cameras.length && <p className="hint">Joint telemetry only.</p>}
        </div>
          <button type="button" className="text-button" disabled={config.cameras.length >= 4} onClick={addCamera}><Plus size={14} /> Add camera</button>
        </div>
      </details>
      <details className="setup-section">
        <summary><SlidersHorizontal size={16} /> Capture preferences <span className="disclosure-value">{config.fps} FPS</span><ChevronDown size={14} /></summary>
        <div className="setup-section-content">
          <div className="split">
            <label className="field">Dataset FPS<select value={config.fps} onChange={e => set('fps', Number(e.target.value))}><option>10</option><option>15</option><option>20</option><option>24</option><option>30</option></select></label>
            <label className="field">Collection mode
              <select value={config.mode} onChange={e => set('mode', e.target.value as Config['mode'])}>
                {modes.map(mode => <option key={mode.value} value={mode.value}>{mode.label}</option>)}
              </select>
            </label>
          </div>
          <div className="split">
            <label className="field">Target replay pace
              <select value={config.speed} onChange={e => set('speed', Number(e.target.value))} aria-describedby="replay-timing-hint">
                {[0.25, 0.5, 0.75, 1, 1.25, 1.5, 2].map(n => <option key={n} value={n}>{n === 1 ? '1× · Teaching pace' : `${n}×`}</option>)}
              </select>
            </label>
            <label className="field">Replay speed and acceleration limits
              <select value={config.replay_profile} onChange={e => set('replay_profile', e.target.value as Config['replay_profile'])} aria-describedby="replay-timing-hint">
                {Object.entries(replayProfileLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
              </select>
            </label>
          </div>
          <p className="hint" id="replay-timing-hint">Replay can run slower than the target pace to stay within these limits.</p>
          {config.device === 'so101' && <>
            <label className="field">Calibration directory<input value={config.calibration_dir ?? ''} onChange={e => set('calibration_dir', e.target.value || null)} placeholder="Default LeRobot directory" /></label>
            <label className="check"><input type="checkbox" checked={config.automatic_reset} onChange={e => set('automatic_reset', e.target.checked)} /> Allow automatic arm reset</label>
          </>}
        </div>
      </details>
      <div className="setup-actions">
        <button className="button primary" type="submit" disabled={!available}><Radio size={16} /> {config.device === 'simulation' ? 'Connect simulator' : 'Connect SO-101'}</button>
      </div>
    </form>
  </section>
}

export function Telemetry({ state }: { state: State | null }) {
  const resetTarget = state && ['ready', 'teach_paused', 'reset', 'reset_released', 'returning'].includes(state.phase)
    ? state.reset_target : null
  return <section className="panel">
    <div className="panel-heading"><h2>Joint readings</h2></div>
    <div className="monitor-content">
      <div className="joint-list">
        {joints.map(joint => {
          const value = state?.pose?.[joint]
          const target = resetTarget?.[joint]
          const unit = joint === 'gripper' ? '%' : '°'
          const [min, max] = state?.limits?.[joint] ?? [-100, 100]
          const ratio = value === undefined ? 0 : Math.min(1, Math.max(0, (value - min) / (max - min)))
          return <div className="joint" key={joint}>
            <div><span>{jointLabels[joint]}</span><output>{value === undefined ? '—' : `${value.toFixed(1)}${unit}`}</output></div>
            <div className="joint-track"><span style={{ width: `${ratio * 100}%` }} /></div>
            {target !== undefined && <span className="joint-target">Target {target.toFixed(1)}{unit}</span>}
          </div>
        })}
      </div>
      {resetTarget && <div className="pose-status"><Crosshair size={16} /><strong>{state?.at_start ? 'At start pose' : 'Away from start pose'}</strong></div>}
    </div>
  </section>
}

function useCameraFrame(name: string, active: boolean, live: boolean, review?: number) {
  const [frame, setFrame] = useState<string>()
  useEffect(() => {
    setFrame(undefined)
    if (!active) return
    const abort = new AbortController()
    let timer = 0
    let objectUrl: string | undefined
    const pull = async () => {
      try {
        const query = review === undefined ? '' : `?review=${review}`
        const response = await fetch(`/api/preview/${encodeURIComponent(name)}${query}`, {
          signal: abort.signal, cache: 'no-store',
        })
        if (response.ok) {
          const blob = await response.blob()
          if (abort.signal.aborted) return
          if (objectUrl) URL.revokeObjectURL(objectUrl)
          objectUrl = URL.createObjectURL(blob)
          setFrame(objectUrl)
        } else {
          setFrame(undefined)
        }
      } catch {
        if (!abort.signal.aborted) setFrame(undefined)
      } finally {
        if (live && !abort.signal.aborted) timer = window.setTimeout(pull, 190)
      }
    }
    void pull()
    return () => {
      abort.abort()
      window.clearTimeout(timer)
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [name, active, live, review])
  return frame
}

function CameraTile({ spec, state, demo, online, review, datasetFps }: {
  spec: CameraSpec
  state: State | null
  demo: boolean
  online: boolean
  review?: number
  datasetFps: number
}) {
  const status = state?.cameras[spec.name]
  const savedPreview = review !== undefined
  const isLive = Boolean(online && state?.connected && !savedPreview)
  const frame = useCameraFrame(spec.name, online && (isLive || savedPreview), isLive, review)
  const ready = isLive && status?.ready
  const image = isLive && !ready ? undefined : frame
  const badge = savedPreview ? 'TAKE' : demo && isLive ? 'SIM' :
    ready ? 'LIVE' : isLive && status?.retrying ? 'RECONNECTING' :
    isLive && !status?.problem ? 'CONNECTING' : 'OFFLINE'
  return <figure className="camera-tile">
    <div className="feed">
      {image ? <img src={image} draggable="false" alt={`${humanize(spec.name)} ${savedPreview ? 'recorded' : demo ? 'simulated' : 'live'} view`} /> :
        <div className="empty-camera"><Camera size={24} /><span>{savedPreview ? online ? 'Waiting for preview' : 'Workspace offline' : status?.problem || (isLive ? 'Waiting for frames' : 'Connect camera')}</span></div>}
      <div className="feed-top">
        <span className="feed-id"><Video size={13} /> {humanize(spec.name)}</span>
        <span className={`feed-label ${savedPreview ? 'captured' : ready ? 'active' : ''}`}><i />{badge}</span>
      </div>
      {image && <div className="viewfinder"><i /><i /><i /><i /></div>}
    </div>
    <figcaption>
      <span>{status?.mode?.label && !savedPreview && <>{status.mode.label} <b>·</b> </>}
        {status?.mode?.width ?? spec.width} × {status?.mode?.height ?? spec.height} <b>·</b>{' '}
        {savedPreview ? `${datasetFps} FPS` : status?.fps ? `${status.fps} measured FPS` : `${spec.capture_fps} FPS requested`}</span>
      <span className={spec.record ? 'record-source' : ''}>{spec.record ? <><span className="tiny-dot" /> {savedPreview ? 'Recorded' : 'Record'}</> : <><Eye size={12} /> Preview only</>}</span>
    </figcaption>
  </figure>
}

export function CameraPanel({ config, state, online, reviewIndex, setReviewIndex }: {
  config: Config, state: State | null, online: boolean, reviewIndex: number, setReviewIndex: (index: number) => void
}) {
  const reviewing = state?.phase === 'review'
  const reviewFrames = state?.quality?.review_frames ?? 0
  const specs = reviewing ? config.cameras.filter(spec => spec.record) : config.cameras
  if (reviewing && !reviewFrames) return null
  const title = reviewing ? 'Recorded views' : 'Cameras'
  return <section className="panel camera-panel" aria-label={title}>
    <div className="panel-heading">
      <h2>{title}</h2>
    </div>
    {reviewing && reviewFrames > 1 && <label className="scrub">
      <span className="scrub-label"><Film size={15} /> Preview frame</span>
      <input aria-label="Review preview frames" aria-valuetext={`${reviewIndex + 1} of ${reviewFrames}`} type="range" value={reviewIndex} min={0} max={reviewFrames - 1} onChange={event => setReviewIndex(Number(event.target.value))} />
      <span>{reviewIndex + 1} / {reviewFrames}</span>
    </label>}
    {specs.length ? <div className="camera-grid">
      {specs.map(spec => <CameraTile key={spec.name} spec={spec} state={state} demo={config.device === 'simulation'} online={online} review={reviewing ? reviewIndex : undefined} datasetFps={config.fps} />)}
    </div> : <div className="empty-collection"><EyeOff size={27} /><p>No cameras selected</p><span>Episodes will capture joint telemetry.</span></div>}
  </section>
}

export function Timeline({ state }: { state: State | null }) {
  const chart = state?.path?.chart
  const tracks = [
    { index: 0, color: '#ff9a69', name: 'Shoulder' },
    { index: 2, color: '#88adcf', name: 'Elbow' },
    { index: 5, color: '#bd9bdb', name: 'Gripper' },
  ]
  const all = chart?.flat()
  const low = Math.min(-20, ...(all ?? []))
  const high = Math.max(90, ...(all ?? []))
  const duration = state?.path?.duration ?? 6
  return <section className="panel timeline-panel">
    <div className="panel-heading">
      <div className="heading-icon"><Activity size={16} /></div>
      <div><h2>Trajectory</h2><p>{state?.phase === 'teaching' ? 'Sampling raw joint positions' : state?.path ? `${elapsed(state.path.teaching_duration)} taught · ${state.path.keyframes} keyframes` : 'Joint positions over time'}</p></div>
      <span className="meta-pill push"><span className={chart ? 'warm' : ''}>{chart ? `${elapsed(duration)} planned` : 'No path taught'}</span></span>
    </div>
    <div className="timeline-content">
      <div className="timeline-chart">
        <svg viewBox="0 0 720 86" preserveAspectRatio="none" aria-label={chart ? 'Taught joint curves' : 'Empty trajectory plot'} role="img">
          {[0, 1, 2, 3, 4, 5, 6].map(i => <line key={i} x1={i * 120} x2={i * 120} y1={0} y2={86} />)}
          {[0, 1, 2, 3].map(i => <line key={i} y1={i * 28.5} y2={i * 28.5} x1={0} x2={720} />)}
          {chart && tracks.map(({ index, color }) => <polyline key={index} fill="none" stroke={color} strokeWidth={1.7} vectorEffect="non-scaling-stroke"
            points={chart.map((row, i) => `${i * 720 / (chart.length - 1)},${81 - (row[index] - low) / (high - low) * 74}`).join(' ')} />)}
          {chart && state?.phase === 'replay' && <line x1={state.progress * 720} x2={state.progress * 720} y1={0} y2={86} className="playhead" />}
        </svg>
        {!chart && <div className="chart-empty"><Activity size={15} /><span>{state?.phase === 'teaching' ? `Teaching · ${elapsed(state.elapsed)}` : 'Teach or load a path to see the joint traces'}</span></div>}
      </div>
      <div className="chart-axis">{[0, 1, 2, 3, 4, 5, 6].map(i => <span key={i}>{(duration * i / 6).toFixed(i && chart ? 1 : 0)}s</span>)}</div>
      <div className="chart-legend">{tracks.map(track => <span key={track.index}><i style={{ background: track.color }} />{track.name}</span>)}</div>
    </div>
  </section>
}

export function ReviewMetrics({ state }: { state: State }) {
  const quality = state.quality
  if (!quality) return null
  const cameraCount = state.config.cameras.filter(camera => camera.record).length
  const timing = quality.replay_timing
  return <div className="review-info">
    <div className="quality-grid">
      <div><span>Frames at {state.config.fps} FPS</span><strong>{quality.frames}</strong></div>
      <div><span>Dataset duration</span><strong>{quality.seconds.toFixed(2)} s</strong></div>
    </div>
    {!!quality.discontinuities?.length && <p className="hint">This take has camera gaps. Exclude it from training unless your loader respects segment boundaries.</p>}
    <details className="episode-options">
      <summary><ChevronDown size={14} /> Capture and tracking details</summary>
      <div>
        <div className="quality-grid">
          {state.path && <>
            <div><span>Teaching time</span><strong>{state.path.teaching_duration.toFixed(2)} s</strong></div>
            <div><span>Planned path time</span><strong>{state.path.duration.toFixed(2)} s</strong></div>
          </>}
          {quality.max_arm_path_error_deg != null && <div><span>Peak arm path error{quality.arm_path_peak && ` · ${jointLabels[quality.arm_path_peak.joint]} at ${quality.arm_path_peak.seconds.toFixed(2)} s`}</span><strong>{quality.max_arm_path_error_deg}°</strong></div>}
          {quality.max_gripper_path_error_pct != null && <div><span>Peak gripper path error</span><strong>{quality.max_gripper_path_error_pct}%</strong></div>}
          {cameraCount > 0 && quality.skipped_camera_samples != null && <div><span>Skipped camera samples</span><strong>{quality.skipped_camera_samples}</strong></div>}
          {quality.capture_span_seconds != null && <div><span>First to last saved frame</span><strong>{quality.capture_span_seconds.toFixed(2)} s</strong></div>}
          {timing && <>
            <div><span>Alignment</span><strong>{timing.alignment_seconds.toFixed(2)} s</strong></div>
            <div><span>Path traversal</span><strong>{timing.motion_seconds.toFixed(2)} s</strong></div>
            <div><span>Endpoint settling</span><strong>{timing.settle_seconds.toFixed(2)} s</strong></div>
          </>}
          <div><span>Max frame gap</span><strong>{quality.max_gap_ms} ms</strong></div>
          {cameraCount > 1 && <div><span>Max camera-to-camera skew</span><strong>{quality.camera_skew_ms} ms</strong></div>}
          {cameraCount > 0 && <div><span>Max camera-to-encoder offset</span><strong>{quality.sensor_offset_ms} ms</strong></div>}
          <div><span>Peak arm command-to-encoder gap</span><strong>{quality.max_arm_error_deg}°</strong></div>
          <div><span>Peak gripper command-to-encoder gap</span><strong>{quality.max_gripper_error_pct}%</strong></div>
        </div>
        <p className="hint">The nominal frame gap is {Number((1000 / state.config.fps).toFixed(1))} ms. {cameraCount > 0 && 'Timing uses camera presentation timestamps on macOS and host read times elsewhere. Neither is a hardware exposure measurement.'}</p>
        {Object.entries(quality.camera_offset_ms ?? {}).map(([name, offsets]) => <p key={name} className="hint">{humanize(name)}: median {offsets.p50} ms relative to the encoder read midpoint (p1 {offsets.p1} ms, p95 {offsets.p95} ms, p99 {offsets.p99} ms).</p>)}
        {quality.arm_path_peak?.excursion_ms != null && <p className="hint">Around the peak, {jointLabels[quality.arm_path_peak.joint]} was at least {quality.arm_path_peak.excursion_deg}° off the reference for {quality.arm_path_peak.excursion_ms > 0 ? `${quality.arm_path_peak.excursion_ms} ms` : 'one control tick'}.</p>}
        {quality.max_arm_path_error_deg != null && <p className="hint">Path error compares encoder readings with the replay reference at the same sample time. Servo commands can aim ahead of that reference.</p>}
      </div>
    </details>
  </div>
}

export function SavedPaths({ state, send, available }: { state: State, send: Send, available: boolean }) {
  const [picked, setPicked] = useState('')
  if (!state.paths.length) return null
  return <div className="saved-paths">
    <select aria-label="Saved trajectory" value={picked} onChange={e => setPicked(e.target.value)}>
      <option value="">Saved trajectories</option>
      {state.paths.map(p => <option key={p.id} value={p.id}>{p.task}{p.grip_edited ? ' · Grip edited' : ''} · {p.duration.toFixed(1)}s · {new Date(p.created).toLocaleString(undefined, { dateStyle: 'short', timeStyle: 'short' })}</option>)}
    </select>
    <button type="button" className="button" disabled={!available || !picked} onClick={() => send('load_path', { path_id: picked })}><FolderOpen size={15} /> Load path</button>
  </div>
}

interface JogInput {
  joint: Joint
  value: -1 | 1
}

interface JogButtons {
  active: Partial<Record<Joint, number>>
  start: (input: JogInput) => void
  release: () => void
}

export function KeyboardLegend({ upcoming = false, buttons }: { upcoming?: boolean; buttons?: JogButtons }) {
  return <div className="keyboard-legend" aria-label="Keyboard shortcuts">
    <p>{upcoming && 'Controls unlock after replay. '}Hold <kbd>Space</kbd> + a shortcut to move. Release to hold position.</p>
    <dl>{joints.map(joint => <div key={joint}>
      <dt>{jointLabels[joint]}</dt>
      {jogKeys[joint].map(({ code, label, value }) => <dd key={code}>
        {buttons ? <button type="button" className={`jog-shortcut ${buttons.active[joint] === value ? 'held' : ''}`}
          aria-label={`${jointLabels[joint]} ${label.toLowerCase()}`}
          onPointerDown={event => {
            if (event.button !== 0) return
            event.preventDefault()
            event.currentTarget.setPointerCapture(event.pointerId)
            buttons.start({ joint, value })
          }}
          onPointerUp={buttons.release} onPointerCancel={buttons.release} onLostPointerCapture={buttons.release}
          onKeyDown={event => {
            if ((event.key === ' ' || event.key === 'Enter') && !event.repeat) {
              event.preventDefault()
              buttons.start({ joint, value })
            }
          }}
          onKeyUp={event => {
            if (event.key === ' ' || event.key === 'Enter') buttons.release()
          }}
          onBlur={buttons.release}
        ><kbd>{keyLabel(code)}</kbd><span>{label}</span></button>
        : <span className="jog-shortcut"><kbd>{keyLabel(code)}</kbd><span>{label}</span></span>}
      </dd>)}
    </div>)}</dl>
    {buttons && <small>Hold an on-screen button to jog. <kbd>Esc</kbd> stops motion.</small>}
  </div>
}

export function ManualControls({ send }: { send: Send }) {
  const panel = useRef<HTMLDivElement>(null)
  const input = useRef<JogInput | null>(null)
  const pressed = useRef(new Set<string>())
  const active = useRef<Partial<Record<Joint, number>>>({})
  const [highlight, setHighlight] = useState(active.current)

  const transmit = () => {
    const current = input.current
    const keys = pressed.current
    const axes: Partial<Record<Joint, number>> = current ? { [current.joint]: current.value } : {}
    if (!current && keys.has('Space')) {
      for (const joint of joints) {
        const direction = jogKeys[joint].reduce((sum, key) => sum + (keys.has(key.code) ? key.value : 0), 0)
        if (direction) axes[joint] = direction
      }
    }
    active.current = axes
    setHighlight(axes)
    send('jog', { axes })
  }

  useEffect(() => {
    panel.current?.focus({ preventScroll: true })
    const keys = pressed.current
    const keyCodes = new Set(['Space', ...Object.values(jogKeys).flat().map(key => key.code)])
    const tick = window.setInterval(() => {
      if (input.current || keys.has('Space')) send('jog', { axes: active.current })
    }, 80)
    const release = () => {
      input.current = null
      keys.clear()
      transmit()
    }
    const keyDown = (event: KeyboardEvent) => {
      const target = event.target
      if (!keyCodes.has(event.code) ||
          target instanceof HTMLElement && target.closest('input, button, select, textarea, summary, a, [contenteditable]') ||
          event.metaKey || event.ctrlKey || event.altKey || event.shiftKey) return
      if (event.code === 'Space' || keys.has('Space')) event.preventDefault()
      if (!keys.has(event.code) && !event.repeat) {
        keys.add(event.code)
        if (keys.has('Space')) transmit()
      }
    }
    const keyUp = (event: KeyboardEvent) => {
      if (keys.delete(event.code) && (event.code === 'Space' || keys.has('Space'))) {
        event.preventDefault()
        transmit()
      }
    }
    const visibility = () => { if (document.hidden) release() }
    window.addEventListener('blur', release)
    window.addEventListener('keydown', keyDown)
    window.addEventListener('keyup', keyUp)
    document.addEventListener('visibilitychange', visibility)
    return () => {
      window.clearInterval(tick)
      window.removeEventListener('blur', release)
      window.removeEventListener('keydown', keyDown)
      window.removeEventListener('keyup', keyUp)
      document.removeEventListener('visibilitychange', visibility)
    }
  }, [send])

  const start = (next: JogInput) => {
    pressed.current.clear()
    input.current = next
    transmit()
  }
  const release = () => {
    if (!input.current) return
    input.current = null
    transmit()
  }

  return <div className="manual-controls" ref={panel} tabIndex={-1} aria-label="Manual controls">
    <KeyboardLegend buttons={{ active: highlight, start, release }} />
  </div>
}

export function VirtualJoints({ state, send }: { state: State, send: Send }) {
  return <div className="virtual-grid">
    {joints.map(joint => <label key={joint}>{jointLabels[joint]}<input type="range" aria-label={jointLabels[joint]} min={state.limits?.[joint][0]} max={state.limits?.[joint][1]} value={state.pose?.[joint] ?? 0}
      onChange={e => send('demo_move', { axes: { [joint]: Number(e.target.value) } })} /></label>)}
  </div>
}
