import {
  Activity, Camera, ChevronDown, ChevronRight, Cpu, Crosshair, Eye, EyeOff, Film, FolderOpen,
  Layers3, Monitor, Plus, Radio, ScanLine, SlidersHorizontal, Trash2, Video,
} from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { jogKeys, keyLabel } from './controls'
import { camera, elapsed, humanize, jointLabels, joints } from './types'
import type { CameraSpec, Config, Joint, Send, State } from './types'

const modes = [
  { value: 'teach', label: 'Teach every episode', detail: 'Guide a new path for each take.' },
  { value: 'repeat', label: 'Repeat path', detail: 'Reuse the trajectory across takes.' },
  { value: 'hybrid', label: 'Hybrid / manual finish', detail: 'Add powered jogs after replay.' },
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
  const [expanded, setExpanded] = useState<number | null>(null)
  const [advanced, setAdvanced] = useState(false)
  const set = <K extends keyof Config>(key: K, value: Config[K]) => change({ ...config, [key]: value })
  const updateCamera = (index: number, patch: Partial<CameraSpec>) => {
    set('cameras', config.cameras.map((item, i) => i === index ? { ...item, ...patch } : item))
  }
  const scanPorts = async () => {
    setScan('Scanning…')
    try {
      const response = await fetch('/api/devices')
      if (!response.ok) throw Error()
      const result: Devices = await response.json()
      setDevices(result)
      setScan(result.ports.length ? '' : 'No USB serial device found.')
      if (result.ports.length === 1) set('port', result.ports[0].port)
    } catch {
      setScan('Device scan unavailable.')
    }
  }

  return <aside className="inspector panel setup-panel">
    <div className="panel-heading">
      <div className="heading-icon"><SlidersHorizontal size={16} /></div>
      <div><h2>Session setup</h2><p>Robot, task and capture preferences</p></div>
    </div>
    <form onSubmit={event => { event.preventDefault(); connect() }} className="setup-form">
      <div className="form-section">
        <label className="field">Robot source</label>
        <div className="segmented" aria-label="Robot source">
          <button type="button" className={config.device === 'simulation' ? 'selected' : ''} onClick={() => set('device', 'simulation')}><Monitor size={14} /> Simulator</button>
          <button type="button" className={config.device === 'so101' ? 'selected' : ''} onClick={() => change({ ...config, device: 'so101', fps: config.device === 'simulation' && config.fps === 30 ? 20 : config.fps })}><Cpu size={14} /> SO-101</button>
        </div>
        {config.device === 'so101' && <div className="hardware-fields">
          <label className="field">Serial port
            <div className="row">
              <select value={config.port} onChange={e => set('port', e.target.value)} required>
                <option value="">Select a device</option>
                {config.port && !devices.ports.some(p => p.port === config.port) && <option value={config.port}>{config.port}</option>}
                {devices.ports.map(p => <option key={p.port} value={p.port}>{p.port}</option>)}
              </select>
              <button type="button" className="button icon-button" onClick={scanPorts} aria-label="Scan serial ports" title="Scan serial ports"><ScanLine size={16} /></button>
            </div>
          </label>
          {scan && <p className="hint">{scan}</p>}
          <label className="field">LeRobot calibration ID
            <input value={config.robot_id} onChange={e => set('robot_id', e.target.value)} list="calibrations" required />
            <datalist id="calibrations">{devices.calibrations.map(id => <option key={id} value={id} />)}</datalist>
          </label>
        </div>}
        <label className="field">Task description
          <textarea value={config.task} maxLength={500} rows={2} onChange={event => set('task', event.target.value)} required />
        </label>
        <label className="field">Collection mode
          <select value={config.mode} onChange={e => set('mode', e.target.value as Config['mode'])}>
            {modes.map(mode => <option key={mode.value} value={mode.value}>{mode.label}</option>)}
          </select>
          <span className="hint">{modes.find(mode => mode.value === config.mode)?.detail}</span>
        </label>
      </div>
      <div className="form-section">
        <div className="section-title"><Video size={15} /><h3>Camera sources</h3>
          <button className="icon-button bare" type="button" aria-label="Add camera" title="Add camera" disabled={config.cameras.length >= 4} onClick={() => {
            const source = [0, 1, 2, 3, 4, 5, 6, 7].find(i =>
              !config.cameras.some(item => String(item.source) === String(i) || item.name === `camera_${i}`))
            if (source !== undefined) {
              setExpanded(config.cameras.length)
              set('cameras', [...config.cameras, camera(`camera_${source}`, source)])
            }
          }}><Plus size={16} /></button>
        </div>
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
                <label className="field">Source<input value={spec.source} onChange={e => updateCamera(index, { source: e.target.value })} required /></label>
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
      </div>
      <div className="form-section setup-bottom">
        <div className="split">
          <label className="field">Dataset FPS<select value={config.fps} onChange={e => set('fps', Number(e.target.value))}><option>10</option><option>15</option><option>20</option><option>24</option><option>30</option></select></label>
          <label className="field">Replay speed<select value={config.speed} onChange={e => set('speed', Number(e.target.value))}>
            {[0.25, 0.5, 0.75, 1, 1.25, 1.5, 2].map(n => <option key={n} value={n}>{n === 1 ? '1.0' : n}×</option>)}
          </select></label>
        </div>
        <p className="hint">Start at 20 FPS for this arm. Use 30 FPS when all recorded cameras can sustain it.</p>
        <button type="button" className="text-button preferences-button" aria-expanded={advanced} onClick={() => setAdvanced(!advanced)}><ChevronRight size={13} className={advanced ? 'rotate-right' : ''} /> More preferences</button>
        {advanced && <div className="advanced-fields">
          <label className="field">Dataset name<input value={config.dataset} onChange={e => set('dataset', e.target.value)} pattern={String.raw`[a-z0-9][a-z0-9_\-]{0,63}`} required /></label>
          {config.device === 'so101' && <>
            <label className="field">Calibration directory<input value={config.calibration_dir ?? ''} onChange={e => set('calibration_dir', e.target.value || null)} placeholder="Default LeRobot directory" /></label>
            <label className="check"><input type="checkbox" checked={config.automatic_reset} onChange={e => set('automatic_reset', e.target.checked)} /> Allow automatic arm reset</label>
          </>}
        </div>}
        <button className="button primary full" type="submit" disabled={!available}>
          <Radio size={16} /> {config.device === 'simulation' ? 'Connect simulator' : 'Connect SO-101'}<ChevronRight size={16} className="push" />
        </button>
      </div>
    </form>
  </aside>
}

export function Telemetry({ state, config, disconnect, available }: {
  state: State | null
  config: Config
  disconnect: () => void
  available: boolean
}) {
  const [view, setView] = useState<'joints' | 'settings'>('joints')
  const resetTarget = state && ['ready', 'teach_paused', 'reset', 'reset_released', 'returning'].includes(state.phase)
    ? state.reset_target : null
  return <aside className="inspector panel">
    <div className="panel-heading">
      <div className="heading-icon"><Activity size={16} /></div>
      <div><h2>Device monitor</h2><p>Robot feedback and capture setup</p></div>
    </div>
    <div className="monitor-content">
      <div className="segmented">
        <button type="button" className={view === 'joints' ? 'selected' : ''} onClick={() => setView('joints')}><Activity size={14} /> Joints</button>
        <button type="button" className={view === 'settings' ? 'selected' : ''} onClick={() => setView('settings')}><SlidersHorizontal size={14} /> Session</button>
      </div>
      {view === 'joints' ? <>
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
        <div className="pose-status"><Crosshair size={16} /><div><strong>{state?.at_start ? 'At start pose' : state?.reset_target ? 'Away from start' : 'Choose a start pose'}</strong><p>{state?.reset_target ? 'Compared with the target joint positions' : 'Use the position controls in the workflow'}</p></div></div>
      </> : <>
        <dl className="session-specs">
          <dt>Robot</dt><dd>{config.device === 'simulation' ? 'Simulator' : config.robot_id}</dd>
          <dt>Mode</dt><dd>{modes.find(mode => mode.value === config.mode)?.label}</dd>
          <dt>Task</dt><dd>{config.task}</dd>
          <dt>Dataset</dt><dd>{config.dataset}</dd>
          <dt>Capture</dt><dd>{config.fps} FPS · {config.speed}× replay</dd>
          <dt>Recorded</dt><dd>{config.cameras.filter(c => c.record).map(c => humanize(c.name)).join(', ') || 'Joints only'}</dd>
        </dl>
        <button className="button full" disabled={!available} onClick={disconnect}><Radio size={14} /> Disconnect robot</button>
      </>}
      <div className="monitor-foot"><Layers3 size={14} /> LeRobotDataset <span>v3.0</span></div>
    </div>
  </aside>
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

function CameraTile({ spec, state, demo, online, review }: {
  spec: CameraSpec
  state: State | null
  demo: boolean
  online: boolean
  review?: number
}) {
  const status = state?.cameras[spec.name]
  const savedPreview = review !== undefined && spec.record
  const isLive = Boolean(online && state?.connected && review === undefined)
  const visible = isLive || savedPreview || demo && review === undefined
  const frame = useCameraFrame(spec.name, visible && online, isLive, savedPreview ? review : undefined)
  const ready = isLive && status?.ready
  const image = isLive && !ready ? undefined : frame
  const badge = savedPreview ? 'TAKE' : review !== undefined ? 'PREVIEW ONLY' : demo ? 'SIM' :
    ready ? 'LIVE' : isLive && status?.retrying ? 'RECONNECTING' :
    isLive && !status?.problem ? 'CONNECTING' : 'OFFLINE'
  return <figure className="camera-tile">
    <div className="feed">
      {image ? <img src={image} draggable="false" alt={`${humanize(spec.name)} ${savedPreview ? 'recorded' : demo ? 'simulated' : 'live'} view`} /> :
        <div className="empty-camera"><Camera size={24} /><span>{review !== undefined ? 'Preview only' : status?.problem || (isLive ? 'Waiting for frames' : 'Connect camera')}</span></div>}
      <div className="feed-top">
        <span className="feed-id"><Video size={13} /> {humanize(spec.name)}</span>
        <span className={`feed-label ${savedPreview ? 'captured' : ready ? 'active' : ''}`}><i />{badge}</span>
      </div>
      {image && <div className="viewfinder"><i /><i /><i /><i /></div>}
    </div>
    <figcaption>
      <span>{spec.width} × {spec.height} <b>·</b> {status?.fps || spec.capture_fps} FPS</span>
      <span className={spec.record ? 'record-source' : ''}>{spec.record ? <><span className="tiny-dot" /> Record</> : <><Eye size={12} /> Preview only</>}</span>
    </figcaption>
  </figure>
}

export function CameraPanel({ config, state, online, reviewIndex }: {
  config: Config, state: State | null, online: boolean, reviewIndex: number
}) {
  const review = state?.phase === 'review' ? reviewIndex : undefined
  const count = config.cameras.filter(c => c.record).length
  return <section className="panel camera-panel" aria-label="Camera monitor">
    <div className="panel-heading">
      <div className="heading-icon"><Video size={16} /></div>
      <div><h2>Camera monitor</h2><p>{review === undefined ? 'Preview your capture sources' : 'Review staged camera frames'}</p></div>
      <span className="meta-pill push"><i className={count ? 'warm' : ''} />{count} selected</span>
    </div>
    {config.cameras.length ? <div className="camera-grid">
      {config.cameras.map(spec => <CameraTile key={spec.name} spec={spec} state={state} demo={config.device === 'simulation'} online={online} review={review} />)}
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
      <div><h2>Trajectory</h2><p>{state?.phase === 'teaching' ? 'Sampling raw joint positions' : state?.path ? `${state.path.keyframes} keyframes · bounded replay` : 'Joint positions over time'}</p></div>
      <span className="meta-pill push"><span className={chart ? 'warm' : ''}>{chart ? elapsed(duration) : 'No path taught'}</span></span>
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

export function ReviewControls({ state, reviewIndex, change }: {
  state: State, reviewIndex: number, change: (index: number) => void
}) {
  const quality = state.quality
  if (!quality) return null
  return <div className="review-info">
    <div className="quality-grid">
      <div><span>Frames</span><strong>{quality.frames}</strong></div>
      <div><span>Duration</span><strong>{quality.seconds.toFixed(2)} s</strong></div>
      <div><span>Max frame gap</span><strong>{quality.max_gap_ms} ms</strong></div>
      {quality.review_frames > 0 && <div><span>Camera skew</span><strong>{quality.camera_skew_ms} ms</strong></div>}
      <div><span>Peak arm command-to-encoder gap</span><strong>{quality.max_arm_error_deg}°</strong></div>
      <div><span>Peak gripper command-to-encoder gap</span><strong>{quality.max_gripper_error_pct}%</strong></div>
    </div>
    {quality.review_frames > 1 && <label className="scrub"><Film size={15} /><input aria-label="Review captured frames" type="range" value={reviewIndex} min={0} max={quality.review_frames - 1} onChange={e => change(Number(e.target.value))} /><span>{reviewIndex + 1}/{quality.review_frames}</span></label>}
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
