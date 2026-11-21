import '@fontsource-variable/inter/wght.css'
import '@fontsource/ibm-plex-mono/latin-400.css'
import {
  ArrowDownToLine, ArrowDownToDot, ArrowRight, BadgeCheck, Check, ChevronRight, CircleHelp,
  ClipboardCheck, Crosshair, Database, Hand, Link2, Play, PlugZap, Radio, RotateCcw, Save,
  Square, SquareActivity, TriangleAlert, Video, X, Zap, ZapOff,
} from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { GripPanel } from './grip'
import {
  CameraPanel, KeyboardLegend, ManualControls, ReviewControls, SavedPaths, SetupPanel, Telemetry, Timeline,
  VirtualJoints,
} from './panels'
import { useWorkspace } from './socket'
import { defaults, elapsed, jointLabels } from './types'
import type { Config, Phase, Send, State } from './types'
import './styles.css'

const steps = [
  { title: 'Setup', detail: 'Connect your devices', icon: Link2 },
  { title: 'Teach', detail: 'Guide the arm by hand', icon: Hand },
  { title: 'Reset', detail: 'Restore the start scene', icon: RotateCcw },
  { title: 'Replay', detail: 'Capture a clean take', icon: Play },
  { title: 'Review', detail: 'Save or retry the take', icon: ClipboardCheck },
]

const stepFor: Record<Phase, number> = {
  setup: 0, ready: 1, teaching: 1, teach_paused: 1, reset: 2, reset_released: 2, returning: 2,
  replay: 3, manual: 3, processing: 3, review: 4, saving: 4, next: 4, finished: 5, fault: -1,
}

const activePhases: Phase[] = ['teaching', 'reset_released', 'returning', 'replay', 'manual']
type Confirm = { title: string; text: string; label: string; submit: () => void }

function Header({ state, online, send }: { state: State | null; online: boolean; send: Send }) {
  const connected = online && state?.connected
  const torque = online ? state?.torque : 'unknown'
  const phase = state?.phase ?? 'setup'
  const canStop = activePhases.includes(phase) || state?.operation === 'prepare'
  const recording = state?.recording
  useEffect(() => {
    const emergency = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && canStop && !document.querySelector('dialog[open]')) send('stop')
    }
    window.addEventListener('keydown', emergency)
    return () => window.removeEventListener('keydown', emergency)
  }, [canStop, send])
  return <header className="topbar">
    <a className="brand" href="/" aria-label="GhostPath home">
      <img src="/ghostpath.svg" alt="" /><span>Ghost<span className="brand-accent">Path</span></span>
    </a>
    <div className="top-context"><span className="top-divider" /><span>Workspace</span><ChevronRight size={13} /><strong>Collection</strong></div>
    <div className="top-status">
      <span className="status-chip"><i className={connected ? 'green' : ''} />{connected ? state.config.device === 'simulation' ? 'Simulator connected' : 'SO-101 connected' : online ? 'Robot offline' : 'Link offline'}</span>
      <span className={`status-chip torque-${torque ?? 'unknown'}`}>{torque === 'on' ? <Zap size={13} /> : <ZapOff size={13} />} Torque {torque === 'on' ? 'on' : torque === 'off' ? 'off' : torque === 'mixed' ? 'mixed' : '—'}</span>
      {recording && <span className="status-chip taking"><i /> Recording</span>}
      <button className="button emergency" disabled={!canStop || !online} onClick={() => send('stop')} title="Stop motion and discard the active take (Esc)"><Square size={13} fill="currentColor" />Stop motion</button>
    </div>
  </header>
}

function Sidebar({ state, online }: { state: State | null; online: boolean }) {
  const step = stepFor[state?.phase ?? 'setup']
  return <nav className="sidebar" aria-label="Episode workflow">
    <div className="sidebar-main">
      <div className="nav-heading">WORKFLOW</div>
      <ol className="steps">{steps.map(({ title, detail, icon: Icon }, index) => {
        const current = index === step
        const done = index < step
        return <li key={title} className={current ? 'current' : done ? 'done' : ''} aria-current={current ? 'step' : undefined}>
          <span className="step-icon">{done ? <Check size={16} /> : <Icon size={16} />}</span>
          <div><strong>{title}</strong><span>{detail}</span></div>
          {current && <span className="step-indicator" />}
        </li>
      })}</ol>
      <div className="sidebar-session">
        <div className="session-mark"><Database size={16} /></div><div><strong>Dataset session</strong><span>{state?.accepted ?? 0} saved {state?.accepted === 1 ? 'episode' : 'episodes'}</span></div>
      </div>
    </div>
    <div className="sidebar-bottom">
      <div className="local-status"><i className={online ? 'green' : ''} /><div><strong>Local workspace</strong><span>{online ? 'Control link online' : 'Waiting for backend'}</span></div></div>
      <a className="help-link" href="https://huggingface.co/docs/lerobot/v0.7.0/so101" target="_blank" rel="noreferrer"><CircleHelp size={15} /> SO-101 setup guide</a>
    </div>
  </nav>
}

function Confirmation({ value, close }: { value: Confirm; close: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null)
  useEffect(() => { dialog.current?.showModal() }, [])
  return <dialog ref={dialog} className="confirmation" onCancel={close}>
    <div className="confirmation-top"><div className="heading-icon"><TriangleAlert size={18} /></div><button className="icon-button bare push" aria-label="Close" onClick={close}><X size={19} /></button></div>
    <h2>{value.title}</h2><p>{value.text}</p>
    <div className="actions"><button type="button" className="button" onClick={close} autoFocus>Cancel</button><button type="button" className="button primary" onClick={() => { close(); value.submit() }}>{value.label}</button></div>
  </dialog>
}

function Workflow({ state, config, send, available, reviewIndex, setReviewIndex }: {
  state: State | null; config: Config; send: Send; available: boolean;
  reviewIndex: number; setReviewIndex: (index: number) => void
}) {
  const [confirm, setConfirm] = useState<Confirm | null>(null)
  const [manualFinish, setManualFinish] = useState(config.mode === 'hybrid')
  const [gripAssist, setGripAssist] = useState(false)
  useEffect(() => setManualFinish(config.mode === 'hybrid'), [config.mode])
  const phase = state?.phase ?? 'setup'
  const simulated = config.device === 'simulation'
  const atStart = state?.at_start
  const hasStart = Boolean(state?.start_pose)
  const canReturn = available && (simulated || state?.torque === 'on')
  const canManagePower = Boolean(state?.connected && ['ready', 'teach_paused', 'reset', 'review', 'next', 'finished'].includes(phase))
  const instruct = (action: string, title: string, text: string, label: string, extra = {}) => {
    const submit = () => send(action, { confirmed: true, ...extra })
    if (simulated) submit()
    else setConfirm({ title, text, label, submit })
  }
  const release = () => instruct('release_reset', 'Release the arm for reset?', 'Support the arm and anything it is holding. Every joint, including the gripper, will lose torque.', 'Release torque')
  const teaching = () => state?.torque === 'off'
    ? send('teach')
    : instruct('teach', 'Start teaching?', 'Support the arm and gripper. Teaching releases torque so you can guide the arm by hand.', 'Start teaching')
  const continueTeaching = () => instruct(
    'continue_teach', 'Continue this path?', 'Support the arm and gripper before releasing torque. Continue from the current endpoint.', 'Continue teaching',
  )
  const startReplay = () => instruct(
    'replay', 'Replay and record this take?', 'Verify the scene is reset and clear the arm’s full path and camera views.', 'Replay & record',
    { manual_finish: manualFinish },
  )
  const autoReset = () => instruct(
    'auto_reset',
    'Return to the start pose?',
    'Empty the gripper and clear people, objects, and cables from the full return path. The arm will move directly to the start pose and settle there. Restore the objects by hand afterward.',
    'Confirm return',
    { allow_auto_reset_once: !config.automatic_reset },
  )
  const returnButton = <button className={`button ${simulated || config.automatic_reset ? 'primary' : ''}`}
    disabled={!canReturn} onClick={autoReset}><RotateCcw size={15} /> Return to start</button>
  const torqueOn = state?.torque === 'on'
  const changePower = () => torqueOn
    ? instruct('release_torque', 'Turn torque off?', 'Support the arm and empty the gripper first. The arm can drop when its motors release.', 'Turn torque off')
    : instruct('hold_torque', 'Turn torque on?', 'Hold the arm still and supported while the motors take over at the measured pose.', 'Turn torque on')
  const reverseReset = () => instruct(
    'reverse_reset',
    'Reset the scene with Boomerang?',
    state?.at_path_end
      ? 'Verify the object is at its taught final placement and clear the reverse route. Boomerang retraces the taught arm path.'
      : 'Verify the object is at its taught final placement and the gripper is empty. Clear both the route to the taught endpoint and the reverse route.',
    'Start Boomerang',
    { allow_auto_reset_once: true, grip_assist: gripAssist },
  )

  let title = ''
  let detail = ''
  let controls = null
  let extra = null

  switch (phase) {
    case 'setup':
      title = 'Connect your workspace'
      detail = 'Start with the simulator or select your calibrated SO-101 in session setup.'
      break
    case 'ready': {
      const savedPoses = state?.start_poses ?? []
      title = !hasStart ? 'Choose a start pose' : atStart ? 'Ready to teach' : 'Move to the selected start'
      detail = !hasStart
        ? 'Position the arm by hand and use its current position, or select a saved pose.'
        : atStart ? 'Start teaching from the selected pose. Reset will return the arm here.'
        : canReturn ? 'Return to start to get ready to teach.'
        : 'Position the arm by hand, or turn torque on for an automatic return.'
      extra = <>
        <div className="start-picker">
          <button className="button" disabled={!available} onClick={() => send('capture_start')}><Crosshair size={15} /> Use current position</button>
          <label className="field">Saved start poses
            <select value={state?.start_pose_id ?? ''} disabled={!available || !savedPoses.length}
              onChange={event => { if (event.target.value) send('select_start', { pose_id: event.target.value }) }}>
              <option value="">{savedPoses.length ? 'Choose a saved pose' : 'No saved poses yet'}</option>
              {savedPoses.map(saved => <option value={saved.id} key={saved.id}>{saved.label} · {new Date(saved.created).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })}</option>)}
            </select>
          </label>
        </div>
        {simulated && state?.torque === 'off' && <details className="virtual"><summary>Position the simulated arm</summary><VirtualJoints state={state} send={send} /></details>}
        {state && state.paths.length > 0 && <details className="virtual"><summary>Load a taught path</summary><SavedPaths state={state} send={send} available={available} /></details>}
        {!!state?.accepted && <button className="text-button" disabled={!available} onClick={() => send('finish')}><Database size={13} /> Finish session</button>}
      </>
      controls = <>
        {hasStart && !atStart && returnButton}
        {simulated && hasStart && atStart && <button className="button" disabled={!available} onClick={() => send('demo_path')}><Play size={15} /> Example path</button>}
        <button className="button primary" disabled={!available || !hasStart || !atStart} onClick={teaching}><Hand size={15} /> Start teaching</button>
      </>
      break
    }
    case 'teaching':
      title = 'Teaching in progress'
      detail = `Guide the arm through the task. Hold it still when you stop. · ${elapsed(state?.elapsed ?? 0)} taught`
      controls = <button className="button primary" disabled={!available} onClick={() => send('stop_teach')}><Square size={14} fill="currentColor" /> Stop teaching</button>
      extra = simulated && state && <details className="virtual" open><summary>Virtual joint controls</summary><VirtualJoints state={state} send={send} /></details>
      break
    case 'reset_released':
      title = 'Position the arm by hand'
      detail = atStart ? 'Start pose matched. Hold the arm still before re-enabling torque.' : 'Match the saved start pose, then hold still and confirm.'
      controls = <button className="button primary" disabled={!available} onClick={() => send('hold_reset')}><Zap size={15} /> Hold & verify</button>
      extra = simulated && state && <details className="virtual" open><summary>Virtual joint controls</summary><VirtualJoints state={state} send={send} /></details>
      break
    case 'teach_paused':
    case 'reset': {
      const paused = phase === 'teach_paused'
      const hasPath = Boolean(state?.path)
      title = paused ? 'Teaching paused' : atStart ? 'Ready to replay' : 'Reset the arm and scene'
      detail = paused
        ? hasPath ? 'Keep teaching from this endpoint, or use Boomerang to reverse the task back to the start.' : state?.notice || 'Continue teaching to create a playable path.'
        : state?.scene_reversed && atStart ? state.notice
        : atStart ? 'Verify the scene and clear the camera views before recording.'
        : 'Use Boomerang from the taught endpoint, or return to start and reset the scene by hand.'
      controls = <>
        {paused && <button className="button" disabled={!available || !state?.can_continue} onClick={continueTeaching}><Hand size={15} /> Continue teaching</button>}
        {paused && !hasPath && <button className="text-button" disabled={!available} onClick={() => send('discard_teaching')}>Start over</button>}
        {hasPath && (!atStart || !canReturn) && <>
          <button className="button" disabled={!available} onClick={release}><Hand size={15} /> Manual reset</button>
          {!atStart && returnButton}
        </>}
        {hasPath && !state?.scene_reversed && <button className={`button ${paused && !atStart ? 'primary' : ''}`}
          disabled={!canReturn || !(state?.at_path_end || state?.at_start)} onClick={reverseReset}><RotateCcw size={15} /> Boomerang scene reset</button>}
        {hasPath && <button className={`button ${atStart ? 'primary' : ''}`} disabled={!canReturn || !atStart} onClick={startReplay}><Play size={15} fill="currentColor" /> Replay & record</button>}
      </>
      extra = <>
        {paused && !state?.can_continue && <p className="hint">Continuing requires the arm at the paused endpoint and teaching time remaining.</p>}
        {state?.notice.startsWith('Recording stopped before') && <p className="hint">{state.notice}</p>}
        {hasPath && !state?.scene_reversed && <label className="manual-option">
          <input type="checkbox" checked={gripAssist} disabled={!available} onChange={event => setGripAssist(event.target.checked)} />
          <span><strong>Optional grip adjustment during Boomerang</strong><small>Hold C to close more or O to open more. A completed reset saves your adjustments for replay.</small></span>
        </label>}
        {state?.grip?.saved && <GripPanel grip={state.grip} duration={state.path?.duration ?? 0} send={send} />}
        {hasPath && <label className="manual-option">
          <input type="checkbox" checked={manualFinish} disabled={!available} onChange={event => setManualFinish(event.target.checked)} />
          <span><strong>Keyboard finish after replay</strong><small>Keep recording and take over after the ghost path settles.</small></span>
        </label>}
        {!!state?.accepted && <button className="text-button" disabled={!available} onClick={() => send('finish')}><Database size={13} /> Finish session</button>}
      </>
      break
    }
    case 'returning':
    case 'replay':
      title = phase === 'returning'
        ? state?.reset_action === 'reverse' ? 'Boomerang is reversing the taught path'
          : state?.reset_action === 'reverse_approach' ? 'Boomerang is approaching the taught endpoint'
          : state?.reset_action === 'park' ? 'Returning to rest' : 'Returning to start'
        : 'Replaying & recording'
      detail = state?.tracking_joint ? `Slowing for ${jointLabels[state.tracking_joint].toLowerCase()} to catch up.`
        : phase === 'returning' ? state?.reset_action === 'reverse'
          ? 'Verify the scene before recording.'
          : 'Keep the route clear until the arm settles. Recording starts after you verify the scene.'
        : `${state?.frames ?? 0} frames captured · ${elapsed(state?.elapsed ?? 0)} elapsed`
      extra = <>
        <div className="motion-progress">
          <div><strong>{phase === 'replay' ? 'Ghost path progress' : 'Reset progress'}</strong>
            <span>{state?.replay_stage === 'aligning' ? 'Aligning · recording' : state?.replay_stage === 'settling' ? '100% · settling' : `${Math.floor((state?.progress ?? 0) * 100)}%`}</span></div>
          <progress max={1} value={state?.progress ?? 0} aria-label={phase === 'replay' ? 'Ghost path progress' : 'Reset progress'} />
        </div>
        {state?.grip && (state.grip.enabled || phase === 'replay') &&
          <GripPanel grip={state.grip} duration={state.path?.duration ?? 0} send={send} />}
        {phase === 'replay' && state?.manual_finish && <KeyboardLegend upcoming />}
      </>
      if (phase === 'replay') controls = <button className="button primary" disabled={!available} onClick={() => send('stop_recording')}><Square size={14} /> Stop recording</button>
      break
    case 'manual':
      title = 'Keyboard teleoperation'
      detail = 'Ghost path complete. Take over with the keyboard while recording continues.'
      controls = <button className="button primary" disabled={!available} onClick={() => send('stop_recording')}><Square size={14} /> Finish recording</button>
      extra = <ManualControls send={send} />
      break
    case 'processing':
    case 'saving':
      title = state?.operation === 'prepare' ? 'Starting recording' : state?.operation === 'seal' ? 'Validating the take' : state?.operation === 'save' ? 'Saving the episode' : state?.operation === 'finish' ? 'Finalizing your dataset' : 'Updating the session'
      detail = state?.notice || 'Please wait for the dataset operation to complete.'
      extra = <div className="processing-bar"><i /></div>
      break
    case 'review':
      title = 'Review this take'
      detail = state?.stopped_early ? 'Recording stopped early. Review the captured portion before saving.'
        : 'Capture passed validation. Check the recorded views before saving.'
      controls = <>
        <button className="button quiet-danger" disabled={!available} onClick={() => send('discard')}><X size={15} /> Discard</button>
        <button className="button" disabled={!available} onClick={() => send('retry')}><RotateCcw size={15} /> Retry</button>
        <button className="button primary" disabled={!available} onClick={() => send('save')}><Save size={15} /> Save episode</button>
      </>
      extra = state && <ReviewControls state={state} reviewIndex={reviewIndex} change={setReviewIndex} />
      break
    case 'next':
      title = state?.quality ? 'Episode saved' : 'Take discarded'
      detail = `${state?.accepted ?? 0} accepted. Start another episode or finalize the dataset.`
      controls = <>
        {!!state?.accepted && <button className="button" disabled={!available} onClick={() => send('finish')}><Database size={15} /> Finish session</button>}
        <button className="button primary" disabled={!available} onClick={() => send('next')}><ArrowRight size={15} /> Next episode</button>
      </>
      extra = <>
        {config.mode !== 'repeat' && state?.path && <button className="text-button" disabled={!available} onClick={() => send('next', { reuse: true })}><RotateCcw size={13} /> Reuse this path</button>}
        {config.mode === 'repeat' && <button className="text-button" disabled={!available} onClick={() => send('next', { reuse: false })}><Hand size={13} /> Teach a new path</button>}
      </>
      break
    case 'finished':
      title = 'Dataset ready'
      detail = `${state?.accepted ?? 0} saved episodes · LeRobotDataset v3.0`
      controls = <button className="button primary" disabled={!available} onClick={() => send('disconnect')}><ArrowRight size={15} /> New session</button>
      extra = <div className="export-path"><ArrowDownToLine size={15} /><code>{state?.dataset_path}</code>
        <button className="button" onClick={async e => { const button = e.currentTarget; try { await navigator.clipboard.writeText(state?.dataset_path || ''); button.textContent = 'Copied' } catch { button.textContent = 'Select path to copy' } }}>Copy</button></div>
      break
    case 'fault':
      title = 'Workspace stopped'
      detail = state?.fault || 'Check the arm before resuming.'
      controls = <>
        <button className="button" disabled={!available || Boolean(state?.operation)} onClick={() => send('disconnect')}><Radio size={15} /> Disconnect</button>
        {!!state?.accepted && <button className="button" disabled={!available || Boolean(state?.operation)} onClick={() => send('finish')}>Finish session</button>}
        <button className="button primary" disabled={!available || Boolean(state?.operation)} onClick={() => instruct('recover', 'Recover control?', 'Support the arm and clear the workspace. If torque is mixed, recovering will seed fresh measured goals before re-enabling it.', 'Recover')}><RotateCcw size={15} /> Recover</button>
      </>
      break
  }

  return <section className={`panel workflow-card ${phase === 'fault' ? 'fault-card' : ''}`} aria-label="Current workflow step">
    <div className="workflow-row">
      <div className="heading-icon">{phase === 'fault' ? <TriangleAlert size={17} /> : phase === 'finished' ? <BadgeCheck size={18} /> : phase === 'teaching' ? <Hand size={17} /> : <ArrowDownToDot size={17} />}</div>
      <div className="workflow-copy"><h2>{title}</h2><p>{detail}</p></div>
    </div>
    {(controls || extra) && <div className="workflow-content">{extra}{controls && <div className="actions">{controls}</div>}</div>}
    {canManagePower && <div className="workflow-content idle-controls">
      <div className="actions">
        <button className="button" disabled={!available} onClick={changePower}>{torqueOn ? <ZapOff size={15} /> : <Zap size={15} />}{torqueOn ? phase === 'ready' ? 'Release arm to position' : 'Turn torque off' : 'Turn torque on'}</button>
      </div>
    </div>}
    {confirm && <Confirmation value={confirm} close={() => setConfirm(null)} />}
  </section>
}

function App() {
  const { state, online, send, error, setError, pending } = useWorkspace()
  const [config, setConfig] = useState<Config>(defaults)
  const [takeFrame, setTakeFrame] = useState(0)
  const phase = state?.phase ?? 'setup'
  const setupDataset = state?.phase === 'setup' ? state.config.dataset : undefined
  useEffect(() => {
    if (setupDataset) setConfig(previous => ({ ...previous, dataset: setupDataset }))
  }, [setupDataset])
  const activeConfig = phase === 'setup' ? config : state?.config ?? config
  const available = online && !pending
  const canDisconnect = available && ['ready', 'teach_paused', 'reset', 'next', 'finished', 'fault'].includes(phase)
  useEffect(() => {
    if (phase !== 'review') setTakeFrame(0)
  }, [phase])

  const workflow = <Workflow state={state} config={activeConfig} send={send} available={available} reviewIndex={takeFrame} setReviewIndex={setTakeFrame} />

  return <div className="shell">
    <Header state={state} online={online} send={send} />
    <Sidebar state={state} online={online} />
    <main className="main">
      <div className="workspace-head">
        <div className="workspace-title">
          <div className="eyebrow"><SquareActivity size={14} /> DATA COLLECTION</div>
          <h1>Collection workspace</h1>
          <p>Teach by hand. Replay without a trace.</p>
        </div>
        <div className="episode-counter"><Video size={16} /><div><span>{phase === 'finished' ? 'SESSION COMPLETE' : 'CURRENT EPISODE'}</span><strong>{String(phase === 'finished' ? state?.accepted ?? 0 : state?.episode ?? 1).padStart(2, '0')}</strong></div></div>
      </div>
      {error && <div className="alert" role="alert"><TriangleAlert size={17} /><span>{error}</span><button className="icon-button bare push" aria-label="Dismiss error" onClick={() => setError('')}><X size={17} /></button></div>}
      <div className="workspace-grid">
        <div className="main-stack">
          {phase !== 'setup' && workflow}
          <CameraPanel config={activeConfig} state={state} online={online} reviewIndex={takeFrame} />
          <Timeline state={state} />
          {phase === 'setup' && workflow}
        </div>
        {phase === 'setup' ? <SetupPanel config={config} change={setConfig} available={available} connect={() => send('connect', { config })} /> :
          <Telemetry state={state} config={activeConfig} disconnect={() => send('disconnect')} available={canDisconnect} />}
      </div>
      <footer className="workspace-footer"><span><PlugZap size={13} /> Local recording</span><span>GhostPath <i>·</i> LeRobot v3</span></footer>
    </main>
  </div>
}

createRoot(document.getElementById('root')!).render(<App />)
