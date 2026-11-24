import '@fontsource-variable/inter/wght.css'
import '@fontsource/ibm-plex-mono/latin-400.css'
import {
  ArrowDownToLine, ArrowRight, Check, ChevronDown, CircleHelp, ClipboardCheck, Crosshair,
  Camera, Database, Hand, Play, Radio, RotateCcw, Save, Square, TriangleAlert, X, Zap, ZapOff,
} from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { GripPanel } from './grip'
import {
  CameraPanel, ManualControls, ReviewMetrics, SavedPaths, SetupPanel, Telemetry, Timeline,
  VirtualJoints,
} from './panels'
import { useWorkspace } from './socket'
import { defaults, elapsed, humanize, jointLabels, replayProfileLabels } from './types'
import type { Config, Phase, Send, State } from './types'
import './styles.css'

const steps = [
  { title: 'Teach', icon: Hand },
  { title: 'Reset', icon: RotateCcw },
  { title: 'Record', icon: Play },
  { title: 'Review', icon: ClipboardCheck },
]

const stepFor: Record<Phase, number> = {
  setup: -1, ready: 0, practice: 0, teaching: 0, teach_paused: 0, reset: 1, reset_released: 1, returning: 1,
  replay: 2, manual: 2, processing: 2, review: 3, saving: 3, next: 3, finished: 4, fault: -1,
}

const activePhases: Phase[] = ['practice', 'teaching', 'reset_released', 'returning', 'replay', 'manual']
type Confirm = { title: string; text: string; label: string; submit: () => void }

function Header({ state, online, available, send }: { state: State | null; online: boolean; available: boolean; send: Send }) {
  const connected = online && state?.connected
  const torque = state?.torque ?? 'unknown'
  const phase = state?.phase ?? 'setup'
  const canStop = activePhases.includes(phase) || state?.operation === 'prepare'
  const canFinish = Boolean(state?.accepted) && ['ready', 'teach_paused', 'reset', 'next', 'fault'].includes(phase)
  const canDisconnect = ['ready', 'teach_paused', 'reset', 'next', 'finished', 'fault'].includes(phase) &&
    (!state?.accepted || phase === 'finished' || phase === 'fault')
  useEffect(() => {
    const emergency = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && canStop) send('stop')
    }
    window.addEventListener('keydown', emergency)
    return () => window.removeEventListener('keydown', emergency)
  }, [canStop, send])
  return <header className="topbar">
    <div className="topbar-inner">
      <div className="brand"><img src="/ghostpath.svg" alt="" /><span>Ghost<span className="brand-accent">Path</span></span></div>
      <div className="top-status">
        {(!online || connected) && <span className="status-chip" title={connected ? 'Robot connected' : 'Control link offline'}><i className={connected ? 'green' : ''} />{connected ? state?.config.device === 'simulation' ? 'Simulator' : 'SO-101' : 'Offline'}</span>}
        {connected && <span className={`status-chip torque-${torque}`}>Torque {torque === 'unknown' ? '—' : torque}</span>}
        {canStop && <button className="button emergency" disabled={!online} onClick={() => send('stop')} title="Emergency stop (Esc)"><Square size={12} fill="currentColor" />Stop motion</button>}
        <details className="session-menu">
          <summary className="button">{connected ? 'Session' : 'Help'} <ChevronDown size={13} /></summary>
          <div className="session-menu-content" onClick={event => {
            if (event.target instanceof Element && event.target.closest('button, a')) {
              event.currentTarget.parentElement?.removeAttribute('open')
            }
          }}>
            {connected && <>
              <button type="button" disabled={!available || !canFinish || Boolean(state?.operation)} onClick={() => send('finish')}><Database size={14} /> Finish dataset</button>
              <button type="button" disabled={!available || !canDisconnect || Boolean(state?.operation)} onClick={() => send('disconnect')}><Radio size={14} /> Disconnect</button>
            </>}
            <a href="https://huggingface.co/docs/lerobot/so101" target="_blank" rel="noreferrer"><CircleHelp size={14} /> SO-101 setup guide</a>
          </div>
        </details>
      </div>
    </div>
  </header>
}

function EpisodeHeader({ state, config }: { state: State; config: Config }) {
  const step = stepFor[state.phase]
  const saved = state.phase === 'next' && !!state.quality
  const complete = state.phase === 'finished'
  const preparing = state.phase === 'next' && !saved
  const reviewing = state.phase === 'review'
  return <>
    <div className="episode-heading">
      <div className="dataset-name"><Database size={14} /><span>Dataset</span><strong>{config.dataset}</strong></div>
      <div className="episode-title">
        <h1>{complete ? 'Dataset complete' : `Episode ${saved ? state.accepted : state.episode}`}</h1>
        {!complete && <span className={`episode-status ${state.recording ? 'taking' : saved ? 'saved' : ''}`}>{state.recording ? 'Recording' : saved ? 'Saved' : reviewing ? state.stopped_early ? 'Partial take' : 'In review' : preparing ? 'Up next' : 'Draft'}</span>}
        <span className="saved-count">{state.accepted} saved</span>
      </div>
      <p>{config.task}</p>
    </div>
    <nav className="episode-steps" aria-label="Episode workflow">
      <ol>{steps.map(({ title, icon: Icon }, index) => <li key={title}
        className={index === step ? 'current' : index < step ? 'done' : ''}
        aria-current={index === step ? 'step' : undefined}>
        {index < step ? <Check size={15} /> : <Icon size={15} />}<span>{title}</span>
      </li>)}</ol>
    </nav>
  </>
}

function Confirmation({ value, close }: { value: Confirm; close: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null)
  useEffect(() => { dialog.current?.showModal() }, [])
  return <dialog ref={dialog} className="confirmation" aria-labelledby="confirmation-title" onCancel={close}>
    <div className="confirmation-top"><div className="heading-icon"><TriangleAlert size={18} /></div><button className="icon-button bare push" aria-label="Close" onClick={close}><X size={19} /></button></div>
    <h2 id="confirmation-title">{value.title}</h2><p>{value.text}</p>
    <div className="actions"><button type="button" className="button" onClick={close} autoFocus>Cancel</button><button type="button" className="button primary" onClick={() => { close(); value.submit() }}>{value.label}</button></div>
  </dialog>
}

function DatasetOutput({ path }: { path: string }) {
  const [feedback, setFeedback] = useState('')
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(path)
      setFeedback('Dataset path copied.')
    } catch {
      setFeedback('Could not copy automatically. Select the path to copy it.')
    }
  }
  return <div>
    <div className="export-path"><ArrowDownToLine size={15} /><code>{path}</code>
      <button className="button" onClick={copy}>Copy path</button>
    </div>
    <p className="hint">Keep ghostpath.json with the dataset. It records policy inputs, timing, and each episode’s start and rest positions.</p>
    {feedback && <p className="hint" role="status">{feedback}</p>}
  </div>
}

function Workflow({ state, config, send, online, available }: {
  state: State | null; config: Config; send: Send; online: boolean; available: boolean
}) {
  const [confirm, setConfirm] = useState<Confirm | null>(null)
  const [poseLabel, setPoseLabel] = useState('')
  const [manualFinish, setManualFinish] = useState(config.mode === 'hybrid')
  const [firmGripChoice, setFirmCarryGrip] = useState<boolean | null>(null)
  const [gripAssist, setGripAssist] = useState(false)
  useEffect(() => setManualFinish(config.mode === 'hybrid'), [config.mode])
  const phase = state?.phase ?? 'setup'
  const replayKeyboard = phase === 'replay' && Boolean(state?.manual_finish)
  const firmCarryGrip = firmGripChoice ?? state?.firm_carry_grip ?? false
  const simulated = config.device === 'simulation'
  const atStart = state?.at_start
  const hasStart = Boolean(state?.start_pose)
  const motionPowered = simulated || state?.torque === 'on'
  const canReturn = available && motionPowered
  const canManageRest = Boolean(state?.connected && !state.fault && ['ready', 'reset', 'next', 'finished'].includes(phase))
  const canManagePower = Boolean(state?.connected && !state.fault && ['ready', 'teach_paused', 'reset', 'review', 'next', 'finished'].includes(phase))
  const canDiscard = phase === 'teaching' || phase === 'teach_paused' ||
    Boolean(state?.path && ['reset', 'reset_released', 'returning'].includes(phase))
  const discardEpisode = (action: 'discard_episode' | 'discard' | 'retry' = 'discard_episode') => {
    const label = action === 'retry' ? 'Discard & retry' : action === 'discard' ? 'Discard take' : 'Discard episode'
    setConfirm({
      title: `${label}?`,
      text: action === 'retry'
        ? 'This take will be deleted. You can reset the scene and record the same path again.'
        : 'This ends the current attempt. Saved episodes and reusable paths stay in the library.',
      label,
      submit: () => send(action, { confirmed: true }),
    })
  }
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
    { manual_finish: manualFinish, firm_carry_grip: firmCarryGrip },
  )
  const autoReset = () => instruct(
    'auto_reset',
    'Return to the start pose?',
    'Empty the gripper and clear people, objects, and cables from the full return path. The arm will move directly to the start pose and settle there. Restore the objects by hand afterward.',
    'Confirm return',
    { allow_auto_reset_once: !config.automatic_reset },
  )
  const returnButton = (primary: boolean) => <button className={`button ${primary ? 'primary' : ''}`}
    disabled={!canReturn} onClick={autoReset}><RotateCcw size={15} /> Return to start</button>
  const park = () => instruct(
    'park', 'Return to the saved rest pose?',
    'Empty the gripper and clear the entire route. The arm will move directly to the rest pose, not the episode start.',
    'Return to rest', { allow_auto_reset_once: !config.automatic_reset },
  )
  const startPractice = () => instruct(
    'practice', 'Try the manual controls?',
    'Clear the arm’s workspace. If torque is off, support the arm while the motors engage. Nothing is recorded in practice.',
    'Start practice',
  )
  const torqueOn = state?.torque === 'on'
  const changePower = () => torqueOn
    ? instruct('release_torque', 'Turn torque off?', 'Support the arm and empty the gripper first. The arm can drop when its motors release.', 'Turn torque off')
    : instruct('hold_torque', 'Turn torque on?', 'Hold the arm still and supported while the motors take over at the measured pose.', 'Turn torque on')
  const reverseReset = () => instruct(
    'reverse_reset',
    'Reset the scene with Boomerang?',
    (state?.at_path_end
      ? 'Verify the object is at its taught final placement and clear the reverse route. Boomerang retraces the taught arm path.'
      : 'Verify the object is at its taught final placement and the gripper is empty. Clear both the route to the taught endpoint and the reverse route.')
      + (firmCarryGrip ? ' Firm grip will add closure during detected carries.' : ''),
    'Start Boomerang',
    { allow_auto_reset_once: true, firm_carry_grip: firmCarryGrip, grip_assist: gripAssist },
  )

  let title = ''
  let detail = ''
  let controls = null
  let extra = null

  switch (phase) {
    case 'ready': {
      const savedPoses = state?.start_poses ?? []
      const selected = savedPoses.find(pose => pose.id === state?.start_pose_id)
      const earlier = savedPoses.filter(pose => pose.legacy)
      const poseOption = (pose: typeof savedPoses[number]) => <option value={pose.id} key={pose.id}>{pose.label} · {new Date(pose.created).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })}</option>
      title = !hasStart ? 'Choose a start pose' : atStart ? 'Ready to teach' : 'Move to the selected start'
      detail = !hasStart
        ? simulated ? 'Use the current position to set a start pose for the example.' : 'Position the arm by hand and use its current position, or select a saved pose.'
        : atStart ? simulated ? 'Try the example path, or teach your own with the virtual joint controls.' : 'Start teaching from the selected pose. Reset will return the arm here.'
        : motionPowered ? 'Return to start to get ready to teach.'
        : 'Position the arm by hand, or turn torque on for an automatic return.'
      extra = <>
        {state?.notice.match(/^(Episode discarded\.|Start pose saved\.|Saved start deleted\.|Practice ended\.|At the saved rest position\.)/) && <p className="hint" role="status">{state.notice}</p>}
        <div className="start-picker">
          <button className={`button ${hasStart ? '' : 'primary'}`} disabled={!available} onClick={() => { setPoseLabel(''); send('capture_start') }}><Crosshair size={15} /> Use current position</button>
          <label className="field">Saved start poses
            <select value={state?.start_pose_id ?? ''} disabled={!available || !savedPoses.length}
              onChange={event => { if (event.target.value) send('select_start', { pose_id: event.target.value }) }}>
              <option value="">{savedPoses.length ? 'Choose a saved pose' : 'No saved poses yet'}</option>
              {savedPoses.filter(pose => !pose.legacy).map(poseOption)}
              {earlier.length > 0 && <optgroup label="Earlier captures">{earlier.map(poseOption)}</optgroup>}
            </select>
          </label>
        </div>
        {hasStart && (!selected || selected.legacy) && <details className="episode-options">
          <summary><ChevronDown size={14} /> Save this start for later</summary>
          <div>
            <p className="hint">Using the current position does not save it. Name this selected start if you want to reuse it in another session.</p>
            <form className="start-picker" onSubmit={event => { event.preventDefault(); send('save_start', { pose_label: poseLabel.trim() }) }}>
              <button className="button" disabled={!available || !poseLabel.trim()}><Save size={15} /> Save start</button>
              <label className="field">Start name<input value={poseLabel} onChange={event => setPoseLabel(event.target.value)} maxLength={60} /></label>
            </form>
          </div>
        </details>}
        {selected && <div><button className="text-button" disabled={!available} onClick={() => setConfirm({
          title: 'Delete this saved start?', text: 'This removes the library entry. Your current task start, paths, and datasets are unchanged.',
          label: 'Delete saved start', submit: () => send('delete_start', { pose_id: selected.id, confirmed: true }),
        })}><X size={14} /> Delete saved start</button></div>}
        {simulated && state?.torque === 'off' && <details className="virtual"><summary>Position the simulated arm</summary><VirtualJoints state={state} send={send} /></details>}
        {state && state.paths.length > 0 && <details className="virtual"><summary>Load a taught path</summary><SavedPaths state={state} send={send} available={available} /></details>}
      </>
      controls = <>
        {hasStart && !atStart && returnButton(motionPowered)}
        {hasStart && atStart && <button className={`button ${simulated ? '' : 'primary'}`} disabled={!available} onClick={teaching}><Hand size={15} /> Start teaching</button>}
        {simulated && atStart && <button className="button primary" disabled={!available} onClick={() => send('demo_path')}><Play size={15} /> Example path</button>}
      </>
      break
    }
    case 'practice':
      title = 'Practice controls'
      detail = 'Try moving the arm before teaching. Nothing is recorded.'
      controls = <>
        <button className="button primary" disabled={!available} onClick={() => send('finish_practice')}><Check size={15} /> Done</button>
        {state?.rest_target && <button className="button" disabled={!canReturn} onClick={park}><RotateCcw size={15} /> Return to rest</button>}
      </>
      extra = <>
        {state && <ManualControls state={state} send={send} enabled={available && !confirm} />}
        {!state?.rest_target && <p className="hint">To add a parking position, choose Done, then save the current position under Rest position.</p>}
      </>
      break
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
      const canBoomerang = !state?.scene_reversed && (state?.at_path_end || atStart)
      const waitingCameras = config.cameras.filter(spec => {
        const status = state?.cameras[spec.name]
        return spec.record && (!status?.ready || !status.fps)
      })
      const camerasReady = waitingCameras.length === 0
      title = paused ? 'Teaching paused' : !atStart ? 'Reset the arm and scene'
        : !motionPowered ? 'Hold the arm before replay' : !camerasReady ? 'Waiting for cameras' : 'Ready to replay'
      detail = !hasPath ? state?.notice || 'Continue teaching to create a playable path.'
        : !motionPowered ? 'Turn on torque before a powered move or reset the arm by hand.'
        : atStart ? state?.scene_reversed ? state.notice : 'Verify the scene and clear the camera views before recording.'
        : canBoomerang ? 'Use Boomerang for a reversible task, or reset the arm and objects by hand.'
        : 'Return to start and restore the objects before recording.'
      controls = <>
        {paused && <button className="button" disabled={!available || !state?.can_continue} onClick={continueTeaching}><Hand size={15} /> Continue teaching</button>}
        {hasPath && (!atStart || !motionPowered) && <>
          <button className="button" disabled={!available} onClick={release}><Hand size={15} /> Manual reset</button>
          {!atStart && returnButton(motionPowered && !canBoomerang)}
        </>}
        {hasPath && canBoomerang && <button className={`button ${motionPowered && !atStart ? 'primary' : ''}`}
          disabled={!canReturn} onClick={reverseReset}><RotateCcw size={15} /> Boomerang scene reset</button>}
        {hasPath && atStart && <button className="button primary" disabled={!canReturn || !camerasReady} onClick={startReplay}><Play size={15} fill="currentColor" /> Replay & record</button>}
      </>
      extra = <>
        {hasPath && atStart && !camerasReady && <p className="capture-wait" role="status"><Camera size={15} /> Waiting for recording cameras: {waitingCameras.map(spec => humanize(spec.name)).join(', ')}. Check the camera monitor below.</p>}
        {paused && !state?.can_continue && <p className="hint">Continuing requires the arm at the paused endpoint and teaching time remaining.</p>}
        {state?.path && state.path.replay_profile !== state.config.replay_profile &&
          <p className="hint">This path uses the {replayProfileLabels[state.path.replay_profile].toLowerCase()} saved with its grip edits.</p>}
        {state?.notice.startsWith('Recording stopped before') && <p className="hint">{state.notice}</p>}
        {state?.notice.startsWith('Recording could not start') && <p className="hint" role="status">{state.notice}</p>}
        {hasPath && <details className="episode-options">
          <summary><ChevronDown size={14} /> Episode options {firmCarryGrip && <span>Firm grip on</span>}</summary>
          <div>
            <label className="manual-option">
              <input type="checkbox" checked={firmCarryGrip} disabled={!available} onChange={event => setFirmCarryGrip(event.target.checked)} />
              <span><strong>Firm grip</strong><small>Adds closure during carries in Boomerang and replay.</small></span>
            </label>
            {!state?.scene_reversed && <label className="manual-option">
              <input type="checkbox" checked={gripAssist} disabled={!available} onChange={event => setGripAssist(event.target.checked)} />
              <span><strong>Keyboard grip adjustment</strong><small>During Boomerang, hold C to close or O to open.</small></span>
            </label>}
            <label className="manual-option">
              <input type="checkbox" checked={manualFinish} disabled={!available} onChange={event => setManualFinish(event.target.checked)} />
              <span><strong>Manual finish</strong><small>Take over with Space + a direction key during replay, or start manual control when it ends.</small></span>
            </label>
            {state?.grip?.saved && <GripPanel grip={state.grip} duration={state.path?.duration ?? 0} send={send} />}
          </div>
        </details>}
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
          ? state.notice
          : 'Keep the route clear until the arm settles. Recording starts after you verify the scene.'
        : `${state?.frames ?? 0} frames · ${elapsed(state?.elapsed ?? 0)}${state?.replay_firm_carries ? ` · Firm grip on ${state.replay_firm_carries} ${state.replay_firm_carries === 1 ? 'carry' : 'carries'}` : state?.replay_firm_carries === 0 ? ' · No carry matched for firm grip' : ''}`
      extra = <>
        <div className="motion-progress">
          <div><strong>{phase === 'replay' ? 'Ghost path progress' : 'Reset progress'}</strong>
            <span>{state?.replay_stage === 'aligning' ? 'Aligning · recording' : state?.replay_stage === 'settling' ? '100% · settling' : `${Math.floor((state?.progress ?? 0) * 100)}%`}</span></div>
          <progress max={1} value={state?.progress ?? 0} aria-label={phase === 'replay' ? 'Ghost path progress' : 'Reset progress'} />
        </div>
        {state?.grip && (state.grip.enabled || phase === 'replay') &&
          <GripPanel grip={state.grip} duration={state.path?.duration ?? 0} send={send} />}
        {phase === 'replay' && !replayKeyboard && <p className="hint">Taking over ends replay, not the recording.</p>}
      </>
      if (phase === 'replay') controls = <>
        <button className="button primary" disabled={!available} onClick={() => send('take_over', { confirmed: true })}><Hand size={15} /> Take over</button>
        <button className="button" disabled={!available} onClick={() => send('stop_recording')}><Square size={14} /> Stop recording</button>
      </>
      break
    case 'manual':
      title = 'Manual control · recording'
      detail = state?.notice ?? 'Recording continues under manual control.'
      controls = <button className="button primary" disabled={!available} onClick={() => send('finish_manual')}><Square size={14} /> Finish recording</button>
      break
    case 'processing':
    case 'saving':
      title = state?.operation === 'prepare' ? 'Starting recording' : state?.operation === 'seal' ? 'Validating the take' : state?.operation === 'save' ? 'Saving the episode' : state?.operation === 'finish' ? 'Finalizing your dataset' : 'Updating the session'
      detail = state?.notice || 'Please wait for the dataset operation to complete.'
      extra = <div className="processing-bar"><i /></div>
      break
    case 'review': {
      const hasViews = Boolean(state?.quality?.review_frames)
      title = 'Review this take'
      detail = state?.fault ? 'The arm stopped after recording. You can still review and save this take.'
        : state?.stopped_early ? 'Recording stopped early. Review the captured portion before saving.'
        : hasViews ? 'Check the recorded views below before saving this take.' : 'This take contains joint data only. Check the capture and tracking details before saving.'
      controls = <>
        <button className="button quiet-danger" disabled={!available} onClick={() => discardEpisode('discard')}><X size={15} /> Discard take</button>
        <button className="button" disabled={!available || Boolean(state?.fault)} onClick={() => discardEpisode('retry')}><RotateCcw size={15} /> Retry this path</button>
        <button className="button primary" disabled={!available} onClick={() => send('save')}><Save size={15} /> Save episode</button>
      </>
      extra = state && <>
        {state.fault && <p className="hint" role="alert">{state.fault}</p>}
        <ReviewMetrics state={state} />
      </>
      break
    }
    case 'next':
      title = state?.quality ? 'Episode saved' : 'Take discarded'
      detail = state?.accepted ? 'Collect another take or finish the dataset on this computer.' : 'Nothing has been saved yet. Reuse this path or teach a new one.'
      controls = <>
        {Boolean(state?.accepted) && <button className="button" disabled={!available} onClick={() => send('finish')}><Database size={15} /> Finish dataset</button>}
        {state?.path && <button className={`button ${config.mode === 'repeat' ? 'primary' : ''}`} disabled={!available} onClick={() => send('next', { reuse: true })}><RotateCcw size={15} /> Reuse this path</button>}
        <button className={`button ${config.mode !== 'repeat' || !state?.path ? 'primary' : ''}`} disabled={!available} onClick={() => send('next', { reuse: false })}><Hand size={15} /> Teach a new path</button>
      </>
      break
    case 'finished':
      title = 'Dataset ready'
      detail = `${state?.accepted ?? 0} saved episodes · LeRobotDataset v3.0`
      controls = <button className="button primary" disabled={!available} onClick={() => send('disconnect')}><ArrowRight size={15} /> New session</button>
      extra = state?.dataset_path && <DatasetOutput path={state.dataset_path} />
      break
    case 'fault':
      title = 'Workspace stopped'
      detail = state?.fault || 'Check the arm before resuming.'
      if (state?.notice === 'Episode saved.' || state?.notice === 'Take discarded.') {
        extra = <p className="hint" role="status">{state.notice}</p>
      }
      controls = <>
        <button className="button" disabled={!available || Boolean(state?.operation)} onClick={() => send('disconnect')}><Radio size={15} /> Disconnect</button>
        <button className="button primary" disabled={!available || Boolean(state?.operation)} onClick={() => instruct('recover', 'Recover control?', 'Support the arm and clear the workspace. If torque is mixed, recovering will seed fresh measured goals before re-enabling it.', 'Recover')}><RotateCcw size={15} /> Recover</button>
      </>
      break
  }

  return <section className={`panel workflow-card ${phase === 'fault' ? 'fault-card' : ''}`} aria-label="Current episode step">
    <div className="workflow-copy"><h2>{title}</h2><p>{detail}</p></div>
    {(controls || extra) && <div className="workflow-content">
      {controls && <div className="actions">{controls}</div>}
      {extra}
      {state && (replayKeyboard || phase === 'manual') &&
        <ManualControls state={state} send={send} enabled={online && !confirm} pending={!available} replaying={replayKeyboard} />}
      {canManageRest && <details className="episode-options">
        <summary><ChevronDown size={14} /> Rest position<span>{state?.rest_target ? state.at_rest ? 'At rest' : 'Saved' : 'Not set'}</span></summary>
        <div>
          <p className="hint">Rest is a parking position, separate from the task start. Saving it does not move the arm.</p>
          <div className="actions">
            <button className="button" disabled={!available} onClick={() => state?.rest_target ? setConfirm({
              title: 'Replace the saved rest position?', text: 'Save the current arm position as the new rest. The task start does not change.',
              label: 'Replace rest', submit: () => send('capture_rest', { confirmed: true }),
            }) : send('capture_rest')}><Save size={15} /> {state?.rest_target ? 'Replace rest with current' : 'Save current as rest'}</button>
            {state?.rest_target && <>
              <button className="button" disabled={!canReturn || state.at_rest} onClick={park}><RotateCcw size={15} /> Return to rest</button>
              <button className="text-button" disabled={!available} onClick={() => setConfirm({
                title: 'Clear the saved rest position?', text: 'This removes the parking position without moving the arm.',
                label: 'Clear rest', submit: () => send('clear_rest', { confirmed: true }),
              })}><X size={14} /> Clear rest</button>
            </>}
          </div>
        </div>
      </details>}
    </div>}
    {(phase === 'ready' || canDiscard || canManagePower) && <div className="workflow-secondary">
      {phase === 'ready' && <button className="text-button" disabled={!available} onClick={startPractice}><Hand size={14} /> Practice controls</button>}
      {canManagePower && <button className="text-button" disabled={!available} onClick={changePower}>
        {torqueOn ? <ZapOff size={14} /> : <Zap size={14} />}{torqueOn ? phase === 'ready' ? 'Release arm to position' : 'Turn torque off' : 'Turn torque on'}
      </button>}
      {canDiscard && <button className="text-button discard-draft" disabled={!available} onClick={() => discardEpisode()}>
        <X size={14} /> Discard episode
      </button>}
    </div>}
    {confirm && <Confirmation value={confirm} close={() => setConfirm(null)} />}
  </section>
}

function App() {
  const { state, online, send, error, setError, pending } = useWorkspace()
  const [config, setConfig] = useState<Config>(defaults)
  const [datasetEdited, setDatasetEdited] = useState(false)
  const [takeFrame, setTakeFrame] = useState(0)
  const phase = state?.phase ?? 'setup'
  const setupConfig = { ...config, dataset: datasetEdited ? config.dataset : state?.config.dataset ?? config.dataset }
  const activeConfig = state?.config ?? config
  const available = online && !pending
  useEffect(() => {
    if (phase !== 'review') setTakeFrame(0)
    if (phase !== 'setup') setDatasetEdited(false)
  }, [phase])

  return <div className="shell">
    <Header state={state} online={online} available={available} send={send} />
    <main className={`main ${phase === 'setup' ? 'setup-main' : ''}`}>
      {phase === 'setup' ? <div className="setup-intro"><h1 className="setup-title">Set up collection</h1><p>Try the full workflow in the simulator, or connect a calibrated SO-101 follower.</p></div> : state && <EpisodeHeader state={state} config={activeConfig} />}
      {error && <div className="alert" role="alert"><TriangleAlert size={17} /><span>{error}</span><button className="icon-button bare push" aria-label="Dismiss error" onClick={() => setError('')}><X size={17} /></button></div>}
      {phase === 'setup' ? <SetupPanel config={setupConfig} change={next => {
        if (next.dataset !== setupConfig.dataset) setDatasetEdited(true)
        setConfig(next)
      }} available={available} connect={() => send('connect', { config: setupConfig })} /> :
        <div className="main-stack">
          <Workflow state={state} config={activeConfig} send={send} online={online} available={available} />
          {phase !== 'finished' && <>
            <CameraPanel config={activeConfig} state={state} online={online} reviewIndex={takeFrame} setReviewIndex={setTakeFrame} />
            <details className="diagnostics">
              <summary><ChevronDown size={14} /> Diagnostics</summary>
              <div className="diagnostics-content"><Telemetry state={state} /><Timeline state={state} /></div>
            </details>
          </>}
        </div>}
    </main>
  </div>
}

createRoot(document.getElementById('root')!).render(<App />)
