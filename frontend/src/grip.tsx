import { useEffect, useState } from 'react'
import { closeGrip, keyLabel, openGrip } from './controls'
import { elapsed } from './types'
import type { GripState, Send } from './types'

type Direction = GripState['direction']

function useGripKeys(ready: boolean, send: Send) {
  const [held, setHeld] = useState<Direction>(0)

  useEffect(() => {
    if (!ready) return
    const keys = new Set<string>()
    let direction: Direction = 0
    const publish = () => {
      const next = (Number(keys.has(openGrip.code)) - Number(keys.has(closeGrip.code))) as Direction
      if (direction === next) return
      direction = next
      setHeld(next)
      send('grip_adjust', { grip_direction: direction })
    }
    const clear = () => { keys.clear(); publish() }
    const down = (event: KeyboardEvent) => {
      if (event.code !== closeGrip.code && event.code !== openGrip.code) return
      if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return
      if (event.target instanceof HTMLElement &&
          event.target.closest('input, select, textarea, [contenteditable], dialog')) return
      event.preventDefault()
      if (event.repeat && !keys.has(event.code)) return
      keys.add(event.code)
      publish()
    }
    const up = (event: KeyboardEvent) => {
      if (keys.delete(event.code)) publish()
    }
    const visibility = () => { if (document.hidden) clear() }
    const focus = (event: FocusEvent) => {
      if (event.target instanceof HTMLElement &&
          event.target.closest('input, select, textarea, [contenteditable], dialog')) clear()
    }
    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)
    window.addEventListener('blur', clear)
    document.addEventListener('visibilitychange', visibility)
    document.addEventListener('focusin', focus)
    const heartbeat = window.setInterval(() => {
      if (direction) send('grip_adjust', { grip_direction: direction })
    }, 70)
    return () => {
      window.removeEventListener('keydown', down)
      window.removeEventListener('keyup', up)
      window.removeEventListener('blur', clear)
      document.removeEventListener('visibilitychange', visibility)
      document.removeEventListener('focusin', focus)
      window.clearInterval(heartbeat)
      clear()
    }
  }, [ready, send])

  return ready ? held : 0
}

export function GripPanel({ grip, duration, send }: { grip: GripState; duration: number; send: Send }) {
  const held = useGripKeys(grip.ready, send)
  const values = [...grip.taught, ...grip.edited]
  const minimum = Math.min(...values)
  const maximum = Math.max(...values)
  const padding = Math.max(4, (16 - maximum + minimum) / 2)
  const low = minimum - padding
  const high = maximum + padding
  const points = (curve: number[]) => curve.map((value, i) =>
    `${i * 600 / (curve.length - 1)},${56 - 56 * (value - low) / (high - low)}`,
  )
  const taught = points(grip.taught)
  const edited = points(grip.edited)
  const x = grip.cursor === null ? null : grip.cursor * 600
  const active = grip.enabled && (held !== 0 || Math.abs(grip.offset) >= 0.2)

  return <section className={`grip-panel ${active ? 'grip-panel-active' : ''}`} aria-label="Gripper timeline">
    <div className="grip-heading">
      <strong>{grip.enabled ? 'Boomerang grip assist' : 'Gripper profile'}</strong>
      <span>{grip.enabled
        ? grip.ready ? held < 0 ? 'Tightening' : held > 0 ? 'Softening' : active ? 'Easing back' : 'Following path' : 'Waiting for reverse path'
        : 'Edits saved for replay'}</span>
    </div>
    {grip.enabled && <div className="grip-shortcuts" aria-label="Hold C to close or O to open">
      <span className={held < 0 ? 'held' : ''}><kbd>{keyLabel(closeGrip.code)}</kbd> Close more</span>
      <span className={held > 0 ? 'held' : ''}><kbd>{keyLabel(openGrip.code)}</kbd> Open more</span>
      <small>Hold a key to adjust. Release to return to the automatic grip.</small>
    </div>}
    <div className="grip-chart">
      <svg viewBox="0 0 600 56" preserveAspectRatio="none" role="img" aria-label="Taught and edited gripper positions">
        {[14, 28, 42].map(y => <line key={y} x1={0} y1={y} x2={600} y2={y} />)}
        <polygon className="grip-difference" points={[...taught, ...edited.slice().reverse()].join(' ')} />
        <polyline className="grip-edited" points={edited.join(' ')} />
        <polyline className="grip-taught" points={taught.join(' ')} />
        {x !== null && <line className="grip-playhead" x1={x} y1={0} x2={x} y2={56} />}
      </svg>
    </div>
    <div className="grip-legend">
      <span>Start</span>
      <span className="grip-key-taught">Taught grip</span>
      <span className="grip-key-edited">{grip.enabled ? 'With adjustments · ← Boomerang' : 'Edited grip'}</span>
      <span>{elapsed(duration)}</span>
    </div>
    {grip.enabled && <p className="hint">Changes save after a completed Boomerang. Keys adjust closure within the existing torque limit.</p>}
  </section>
}
