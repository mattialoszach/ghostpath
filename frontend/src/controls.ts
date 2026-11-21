import type { Joint } from './types'

interface Shortcut {
  code: string
  label: string
  value: -1 | 1
}

export const closeGrip = { code: 'KeyC', label: 'Close', value: -1 } as const
export const openGrip = { code: 'KeyO', label: 'Open', value: 1 } as const

export const jogKeys: Record<Joint, [Shortcut, Shortcut]> = {
  shoulder_pan: [{ code: 'KeyA', label: 'Left', value: -1 }, { code: 'KeyD', label: 'Right', value: 1 }],
  shoulder_lift: [{ code: 'KeyW', label: 'Up', value: -1 }, { code: 'KeyS', label: 'Down', value: 1 }],
  elbow_flex: [{ code: 'ArrowUp', label: 'Bend', value: 1 }, { code: 'ArrowDown', label: 'Straighten', value: -1 }],
  wrist_flex: [{ code: 'KeyI', label: 'Up', value: -1 }, { code: 'KeyK', label: 'Down', value: 1 }],
  wrist_roll: [{ code: 'KeyJ', label: 'Left', value: 1 }, { code: 'KeyL', label: 'Right', value: -1 }],
  gripper: [closeGrip, openGrip],
}

const arrows: Record<string, string> = { ArrowUp: '↑', ArrowDown: '↓' }
export const keyLabel = (code: string) => arrows[code] ?? code.replace('Key', '')
