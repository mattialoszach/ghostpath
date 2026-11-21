import { useCallback, useEffect, useRef, useState } from 'react'
import type { Send, State } from './types'

export function useWorkspace() {
  const socket = useRef<WebSocket | null>(null)
  const [state, setState] = useState<State | null>(null)
  const [online, setOnline] = useState(false)
  const [error, setError] = useState('')
  const [pending, setPending] = useState(false)
  const request = useRef(0)
  const waiting = useRef<number | null>(null)

  const send = useCallback<Send>((action, extra = {}) => {
    if (socket.current?.readyState !== WebSocket.OPEN) {
      setError('The workspace connection is unavailable.')
      return
    }
    const id = ++request.current
    if (action !== 'jog' && action !== 'demo_move' && action !== 'grip_adjust' && action !== 'stop') {
      if (waiting.current !== null) return
      waiting.current = id
      setPending(true)
      setError('')
    }
    socket.current.send(JSON.stringify({ id, action, ...extra }))
  }, [])

  useEffect(() => {
    let active = true
    let reconnect = 0
    let waitingForState = false
    const finishCommand = () => {
      waiting.current = null
      waitingForState = false
      setPending(false)
    }
    const open = () => {
      const address = new URL('/ws', location.href)
      address.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
      const connection = new WebSocket(address)
      socket.current = connection
      connection.onopen = () => setOnline(true)
      connection.onmessage = (event: MessageEvent) => {
        const message = JSON.parse(event.data)
        if (message.type === 'state') {
          setState(message.state)
          if (waitingForState) finishCommand()
        } else if (message.type === 'ack' && message.id === waiting.current) {
          waitingForState = true
        } else if (message.type === 'error') {
          setError(message.message)
          if (message.id === waiting.current || message.id == null) finishCommand()
        }
      }
      connection.onclose = (event) => {
        if (!active) return
        setOnline(false)
        finishCommand()
        if (event.code !== 1008) reconnect = window.setTimeout(open, 1400)
      }
    }
    open()
    const heartbeat = window.setInterval(() => {
      if (socket.current?.readyState === WebSocket.OPEN) {
        socket.current.send(JSON.stringify({ action: 'heartbeat' }))
      }
    }, 160)
    return () => {
      active = false
      window.clearInterval(heartbeat)
      window.clearTimeout(reconnect)
      socket.current?.close()
    }
  }, [])

  return { state, online, send, error, setError, pending }
}
