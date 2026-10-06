import { useCallback, useEffect, useReducer, useRef } from 'react'
import { buildChatStreamBody, endpoints } from '../lib/api'
import { createId } from '../lib/id'
import { sseLog } from '../lib/logger'
import { parseSseStream } from '../lib/sse'
import { chatReducer, initialChatState, type ChatAction, type ChatState } from '../state/chatReducer'
import { isChatStreamEvent } from '../types/chat'

export const DEFAULT_USER_ID = 'default_user'

export interface ChatStreamApi {
  state: ChatState
  send: (message: string) => void
  stop: () => void
  retry: () => void
  reset: () => void
}

export function useChatStream(): ChatStreamApi {
  const [state, dispatch] = useReducer(chatReducer, initialChatState)

  const controllerRef = useRef<AbortController | null>(null)
  const streamStartedAtRef = useRef(0)
  const frameCountRef = useRef(0)
  const lastRequestRef = useRef<{ message: string; threadId: string } | null>(null)

  // Token frames are coalesced and flushed once per animation frame. A fast
  // provider can emit dozens of chunks per second; dispatching each one
  // separately renders more often than the display refreshes, which is wasted
  // work and visible jitter.
  const pendingTextRef = useRef('')
  const frameRef = useRef<number | null>(null)

  const flushTokens = useCallback(() => {
    if (frameRef.current !== null) {
      cancelAnimationFrame(frameRef.current)
      frameRef.current = null
    }

    if (!pendingTextRef.current) return

    const text = pendingTextRef.current
    pendingTextRef.current = ''
    dispatch({ type: 'token', text })
  }, [])

  const queueToken = useCallback(
    (text: string) => {
      pendingTextRef.current += text

      if (frameRef.current === null) {
        frameRef.current = requestAnimationFrame(() => {
          frameRef.current = null

          if (pendingTextRef.current) {
            const pending = pendingTextRef.current
            pendingTextRef.current = ''
            dispatch({ type: 'token', text: pending })
          }
        })
      }
    },
    [],
  )

  // Abort on unmount so a navigation mid-answer does not leak the request and
  // does not dispatch into a dead component.
  useEffect(() => {
    return () => {
      controllerRef.current?.abort()
      if (frameRef.current !== null) cancelAnimationFrame(frameRef.current)
    }
  }, [])

  const run = useCallback(async (message: string, threadId: string) => {
    controllerRef.current?.abort()

    const controller = new AbortController()
    controllerRef.current = controller

    frameCountRef.current = 0
    streamStartedAtRef.current = performance.now()
    lastRequestRef.current = { message, threadId }

    dispatch({ type: 'user_message', id: createId('msg'), text: message, at: Date.now() })
    dispatch({ type: 'assistant_start', id: createId('msg'), at: Date.now() })

    const body = buildChatStreamBody({ message, userId: DEFAULT_USER_ID, threadId })
    sseLog.request(endpoints.chatStream, body)

    try {
      const response = await fetch(endpoints.chatStream, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
        body: JSON.stringify(body),
        signal: controller.signal,
      })

      if (!response.ok || !response.body) {
        throw new Error(`HTTP ${response.status} ${response.statusText}`)
      }

      for await (const frame of parseSseStream(response.body)) {
        if (controller.signal.aborted) break

        frameCountRef.current += 1

        let payload: unknown
        try {
          payload = JSON.parse(frame.data)
        } catch {
          sseLog.skipped(frameCountRef.current, frame.data)
          continue
        }

        if (!isChatStreamEvent(payload)) {
          sseLog.skipped(frameCountRef.current, frame.data)
          continue
        }

        sseLog.frame(frameCountRef.current, payload.type, payload)

        switch (payload.type) {
          case 'status':
            dispatch({ type: 'status', event: payload })
            break
          case 'token':
            queueToken(payload.text)
            break
          case 'tool_start':
            flushTokens()
            dispatch({
              type: 'tool_start',
              id: createId('tool'),
              event: payload,
              at: Math.round(performance.now() - streamStartedAtRef.current),
            })
            break
          case 'tool_end':
            dispatch({ type: 'tool_end', event: payload })
            break
          case 'done':
            flushTokens()
            dispatch({ type: 'done', event: payload })
            sseLog.done(payload)
            break
          case 'error':
            flushTokens()
            dispatch({ type: 'error', event: payload })
            break
        }

        if (payload.type === 'done' || payload.type === 'error') break
      }
    } catch (error) {
      // Whatever text is already buffered belongs on screen before the stream
      // is marked stopped or failed, otherwise the last chunk is lost.
      flushTokens()

      if (controller.signal.aborted) {
        sseLog.aborted()
        dispatch({ type: 'stopped' })
        return
      }

      const message =
        error instanceof Error ? error.message : 'Unexpected streaming failure'

      sseLog.failed(error)
      dispatch({
        type: 'disconnected',
        message: `Could not reach the backend at ${endpoints.chatStream}. ${message}`,
      })
    } finally {
      if (controllerRef.current === controller) controllerRef.current = null
    }
  }, [flushTokens, queueToken])

  const send = useCallback((message: string) => {
    const trimmed = message.trim()
    if (!trimmed) return

    // One thread per browser tab keeps server-side checkpoints together.
    const threadId = lastRequestRef.current?.threadId ?? createId('thread')
    void run(trimmed, threadId)
  }, [run])

  const stop = useCallback(() => {
    // Flush first so text already received is not lost when the request aborts.
    flushTokens()
    controllerRef.current?.abort()
  }, [flushTokens])

  const retry = useCallback(() => {
    const last = lastRequestRef.current
    if (last) void run(last.message, last.threadId)
  }, [run])

  const reset = useCallback(() => {
    controllerRef.current?.abort()
    lastRequestRef.current = null
    dispatch({ type: 'reset' } satisfies ChatAction)
  }, [])

  return { state, send, stop, retry, reset }
}