import type { DoneEvent, EmailOutcome, ErrorEvent, StatusEvent, ToolEndEvent, ToolStartEvent } from '../types/chat'

export type ConnectionState =
  | 'idle'
  | 'connecting'
  | 'streaming'
  | 'done'
  | 'stopped'
  | 'error'
  | 'disconnected'

export type MessageStatus = 'streaming' | 'complete' | 'stopped' | 'error'

export interface ToolCall {
  id: string
  tool: string
  input: Record<string, unknown>
  result: string | null
  /** ms since the stream opened, so rows sort and display consistently. */
  startedAt: number
  endedAt: number | null
  durationMs: number | null
  state: 'running' | 'ok'
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  text: string
  createdAt: number
  status: MessageStatus
  tools: ToolCall[]
}

export interface Metrics {
  latencyMs: number | null
  /** Time to first token. The number that actually reflects perceived speed. */
  ttftMs: number | null
  model: string | null
  chunks: number
  chars: number
  charsPerSecond: number | null
  /** Set when a real send_email ran. Drives the green "Sent to X" banner. */
  email: EmailOutcome | null
}

export interface ChatState {
  messages: Message[]
  activeNode: string | null
  statusText: string | null
  connection: ConnectionState
  metrics: Metrics
  error: string | null
}

export const initialChatState: ChatState = {
  messages: [],
  activeNode: null,
  statusText: null,
  connection: 'idle',
  metrics: {
    latencyMs: null,
    ttftMs: null,
    model: null,
    chunks: 0,
    chars: 0,
    charsPerSecond: null,
    email: null,
  },
  error: null,
}

export type ChatAction =
  | { type: 'user_message'; id: string; text: string; at: number }
  | { type: 'assistant_start'; id: string; at: number }
  | { type: 'status'; event: StatusEvent }
  | { type: 'token'; text: string }
  | { type: 'tool_start'; id: string; event: ToolStartEvent; at: number }
  | { type: 'tool_end'; event: ToolEndEvent }
  | { type: 'done'; event: DoneEvent }
  | { type: 'error'; event: ErrorEvent }
  | { type: 'stopped' }
  | { type: 'disconnected'; message: string }
  | { type: 'reset' }

/** Applies `update` to the trailing assistant message, which is the live one. */
function patchLastAssistant(
  state: ChatState,
  update: (message: Message) => Message,
): ChatState {
  const index = state.messages.findLastIndex((m) => m.role === 'assistant')

  if (index === -1) return state

  const messages = [...state.messages]
  messages[index] = update(messages[index]!)

  return { ...state, messages }
}

export function chatReducer(state: ChatState, action: ChatAction): ChatState {
  switch (action.type) {
    case 'user_message':
      return {
        ...state,
        connection: 'connecting',
        error: null,
        metrics: initialChatState.metrics,
        messages: [
          ...state.messages,
          {
            id: action.id,
            role: 'user',
            text: action.text,
            createdAt: action.at,
            status: 'complete',
            tools: [],
          },
        ],
      }

    case 'assistant_start':
      return {
        ...state,
        connection: 'streaming',
        messages: [
          ...state.messages,
          {
            id: action.id,
            role: 'assistant',
            text: '',
            createdAt: action.at,
            status: 'streaming',
            tools: [],
          },
        ],
      }

    case 'status':
      return { ...state, statusText: action.event.message, activeNode: action.event.node }

    case 'token':
      return patchLastAssistant(state, (message) => ({
        ...message,
        text: message.text + action.text,
      }))

    case 'tool_start':
      return patchLastAssistant(state, (message) => ({
        ...message,
        tools: [
          ...message.tools,
          {
            id: action.id,
            tool: action.event.tool,
            input: action.event.input,
            result: null,
            startedAt: action.at,
            endedAt: null,
            durationMs: null,
            state: 'running',
          },
        ],
      }))

    case 'tool_end':
      return patchLastAssistant(state, (message) => {
        // Tools can nest, so close the most recent still-running call with a
        // matching name rather than assuming the last one overall.
        const index = message.tools.findLastIndex(
          (tool) => tool.tool === action.event.tool && tool.state === 'running',
        )

        if (index === -1) return message

        const tools = [...message.tools]
        const open = tools[index]!

        tools[index] = {
          ...open,
          result: action.event.result,
          endedAt: action.event.time_ms,
          durationMs: action.event.time_ms - open.startedAt,
          state: 'ok',
        }

        return { ...message, tools }
      })

    case 'done': {
      const withMessage = patchLastAssistant(state, (message) => ({
        ...message,
        // The backend sends the assembled answer as well, which also covers the
        // case where the final chunk arrived after the client stopped reading.
        text: action.event.answer || message.text,
        status: action.event.error ? 'error' : 'complete',
      }))

      return {
        ...withMessage,
        connection: action.event.error ? 'error' : 'done',
        activeNode: null,
        statusText: null,
        error: action.event.error ?? state.error,
        metrics: {
          latencyMs: action.event.latency_ms,
          ttftMs: action.event.time_to_first_token_ms,
          model: action.event.model,
          chunks: action.event.chunks,
          chars: action.event.chars,
          charsPerSecond: action.event.chars_per_second,
          email: action.event.email ?? null,
        },
      }
    }

    case 'error':
      return {
        ...patchLastAssistant(state, (message) => ({ ...message, status: 'error' })),
        connection: 'error',
        activeNode: null,
        error: action.event.message,
      }

    case 'stopped':
      return {
        ...patchLastAssistant(state, (message) => ({
          ...message,
          status: message.text ? 'stopped' : 'error',
        })),
        connection: 'stopped',
        activeNode: null,
        statusText: null,
      }

    case 'disconnected':
      return {
        ...state,
        connection: 'disconnected',
        activeNode: null,
        statusText: null,
        error: action.message,
      }

    case 'reset':
      return initialChatState
  }
}