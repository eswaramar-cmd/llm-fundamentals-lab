/**
 * Wire types for POST /chat/stream.
 *
 * These mirror `backend/main.py::_sse` and `backend/agent/schemas.py::ChatRequest`
 * field for field. If you change one side, change the other: this file is the
 * only place the backend contract is described on the frontend.
 */

/** Request body accepted by /chat/stream. */
export interface ChatStreamRequest {
  message: string
  user_id: string
  thread_id: string
}

interface BaseEvent {
  /** Echoed by the backend on every frame; useful for filtering in the log. */
  type: string
}

export interface StatusEvent extends BaseEvent {
  type: 'status'
  /** Human label for the node, e.g. "Searching knowledge base". */
  message: string
  /** LangGraph node id, e.g. "retrieve_context". */
  node: string
}

export interface TokenEvent extends BaseEvent {
  type: 'token'
  /** Verbatim provider output fragment. Append, never replace. */
  text: string
}

export interface ToolStartEvent extends BaseEvent {
  type: 'tool_start'
  /** Tool name as registered on the graph. */
  tool: string
  input: Record<string, unknown>
}

export interface ToolEndEvent extends BaseEvent {
  type: 'tool_end'
  tool: string
  /** Already compacted server-side by `_brief_result`. */
  result: string
  time_ms: number
}

/** Real send_email outcome, derived by the backend from the actual tool result. */
export interface EmailOutcome {
  /** `sent` only when SMTP accepted the message. `preview` sent nothing. */
  status: 'sent' | 'preview' | 'failed'
  recipient: string
  detail: string
}

/** One attachment as returned by POST /upload. */
export interface UploadedFile {
  file_id: string
  name: string
  ext: string
  mime: string
  /** Drives the icon and colour: pdf | word | excel | image | powerpoint | text. */
  category: 'pdf' | 'word' | 'excel' | 'image' | 'powerpoint' | 'text'
  size: number
  /** Extracted text for preview. Images return a short "ready to send" note. */
  text_preview: string
  thumbnail: string | null
}

export interface UploadResponse {
  user_id: string
  count: number
  total_bytes: number
  max_files_per_request: number
  max_file_bytes: number
  max_attachment_bytes: number
  allowed_extensions: string[]
  files: UploadedFile[]
}

export interface DoneEvent extends BaseEvent {
  type: 'done'
  /** Total wall-clock time for the whole stream. */
  latency_ms: number
  /** Time until the first token arrived. This is what users perceive as speed. */
  time_to_first_token_ms: number | null
  /** e.g. "google/gemini-3.1-flash-lite". */
  model: string
  /** Number of streamed chunks, not tokens. */
  chunks: number
  /** Characters emitted. Used for throughput, since the streaming path has no
   * provider token usage. */
  chars: number
  chars_per_second: number
  /** Set when the graph finished in a failed state. The answer still carries
   * whatever text was produced before the failure. */
  error: string | null
  /** Present only when send_email actually ran. */
  email?: EmailOutcome | null
  answer: string
}

export interface ErrorEvent extends BaseEvent {
  type: 'error'
  message: string
}

export type ChatStreamEvent =
  | StatusEvent
  | TokenEvent
  | ToolStartEvent
  | ToolEndEvent
  | DoneEvent
  | ErrorEvent

/** Narrows an unknown parsed frame to a known event, or null when unrecognised. */
export function isChatStreamEvent(value: unknown): value is ChatStreamEvent {
  if (typeof value !== 'object' || value === null) return false
  const type = (value as { type?: unknown }).type
  return (
    type === 'status' ||
    type === 'token' ||
    type === 'tool_start' ||
    type === 'tool_end' ||
    type === 'done' ||
    type === 'error'
  )
}