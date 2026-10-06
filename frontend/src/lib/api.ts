import type { ChatStreamRequest, UploadedFile, UploadResponse } from '../types/chat'

/**
 * Dev requests go through the Vite proxy at /api so the browser sees a single
 * origin. Set VITE_API_BASE_URL to call the backend directly (for example when
 * deploying the built bundle separately).
 */
const RAW_BASE = import.meta.env.VITE_API_BASE_URL ?? '/api'

export const API_BASE = RAW_BASE.replace(/\/+$/, '')

export const endpoints = {
  health: `${API_BASE}/health`,
  healthLlm: `${API_BASE}/health/llm`,
  chatStream: `${API_BASE}/chat/stream`,
  upload: `${API_BASE}/upload`,
} as const

/**
 * Extensions the backend accepts. Mirrors ALLOWED_EXTS in
 * backend/agent/uploads.py — the server is still authoritative and will reject
 * anything else, but rejecting a file in the picker is friendlier than a 400.
 */
export const ACCEPTED_EXTENSIONS = [
  '.pdf',
  '.doc',
  '.docx',
  '.txt',
  '.csv',
  '.xls',
  '.xlsx',
  '.pptx',
  '.jpg',
  '.jpeg',
  '.png',
  '.webp',
] as const

export const ACCEPT_ATTRIBUTE = ACCEPTED_EXTENSIONS.join(',')

/**
 * Client-side caps for pre-flight checks only.
 *
 * The backend owns the real limits and reports them on every upload response;
 * these deliberately sit at the most generous allowed configuration so the
 * browser never refuses something the server would have accepted. Anything
 * tighter here would invent a restriction the user never agreed to.
 */
export const MAX_FILES_PER_REQUEST = 50
export const MAX_FILE_BYTES = 200 * 1024 * 1024
export const MAX_TOTAL_ATTACHMENT_BYTES = 1000 * 1024 * 1024

/**
 * Upload files and receive a file_id for each.
 *
 * The id is what send_email(file_ids=[...]) takes. The browser never sends a
 * path, so there is nothing for a caller to tamper with between upload and send.
 */
export async function uploadFiles(
  files: File[],
  userId: string,
  signal?: AbortSignal,
): Promise<UploadResponse> {
  const form = new FormData()

  files.forEach((file) => form.append('files', file, file.name))

  const query = new URLSearchParams({ user_id: userId })

  let response: Response

  try {
    response = await fetch(`${endpoints.upload}?${query}`, {
      method: 'POST',
      body: form,
      signal,
    })
  } catch (cause) {
    if (signal?.aborted) throw cause

    // A stopped backend surfaces as a bare TypeError "Failed to fetch", which
    // names nothing. This is the most common failure by far, so it is stated.
    throw new Error(
      'Cannot reach the backend. Is it running on port 8000? Start it with: ' +
        'venv\\Scripts\\python.exe -m uvicorn backend.main:app --port 8000',
    )
  }

  if (!response.ok) {
    let detail = `Upload failed (${response.status})`

    try {
      const body = await response.json()
      if (body?.detail) detail = String(body.detail)
    } catch {
      // Non-JSON error body; the status line above is the best we have.
    }

    throw new Error(detail)
  }

  return (await response.json()) as UploadResponse
}

export function totalBytes(files: UploadedFile[]): number {
  return files.reduce((sum, file) => sum + file.size, 0)
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export function buildChatStreamBody(input: {
  message: string
  userId: string
  threadId: string
}): ChatStreamRequest {
  return {
    message: input.message,
    user_id: input.userId,
    thread_id: input.threadId,
  }
}