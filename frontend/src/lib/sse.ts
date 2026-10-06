/**
 * Minimal SSE frame reader.
 *
 * `EventSource` cannot be used here: it only issues GET requests, while
 * /chat/stream is a POST with a JSON body. So the stream is read by hand.
 *
 * The parser is incremental on purpose. Network chunks split at arbitrary byte
 * offsets, so a frame can arrive in pieces; anything after the last `\n\n` is
 * held in `buffer` until the rest shows up. Comment frames (`:heartbeat`) are
 * dropped, which is how the backend's 15s keepalive is ignored.
 */

export interface SseFrame {
  /** Value of the `event:` line, when present. */
  event: string | null
  /** Joined `data:` lines. */
  data: string
}

function parseFrame(raw: string): SseFrame | null {
  let event: string | null = null
  const dataLines: string[] = []

  for (const line of raw.split('\n')) {
    if (line.length === 0 || line.startsWith(':')) continue

    if (line.startsWith('event:')) {
      event = line.slice('event:'.length).trim()
    } else if (line.startsWith('data:')) {
      // A single optional space after the colon is part of the framing.
      dataLines.push(line.slice('data:'.length).replace(/^ /, ''))
    }
  }

  if (dataLines.length === 0) return null

  return { event, data: dataLines.join('\n') }
}

export async function* parseSseStream(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<SseFrame> {
  const reader = body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  try {
    for (;;) {
      const { done, value } = await reader.read()

      if (done) break

      buffer += decoder.decode(value, { stream: true })
      buffer = buffer.replace(/\r\n/g, '\n')

      let boundary = buffer.indexOf('\n\n')

      while (boundary !== -1) {
        const frame = parseFrame(buffer.slice(0, boundary))
        buffer = buffer.slice(boundary + 2)

        if (frame) yield frame

        boundary = buffer.indexOf('\n\n')
      }
    }

    // Flush the decoder and any final frame that arrived without a trailing break.
    buffer += decoder.decode()

    const tail = parseFrame(buffer.trim())

    if (tail) yield tail
  } finally {
    // Releases the lock so the caller can cancel the body on abort.
    reader.releaseLock()
  }
}