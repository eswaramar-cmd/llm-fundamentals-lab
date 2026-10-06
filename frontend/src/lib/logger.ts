/**
 * Every SSE frame is logged to the console with an `[sse]` prefix.
 *
 * This is deliberate: when a stream misbehaves in the browser, the raw frame log
 * is the fastest way to tell a backend regression apart from a UI bug. The
 * prefix makes the lines easy to filter in DevTools.
 */

const PREFIX = '[sse]'

export const sseLog = {
  request(url: string, body: unknown) {
    console.info(`${PREFIX} -> POST ${url}`, body)
  },

  frame(index: number, type: string, payload: unknown) {
    console.debug(`${PREFIX} <- #${index} ${type}`, payload)
  },

  skipped(index: number, raw: string) {
    console.warn(`${PREFIX} <- #${index} unrecognised frame`, raw)
  },

  done(summary: {
    latency_ms: number
    time_to_first_token_ms: number | null
    model: string
    chunks: number
  }) {
    console.info(
      `${PREFIX} done in ${summary.latency_ms}ms (first token ${
        summary.time_to_first_token_ms ?? 'n/a'
      }ms) on ${summary.model}, ${summary.chunks} chunks`,
    )
  },

  aborted() {
    console.info(`${PREFIX} stream aborted by client`)
  },

  failed(error: unknown) {
    console.error(`${PREFIX} stream failed`, error)
  },
}