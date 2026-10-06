/** Parsing helpers for the send_email preview payload. */

export interface EmailDraft {
  to: string
  subject: string
  body: string
}

/**
 * Read To/Subject/Body out of the tool's PREVIEW ONLY result.
 *
 * The preview text is produced by the backend tool, so this is a parser for a
 * known format rather than guesswork over model prose. Returns null when the
 * string is not a preview.
 */
export function parsePreview(detail: string): EmailDraft | null {
  const to = detail.match(/^To:\s*(.+)$/m)?.[1]?.trim() ?? ''
  const subject = detail.match(/^Subject:\s*(.+)$/m)?.[1]?.trim() ?? ''
  const body = detail.match(/^Body:\s*([\s\S]*)$/m)?.[1]?.trim() ?? ''

  if (!to && !subject) return null

  return { to, subject, body }
}