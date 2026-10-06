import { useState } from 'react'
import { DEFAULT_USER_ID, type ChatStreamApi } from '../../hooks/useChatStream'
import { parsePreview, type EmailDraft } from '../../lib/email'
import type { ChatState } from '../../state/chatReducer'
import type { UploadedFile } from '../../types/chat'
import { AttachmentBar } from './AttachmentBar'
import { Composer } from './Composer'
import { ConnectionBanner } from './ConnectionBanner'
import { EmailPreviewCard } from './EmailPreviewCard'
import { MessageList } from './MessageList'
import { StreamMetrics } from './StreamMetrics'
import styles from './Chat.module.css'

interface ChatPanelProps {
  stream: ChatStreamApi
}

const CONFIRM_PHRASES = /^\s*(yes|yeah|yep|yup|confirm|send it|go ahead|sure|ok|okay|do it)\b/i

/**
 * Append the uploaded files to the outgoing message.
 *
 * The file_id is the only handle send_email accepts, so the model has to see it
 * to pass it through. Listing the ids verbatim — rather than describing the
 * files and hoping the model matches them up — is what keeps "send these files"
 * attached to the files the user actually picked.
 */
function withAttachments(message: string, files: UploadedFile[]): string {
  if (files.length === 0) return message

  const manifest = files
    .map((file) => `- ${file.name} (file_id=${file.file_id})`)
    .join('\n')

  return `${message}\n\nAttached files (pass these file_ids to send_email):\n${manifest}`
}

/**
 * Newest send_email preview in the transcript, or null.
 *
 * Read from the tool result the backend actually streamed rather than re-parsed
 * from the model's prose, because that result is the authoritative record of
 * what would be sent. This runs on every render over a handful of messages,
 * which is cheaper than memoising.
 */
function findDraft(messages: ChatState['messages']): EmailDraft | null {
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const toolCalls = messages[i]?.tools ?? []

    for (let t = toolCalls.length - 1; t >= 0; t -= 1) {
      const tool = toolCalls[t]!

      if (tool.tool !== 'send_email' || !tool.result?.startsWith('PREVIEW ONLY')) {
        continue
      }

      const parsed = parsePreview(tool.result)

      if (parsed) {
        // Fall back to the tool input when the preview text is truncated.
        const input = tool.input as Record<string, unknown>

        return {
          to: parsed.to || String(input.to_email ?? ''),
          subject: parsed.subject || String(input.subject ?? ''),
          body: parsed.body || String(input.body ?? ''),
        }
      }
    }
  }

  return null
}

export function ChatPanel({ stream }: ChatPanelProps) {
  const { state, send, stop, retry } = stream
  const [dismissed, setDismissed] = useState<string | null>(null)
  const [attachments, setAttachments] = useState<UploadedFile[]>([])

  const isStreaming = state.connection === 'streaming' || state.connection === 'connecting'
  const hasFailure = state.connection === 'disconnected' || state.connection === 'error'
  const bannerVisible = hasFailure && state.error !== null && dismissed !== state.error

  /**
   * Send, carrying any staged attachments.
   *
   * The attachment list is cleared once the turn is away: the ids now live in
   * the transcript, and leaving them staged would silently re-attach the same
   * files to the next unrelated message.
   */
  const sendWithAttachments = (message: string) => {
    send(withAttachments(message, attachments))
    setAttachments([])
  }

  /*
   * The draft is read from the send_email tool result the backend already
   * streamed, rather than re-parsed out of the model's prose. The tool result
   * is the authoritative record of what would be sent.
   */
  const draft = findDraft(state.messages)

  const onConfirm = () => {
    if (!draft) return

    // Routed through the same stream as any other turn. The backend re-runs the
    // tool with confirmed=true; the browser never talks to Gmail.
    send(
      `Yes, send it. to=${draft.to} subject="${draft.subject}" body="${draft.body}" confirmed=true`,
    )
  }

  const onCancel = () => {
    send("No, do not send that email. Cancel the send.")
  }

  const awaitingConfirm =
    draft !== null &&
    isStreaming === false &&
    !CONFIRM_PHRASES.test(state.messages.at(-1)?.text ?? '')

  return (
    <section className={styles.panel}>
      {bannerVisible && state.error ? (
        <ConnectionBanner
          message={state.error}
          onRetry={() => {
            setDismissed(null)
            retry()
          }}
          onDismiss={() => setDismissed(state.error)}
        />
      ) : null}

      <MessageList messages={state.messages} onSend={send} />

      {awaitingConfirm || state.metrics.email ? (
        <div className={styles.emailDock}>
          <EmailPreviewCard
            draft={awaitingConfirm ? draft : null}
            outcome={state.metrics.email}
            isStreaming={isStreaming}
            onConfirm={onConfirm}
            onCancel={onCancel}
          />
        </div>
      ) : null}

      <footer className={styles.footer}>
        <StreamMetrics connection={state.connection} metrics={state.metrics} />
        <AttachmentBar
          files={attachments}
          userId={DEFAULT_USER_ID}
          disabled={isStreaming}
          onChange={setAttachments}
        />
        <Composer
          onSend={sendWithAttachments}
          onStop={stop}
          isStreaming={isStreaming}
          statusText={state.statusText}
        />
      </footer>
    </section>
  )
}