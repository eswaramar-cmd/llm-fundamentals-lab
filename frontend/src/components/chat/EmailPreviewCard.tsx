import { useState } from 'react'
import { Badge } from '../ui/Badge'
import { Button } from '../ui/Button'
import { Spinner } from '../ui/Spinner'
import styles from './Chat.module.css'
import type { EmailOutcome } from '../../types/chat'
import type { EmailDraft } from '../../lib/email'

interface EmailPreviewCardProps {
  draft: EmailDraft | null
  outcome: EmailOutcome | null
  isStreaming: boolean
  /** Sends "yes, send it" as a new turn on the same thread. */
  onConfirm: () => void
  onCancel: () => void
}

/**
 * Confirmation gate for real outbound email.
 *
 * This never sends anything itself: Confirm just sends the user's agreement back
 * through the normal stream, and the backend tool performs the actual SMTP call.
 * The card only reflects what the backend reported, so "Sent" is never displayed
 * for a message that was not delivered.
 */
export function EmailPreviewCard({
  draft,
  outcome,
  isStreaming,
  onConfirm,
  onCancel,
}: EmailPreviewCardProps) {
  const [dismissed, setDismissed] = useState(false)

  if (outcome?.status === 'sent') {
    return (
      <div className={styles.emailSent} role="status">
        <Badge tone="success">sent</Badge>
        <div className={styles.emailSentBody}>
          <strong>Sent to {outcome.recipient}</strong>
          <span className={styles.emailDetail}>{outcome.detail}</span>
        </div>
      </div>
    )
  }

  if (outcome && outcome.status !== 'preview') {
    return (
      <div className={styles.emailPreview} role="status">
        <div className={styles.emailHeader}>
          <Badge tone="danger">{outcome.status}</Badge>
          <span className={styles.emailHeaderText}>Email could not be sent</span>
        </div>
        <pre className={styles.emailBody}>{outcome.detail}</pre>
      </div>
    )
  }

  /*
   * A preview is not a terminal state, it is a question. The backend reports
   * it as the `preview` outcome, so this branch used to be unreachable: the
   * `if (outcome)` check above swallowed it and rendered a read-only dump of
   * the tool text, which meant the Confirm button below never appeared and the
   * only way forward was to guess that a second message was expected.
   */
  if (!draft || dismissed) {
    if (!outcome) return null

    return (
      <div className={styles.emailPreview} role="status">
        <div className={styles.emailHeader}>
          <Badge tone="warning">preview</Badge>
          <span className={styles.emailHeaderText}>
            Nothing was sent yet
          </span>
        </div>
        <pre className={styles.emailBody}>{outcome.detail}</pre>
      </div>
    )
  }

  return (
    <div className={styles.emailPreview} role="alert">
      <div className={styles.emailHeader}>
        <Badge tone="warning">confirm</Badge>
        <span className={styles.emailHeaderText}>
          This will send a real email. Review before confirming.
        </span>
      </div>

      <dl className={styles.emailFields}>
        <dt>To</dt>
        <dd>{draft.to || <em>missing</em>}</dd>

        <dt>Subject</dt>
        <dd>{draft.subject || <em>missing</em>}</dd>

        <dt>Body</dt>
        <dd>
          <pre className={styles.emailBody}>{draft.body}</pre>
        </dd>
      </dl>

      <div className={styles.emailActions}>
        <Button variant="primary" onClick={onConfirm} disabled={isStreaming || !draft.to}>
          {isStreaming ? <Spinner size={13} /> : null}
          Confirm and send
        </Button>
        <Button variant="ghost" onClick={() => { setDismissed(true); onCancel() }}>
          Cancel
        </Button>
      </div>
    </div>
  )
}