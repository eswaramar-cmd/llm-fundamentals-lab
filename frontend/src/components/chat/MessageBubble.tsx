import { memo } from 'react'
import type { Message } from '../../state/chatReducer'
import { Badge } from '../ui/Badge'
import { Spinner } from '../ui/Spinner'
import { ToolTimeline } from './ToolTimeline'
import { MarkdownRenderer } from './MarkdownRenderer'
import styles from './Chat.module.css'

const TIME_FORMAT = new Intl.DateTimeFormat(undefined, {
  hour: '2-digit',
  minute: '2-digit',
  second: '2-digit',
})

const STATUS_TONE = {
  streaming: 'accent',
  complete: 'success',
  stopped: 'warning',
  error: 'danger',
} as const

const STATUS_LABEL = {
  streaming: 'streaming',
  complete: 'done',
  stopped: 'stopped',
  error: 'failed',
} as const

/**
 * Memoised because the streaming bubble's `text` changes on every chunk. Without
 * this, each chunk re-renders every message in the transcript, which shows up as
 * visible jank once an answer is long. Only the bubble that actually changed
 * re-renders now.
 */
export const MessageBubble = memo(
  function MessageBubble({ message }: { message: Message }) {
    const isUser = message.role === 'user'
    const isStreaming = message.status === 'streaming'

    return (
      <article className={`${styles.message} ${isUser ? styles.user : styles.assistant}`}>
        <div className={styles.messageMeta}>
          <span className={styles.messageRole}>{isUser ? 'You' : 'Agent'}</span>
          <time
            className={styles.messageTime}
            dateTime={new Date(message.createdAt).toISOString()}
          >
            {TIME_FORMAT.format(message.createdAt)}
          </time>
          {!isUser && message.status !== 'streaming' ? (
            <Badge tone={STATUS_TONE[message.status]}>{STATUS_LABEL[message.status]}</Badge>
          ) : null}
        </div>

        {isUser ? (
          /* User messages: plain pre-wrap, no markdown needed */
          <p className={styles.userText}>{message.text}</p>
        ) : (
          <div className={styles.assistantBody}>
            <ToolTimeline tools={message.tools} />

            {message.text ? (
              /* Agent messages: render as structured markdown */
              <MarkdownRenderer content={message.text} />
            ) : isStreaming ? (
              <div className={styles.waiting}>
                <Spinner size={14} />
                <span>Waiting for first token…</span>
              </div>
            ) : null}
          </div>
        )}
      </article>
    )
  },
)