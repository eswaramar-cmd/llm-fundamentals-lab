import { useAutoScroll } from '../../hooks/useAutoScroll'
import type { Message } from '../../state/chatReducer'
import { EmptyState } from './EmptyState'
import { MessageBubble } from './MessageBubble'
import styles from './Chat.module.css'

interface MessageListProps {
  messages: Message[]
  onSend: (message: string) => void
}

const SUGGESTIONS = [
  'What does this project do?',
  'Summarise the retrieval pipeline',
  'Which tools are available to the agent?',
]

export function MessageList({ messages, onSend }: MessageListProps) {
  // Re-pinning on the trailing text length keeps the view glued to the newest
  // token without re-rendering the list on every frame.
  const tail = messages.at(-1)
  const { ref, onScroll } = useAutoScroll<HTMLDivElement>(
    `${messages.length}:${tail?.text.length ?? 0}`,
  )

  return (
    <div className={styles.scroll} ref={ref} onScroll={onScroll}>
      {messages.length === 0 ? (
        <EmptyState suggestions={SUGGESTIONS} onSelect={onSend} />
      ) : (
        <div className={styles.messageStack}>
          {messages.map((message) => (
            <MessageBubble key={message.id} message={message} />
          ))}
        </div>
      )}
    </div>
  )
}