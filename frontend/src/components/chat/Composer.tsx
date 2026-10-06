import { useEffect, useRef, useState } from 'react'
import type { KeyboardEvent } from 'react'
import { Button } from '../ui/Button'
import { Spinner } from '../ui/Spinner'
import styles from './Chat.module.css'

interface ComposerProps {
  onSend: (message: string) => void
  onStop: () => void
  isStreaming: boolean
  statusText: string | null
  disabled?: boolean
}

const MAX_ROWS_GROWTH = 200

export function Composer({
  onSend,
  onStop,
  isStreaming,
  statusText,
  disabled = false,
}: ComposerProps) {
  const [value, setValue] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement | null>(null)

  // Grow with content up to a ceiling, then scroll internally.
  useEffect(() => {
    const node = textareaRef.current
    if (!node) return

    node.style.height = 'auto'
    node.style.height = `${Math.min(node.scrollHeight, MAX_ROWS_GROWTH)}px`
  }, [value])

  const submit = () => {
    const trimmed = value.trim()
    if (!trimmed || isStreaming || disabled) return

    onSend(trimmed)
    setValue('')
  }

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    // Enter sends; Shift+Enter inserts a newline. Guard ime composition so a
    // Japanese or Chinese candidate is not submitted mid-selection.
    if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault()
      submit()
    }
  }

  return (
    <div className={styles.composer}>
      {statusText ? (
        <div className={styles.statusLine}>
          <Spinner size={12} />
          <span>{statusText}</span>
        </div>
      ) : null}

      <div className={styles.composerRow}>
        <textarea
          ref={textareaRef}
          className={styles.textarea}
          rows={1}
          value={value}
          placeholder="Ask a question, or press Enter to send"
          aria-label="Message"
          disabled={disabled}
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={onKeyDown}
        />

        {isStreaming ? (
          <Button variant="danger" onClick={onStop}>
            Stop
          </Button>
        ) : (
          <Button variant="primary" onClick={submit} disabled={!value.trim() || disabled}>
            Send
          </Button>
        )}
      </div>
    </div>
  )
}