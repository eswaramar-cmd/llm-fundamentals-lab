import { cx } from '../../lib/cx'
import styles from './Layout.module.css'

interface ThemeToggleProps {
  theme: 'light' | 'dark'
  onToggle: () => void
}

export function ThemeToggle({ theme, onToggle }: ThemeToggleProps) {
  const nextLabel = theme === 'dark' ? 'light' : 'dark'

  return (
    <button
      type="button"
      className={styles.iconButton}
      onClick={onToggle}
      aria-label={`Switch to ${nextLabel} theme`}
      title={`Switch to ${nextLabel} theme`}
    >
      {theme === 'dark' ? (
        <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
          <circle cx="12" cy="12" r="4.2" fill="currentColor" />
          <g stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
            <path d="M12 2.6v2.2M12 19.2v2.2M2.6 12h2.2M19.2 12h2.2" />
            <path d="M5.4 5.4l1.6 1.6M17 17l1.6 1.6M18.6 5.4L17 7M7 17l-1.6 1.6" />
          </g>
        </svg>
      ) : (
        <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
          <path
            d="M20 14.2A8.2 8.2 0 0 1 9.8 4a8.4 8.4 0 1 0 10.2 10.2z"
            fill="currentColor"
          />
        </svg>
      )}
    </button>
  )
}

interface StatusDotProps {
  connection: string
  className?: string
}

const DOT_TONE: Record<string, string> = {
  streaming: styles.dotLive,
  connecting: styles.dotBusy,
  done: styles.dotDone,
  error: styles.dotError,
  stopped: styles.dotIdle,
  disconnected: styles.dotError,
}

export function StatusDot({ connection, className }: StatusDotProps) {
  return (
    <span
      className={cx(styles.dot, DOT_TONE[connection] ?? styles.dotIdle, className)}
      aria-hidden="true"
    />
  )
}