import { Badge } from '../ui/Badge'
import { Button } from '../ui/Button'
import { StatusDot, ThemeToggle } from './ThemeToggle'
import styles from './Layout.module.css'
import type { ConnectionState } from '../../state/chatReducer'

const CONNECTION_LABEL: Record<ConnectionState, string> = {
  idle: 'Idle',
  connecting: 'Connecting',
  streaming: 'LIVE',
  done: 'Done',
  stopped: 'Stopped',
  error: 'Error',
  disconnected: 'Disconnected',
}

interface HeaderProps {
  connection: ConnectionState
  model: string | null
  theme: 'light' | 'dark'
  canReset: boolean
  onToggleTheme: () => void
  onReset: () => void
}

export function Header({
  connection,
  model,
  theme,
  canReset,
  onToggleTheme,
  onReset,
}: HeaderProps) {
  return (
    <header className={styles.header}>
      <div className={styles.brand}>
        <span className={styles.mark} aria-hidden="true" />
        <div className={styles.brandText}>
          <h1 className={styles.title}>Research Agent</h1>
          <p className={styles.subtitle}>LangGraph &middot; live SSE</p>
        </div>
      </div>

      <div className={styles.headerMeta}>
        {model ? (
          <Badge tone="accent" className={styles.model}>
            {model}
          </Badge>
        ) : null}

        <span className={styles.connection}>
          <StatusDot connection={connection} />
          {CONNECTION_LABEL[connection]}
        </span>

        <Button variant="ghost" size="sm" onClick={onReset} disabled={!canReset}>
          Clear
        </Button>

        <ThemeToggle theme={theme} onToggle={onToggleTheme} />
      </div>
    </header>
  )
}