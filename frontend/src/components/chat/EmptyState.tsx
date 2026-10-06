import { useState } from 'react'
import { cx } from '../../lib/cx'
import styles from './Chat.module.css'

interface EmptyStateProps {
  suggestions: string[]
  onSelect: (value: string) => void
}

export function EmptyState({ suggestions, onSelect }: EmptyStateProps) {
  const [active, setActive] = useState<string | null>(null)

  return (
    <div className={styles.empty}>
      <h2 className={styles.emptyTitle}>Ask the research agent</h2>
      <p className={styles.emptyBody}>
        Answers stream token by token over Server-Sent Events. Tool calls appear inline as the
        agent works.
      </p>

      <ul className={styles.suggestions}>
        {suggestions.map((suggestion) => (
          <li key={suggestion}>
            <button
              type="button"
              className={cx(styles.suggestion, active === suggestion && styles.suggestionActive)}
              onMouseEnter={() => setActive(suggestion)}
              onMouseLeave={() => setActive(null)}
              onClick={() => onSelect(suggestion)}
            >
              {suggestion}
            </button>
          </li>
        ))}
      </ul>
    </div>
  )
}