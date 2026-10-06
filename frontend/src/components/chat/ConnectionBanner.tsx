import { Button } from '../ui/Button'
import styles from './Chat.module.css'

interface ConnectionBannerProps {
  message: string
  onRetry: () => void
  onDismiss?: () => void
}

/** Shown when the stream fails or the backend cannot be reached at all. */
export function ConnectionBanner({ message, onRetry, onDismiss }: ConnectionBannerProps) {
  return (
    <div className={styles.banner} role="alert">
      <div className={styles.bannerBody}>
        <strong className={styles.bannerTitle}>Stream interrupted</strong>
        <p className={styles.bannerText}>{message}</p>
      </div>

      <div className={styles.bannerActions}>
        <Button size="sm" variant="primary" onClick={onRetry}>
          Retry
        </Button>
        {onDismiss ? (
          <Button size="sm" variant="ghost" onClick={onDismiss}>
            Dismiss
          </Button>
        ) : null}
      </div>
    </div>
  )
}