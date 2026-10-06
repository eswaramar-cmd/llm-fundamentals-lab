import { cx } from '../../lib/cx'
import styles from './ui.module.css'

interface SpinnerProps {
  size?: number
  className?: string
  label?: string
}

export function Spinner({ size = 14, className, label = 'Loading' }: SpinnerProps) {
  return (
    <span
      className={cx(styles.spinner, className)}
      style={{ width: size, height: size }}
      role="status"
      aria-label={label}
    />
  )
}