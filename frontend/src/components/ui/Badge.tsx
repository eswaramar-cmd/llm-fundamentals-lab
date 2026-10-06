import type { ReactNode } from 'react'
import { cx } from '../../lib/cx'
import styles from './ui.module.css'

type Tone = 'neutral' | 'accent' | 'success' | 'warning' | 'danger'

interface BadgeProps {
  tone?: Tone
  children: ReactNode
  className?: string
}

export function Badge({ tone = 'neutral', children, className }: BadgeProps) {
  return <span className={cx(styles.badge, styles[`tone_${tone}`], className)}>{children}</span>
}