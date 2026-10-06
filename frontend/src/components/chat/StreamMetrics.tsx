import { useEffect, useState } from 'react'
import { StatusDot } from '../layout/ThemeToggle'
import type { ConnectionState, Metrics } from '../../state/chatReducer'
import styles from './Chat.module.css'

interface StreamMetricsProps {
  connection: ConnectionState
  metrics: Metrics
}

function formatDuration(ms: number | null): string {
  if (ms === null) return '--'
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)}s` : `${ms}ms`
}

/**
 * Mounted only while a stream is open, so the interval and its zeroed counter
 * are created and torn down with the stream instead of being reset by hand.
 */
function LiveTimer() {
  const [elapsed, setElapsed] = useState(0)

  useEffect(() => {
    const startedAt = Date.now()
    const id = window.setInterval(() => setElapsed(Date.now() - startedAt), 1000)

    return () => window.clearInterval(id)
  }, [])

  return <span className={styles.metricValue}>{formatDuration(elapsed)}</span>
}

export function StreamMetrics({ connection, metrics }: StreamMetricsProps) {
  const isStreaming = connection === 'streaming' || connection === 'connecting'

  return (
    <div className={styles.metrics}>
      <span className={styles.metric}>
        <StatusDot connection={connection} />
        {isStreaming ? 'LIVE' : connection.toUpperCase()}
      </span>

      <span className={styles.metricItem}>
        <span className={styles.metricLabel}>latency</span>
        {isStreaming ? (
          <LiveTimer />
        ) : (
          <span className={styles.metricValue}>{formatDuration(metrics.latencyMs)}</span>
        )}
      </span>

      <span className={styles.metricItem}>
        <span className={styles.metricLabel}>ttft</span>
        <span className={styles.metricValue}>
          {isStreaming ? '--' : formatDuration(metrics.ttftMs)}
        </span>
      </span>

      <span className={styles.metricItem}>
        <span className={styles.metricLabel}>chars</span>
        <span className={styles.metricValue}>{metrics.chars || '--'}</span>
      </span>

      <span className={styles.metricItem}>
        <span className={styles.metricLabel}>chars/s</span>
        <span className={styles.metricValue}>{metrics.charsPerSecond ?? '--'}</span>
      </span>

      <span className={styles.metricItem}>
        <span className={styles.metricLabel}>model</span>
        <span className={styles.metricValue}>{metrics.model ?? '--'}</span>
      </span>
    </div>
  )
}