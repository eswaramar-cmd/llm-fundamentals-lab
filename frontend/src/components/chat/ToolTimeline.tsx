import type { ToolCall } from '../../state/chatReducer'
import { Badge } from '../ui/Badge'
import { Spinner } from '../ui/Spinner'
import styles from './Chat.module.css'

function formatJson(value: Record<string, unknown>): string {
  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return String(value)
  }
}

/**
 * Renders tool calls as a native <details> list.
 *
 * Using disclosure elements instead of click handlers keeps expand/collapse
 * keyboard accessible and screen-reader labelled with no extra ARIA wiring.
 */
export function ToolTimeline({ tools }: { tools: ToolCall[] }) {
  if (tools.length === 0) return null

  return (
    <div className={styles.toolTimeline}>
      <div className={styles.toolHeader}>
        <span className={styles.toolHeaderTitle}>Tool activity</span>
        <Badge tone="neutral">{tools.length}</Badge>
      </div>

      <ul className={styles.toolList}>
        {tools.map((tool) => (
          <li key={tool.id} className={styles.toolItem}>
            <details className={styles.toolDetails}>
              <summary className={styles.toolSummary}>
                <span className={styles.toolTime}>+{tool.startedAt}ms</span>

                <span className={styles.toolName}>{tool.tool}</span>

                {tool.state === 'running' ? (
                  <Spinner size={12} label={`${tool.tool} running`} />
                ) : (
                  <Badge tone="success">{tool.durationMs ?? 0}ms</Badge>
                )}
              </summary>

              <div className={styles.toolBody}>
                <div className={styles.toolSection}>
                  <span className={styles.toolLabel}>input</span>
                  <pre className={styles.toolPre}>{formatJson(tool.input)}</pre>
                </div>

                {tool.result ? (
                  <div className={styles.toolSection}>
                    <span className={styles.toolLabel}>result</span>
                    <pre className={styles.toolPre}>{tool.result}</pre>
                  </div>
                ) : null}
              </div>
            </details>
          </li>
        ))}
      </ul>
    </div>
  )
}