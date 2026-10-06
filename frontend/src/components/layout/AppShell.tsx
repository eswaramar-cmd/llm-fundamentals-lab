import type { ReactNode } from 'react'
import { Header } from './Header'
import styles from './Layout.module.css'
import type { ConnectionState } from '../../state/chatReducer'

interface AppShellProps {
  connection: ConnectionState
  model: string | null
  theme: 'light' | 'dark'
  canReset: boolean
  onToggleTheme: () => void
  onReset: () => void
  children: ReactNode
}

export function AppShell({
  connection,
  model,
  theme,
  canReset,
  onToggleTheme,
  onReset,
  children,
}: AppShellProps) {
  return (
    <div className={styles.shell}>
      <Header
        connection={connection}
        model={model}
        theme={theme}
        canReset={canReset}
        onToggleTheme={onToggleTheme}
        onReset={onReset}
      />
      <main className={styles.main}>{children}</main>
    </div>
  )
}