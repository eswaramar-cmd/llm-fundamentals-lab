import { ChatPanel } from './components/chat/ChatPanel'
import { AppShell } from './components/layout/AppShell'
import { useChatStream } from './hooks/useChatStream'
import { useTheme } from './hooks/useTheme'

export default function App() {
  const { theme, toggle } = useTheme()

  // One stream per app. The hook is owned here rather than inside ChatPanel so
  // the header indicators and the panel always read the same connection state
  // instead of two independent streams.
  const stream = useChatStream()

  return (
    <AppShell
      connection={stream.state.connection}
      model={stream.state.metrics.model}
      theme={theme}
      canReset={stream.state.messages.length > 0}
      onToggleTheme={toggle}
      onReset={stream.reset}
    >
      <ChatPanel stream={stream} />
    </AppShell>
  )
}