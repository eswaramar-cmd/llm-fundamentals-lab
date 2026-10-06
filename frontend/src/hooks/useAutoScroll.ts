import { useCallback, useEffect, useRef } from 'react'

/**
 * Keeps a scroll container pinned to the bottom while new content streams in,
 * but stops fighting the user the moment they scroll up to read earlier output.
 */
export function useAutoScroll<T extends HTMLElement>(dependency: unknown) {
  const ref = useRef<T | null>(null)
  const pinnedRef = useRef(true)

  const onScroll = useCallback(() => {
    const node = ref.current
    if (!node) return

    const distanceFromBottom = node.scrollHeight - node.scrollTop - node.clientHeight
    pinnedRef.current = distanceFromBottom < 80
  }, [])

  useEffect(() => {
    const node = ref.current
    if (!node || !pinnedRef.current) return

    node.scrollTop = node.scrollHeight
  }, [dependency])

  return { ref, onScroll }
}