import { useCallback, useSyncExternalStore } from 'react'

/**
 * Subscribes to a CSS media query.
 *
 * `useSyncExternalStore` is the intended fit here: matchMedia is an external
 * store, so React can read the current value during render and only re-render
 * when the browser fires a real change event.
 */
export function useMediaQuery(query: string): boolean {
  const subscribe = useCallback(
    (onStoreChange: () => void) => {
      const list = window.matchMedia(query)
      list.addEventListener('change', onStoreChange)

      return () => list.removeEventListener('change', onStoreChange)
    },
    [query],
  )

  const getSnapshot = useCallback(() => window.matchMedia(query).matches, [query])

  const getServerSnapshot = useCallback(() => false, [])

  return useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot)
}