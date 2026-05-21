import { useState, useEffect, useCallback } from 'react'
import { getSOPs, getDeviations } from '../api/editorApi'

/**
 * Fetches real-time counts for SOPs and Deviations from the backend.
 * Returns { sopCount, deviationCount } — both default to null while loading.
 * On error, returns 0 so the badge shows 0 rather than stale fake data.
 */
export function useSidebarCounts() {
  const [sopCount, setSopCount] = useState(null)
  const [deviationCount, setDeviationCount] = useState(null)
  const refreshEventName = 'sidebar-counts-refresh'

  const fetchCounts = useCallback(async () => {
    try {
      const [sops, devs] = await Promise.all([
        getSOPs().catch(() => []),
        getDeviations().catch(() => []),
      ])
      setSopCount(Array.isArray(sops) ? sops.length : 0)
      setDeviationCount(Array.isArray(devs) ? devs.length : 0)
    } catch {
      setSopCount(0)
      setDeviationCount(0)
    }
  }, [])

  useEffect(() => {
    let cancelled = false

    async function fetchCountsSafe() {
      try {
        const [sops, devs] = await Promise.all([
          getSOPs().catch(() => []),
          getDeviations().catch(() => []),
        ])
        if (cancelled) return
        setSopCount(Array.isArray(sops) ? sops.length : 0)
        setDeviationCount(Array.isArray(devs) ? devs.length : 0)
      } catch {
        if (!cancelled) {
          setSopCount(0)
          setDeviationCount(0)
        }
      }
    }

    fetchCountsSafe()

    const handleRefresh = () => {
      if (!cancelled) fetchCounts()
    }
    window.addEventListener(refreshEventName, handleRefresh)

    // Refresh every 60 seconds so the badges stay current
    const interval = setInterval(fetchCountsSafe, 60_000)
    return () => {
      cancelled = true
      clearInterval(interval)
      window.removeEventListener(refreshEventName, handleRefresh)
    }
  }, [fetchCounts, refreshEventName])

  return { sopCount, deviationCount }
}
