/**
 * Canonical JWT storage for API calls (chat history, /api/ai/query persistence).
 * Call setCybrainAccessToken after login/register so Bearer auth is sent.
 */
const KEY = 'cybrain_auth'

export function setCybrainAccessToken(accessToken, extra = {}) {
  if (typeof window === 'undefined') return
  const token = String(accessToken || '').trim()
  if (!token) return
  const next = { ...extra, access_token: token }
  localStorage.setItem(KEY, JSON.stringify(next))
}

export function clearCybrainAccessToken() {
  if (typeof window === 'undefined') return
  localStorage.removeItem(KEY)
  localStorage.removeItem('access_token')
}

export function getCybrainAccessToken() {
  if (typeof window === 'undefined') return ''
  try {
    const raw = localStorage.getItem(KEY)
    if (raw) {
      const parsed = JSON.parse(raw)
      const t = parsed?.access_token || parsed?.token
      if (t && String(t).trim()) return String(t).trim()
    }
    const flat = localStorage.getItem('access_token')
    if (flat && String(flat).trim()) return String(flat).trim()
  } catch {
    // ignore
  }
  return ''
}
