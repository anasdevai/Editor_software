const API_BASE = import.meta.env.VITE_API_BASE || ''

async function throwApiError(res, fallback) {
  let detail = fallback
  try {
    const data = await res.json()
    detail = data?.detail || data?.message || fallback
  } catch {
    // ignore
  }
  throw new Error(detail)
}

export async function listClientProfiles() {
  const res = await fetch(`${API_BASE}/api/client-profiles`)
  if (!res.ok) await throwApiError(res, 'Failed to load client profiles')
  return res.json()
}

export async function getClientProfile(profileId) {
  const res = await fetch(`${API_BASE}/api/client-profiles/${profileId}`)
  if (!res.ok) await throwApiError(res, 'Failed to load client profile')
  return res.json()
}

export async function getClientProfileByName(profileName) {
  const res = await fetch(`${API_BASE}/api/client-profiles/by-name/${encodeURIComponent(profileName)}`)
  if (!res.ok) await throwApiError(res, 'Failed to load client profile by name')
  return res.json()
}

export async function listClientProfileVersions(profileId) {
  const res = await fetch(`${API_BASE}/api/client-profiles/${profileId}/versions`)
  if (!res.ok) await throwApiError(res, 'Failed to load profile versions')
  return res.json()
}

export async function getClientProfileVersion(profileId, versionId) {
  const res = await fetch(`${API_BASE}/api/client-profiles/${profileId}/versions/${versionId}`)
  if (!res.ok) await throwApiError(res, 'Failed to load profile version')
  return res.json()
}

export async function getClientProfileMarkdown(profileId) {
  const res = await fetch(`${API_BASE}/api/client-profiles/${profileId}/profile.md`)
  if (!res.ok) await throwApiError(res, 'Failed to load profile markdown')
  return res.json()
}

export async function activateClientProfileVersion(profileId, versionId, payload = {}) {
  const res = await fetch(`${API_BASE}/api/client-profiles/${profileId}/versions/${versionId}/activate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) await throwApiError(res, 'Failed to activate profile version')
  return res.json()
}

export async function saveManualClientProfileVersion(profileId, payload) {
  const res = await fetch(`${API_BASE}/api/client-profiles/${profileId}/versions/manual`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) await throwApiError(res, 'Failed to save manual profile version')
  return res.json()
}
