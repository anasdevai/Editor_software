import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { History, FileText, RefreshCcw, Save, CheckCircle2, Layers3, PencilLine } from 'lucide-react'
import {
  activateClientProfileVersion,
  getClientProfileVersion,
  listClientProfiles,
  listClientProfileVersions,
  saveManualClientProfileVersion,
} from '../api/profileApi'
import './ProfileWorkspacePage.css'

function formatDate(value) {
  if (!value) return ''
  const d = new Date(value)
  if (Number.isNaN(d.getTime())) return ''
  return d.toLocaleString('en-GB', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export default function ProfileWorkspacePage() {
  const [profiles, setProfiles] = useState([])
  const [selectedProfileId, setSelectedProfileId] = useState('')
  const [versions, setVersions] = useState([])
  const [selectedVersionId, setSelectedVersionId] = useState('')
  const [selectedVersion, setSelectedVersion] = useState(null)
  const [editorValue, setEditorValue] = useState('')
  const [changeReason, setChangeReason] = useState('')
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [notice, setNotice] = useState('')
  const [error, setError] = useState('')

  const selectedProfile = useMemo(
    () => profiles.find((item) => String(item.id) === String(selectedProfileId)) || null,
    [profiles, selectedProfileId],
  )
  const activeVersionNumber = useMemo(() => {
    const activeRow = versions.find((row) => String(row.id) === String(selectedProfile?.current_version_id || ''))
    return activeRow?.version_number || null
  }, [versions, selectedProfile])

  const loadProfiles = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const profileRows = await listClientProfiles()
      setProfiles(profileRows)
      const firstProfileId = profileRows?.[0]?.id || ''
      setSelectedProfileId((prev) => prev || firstProfileId)
    } catch (err) {
      setError(err.message || 'Failed to load profiles.')
    } finally {
      setLoading(false)
    }
  }, [])

  const loadVersions = useCallback(async (profileId) => {
    if (!profileId) return
    setError('')
    try {
      const rows = await listClientProfileVersions(profileId)
      setVersions(rows)
      const activeVersionId =
        profiles.find((p) => String(p.id) === String(profileId))?.current_version_id || ''
      const preferredVersionId =
        rows.find((row) => String(row.id) === String(activeVersionId))?.id
        || rows[0]?.id
        || ''
      setSelectedVersionId(preferredVersionId)
    } catch (err) {
      setError(err.message || 'Failed to load profile versions.')
    }
  }, [profiles])

  const loadVersionDetail = useCallback(async (profileId, versionId) => {
    if (!profileId || !versionId) return
    setError('')
    try {
      const detail = await getClientProfileVersion(profileId, versionId)
      setSelectedVersion(detail)
      setEditorValue(detail.profile_md || '')
    } catch (err) {
      setError(err.message || 'Failed to load selected profile version.')
    }
  }, [])

  useEffect(() => {
    loadProfiles()
  }, [loadProfiles])

  useEffect(() => {
    if (selectedProfileId) {
      loadVersions(selectedProfileId)
    }
  }, [selectedProfileId, loadVersions])

  useEffect(() => {
    if (selectedProfileId && selectedVersionId) {
      loadVersionDetail(selectedProfileId, selectedVersionId)
    }
  }, [selectedProfileId, selectedVersionId, loadVersionDetail])

  const handleActivateVersion = useCallback(async () => {
    if (!selectedProfileId || !selectedVersionId) return
    setSaving(true)
    setError('')
    setNotice('')
    try {
      await activateClientProfileVersion(selectedProfileId, selectedVersionId, {
        change_reason: changeReason || 'Activated from profile workspace',
      })
      await loadProfiles()
      await loadVersions(selectedProfileId)
      setNotice('Selected profile version is now active and will be used by editor actions.')
    } catch (err) {
      setError(err.message || 'Failed to activate version.')
    } finally {
      setSaving(false)
    }
  }, [selectedProfileId, selectedVersionId, changeReason, loadProfiles, loadVersions])

  const handleSaveManualVersion = useCallback(async () => {
    if (!selectedProfileId || !editorValue.trim()) return
    setSaving(true)
    setError('')
    setNotice('')
    try {
      await saveManualClientProfileVersion(selectedProfileId, {
        profile_md: editorValue,
        rules_json: selectedVersion?.rules_json || {},
        change_reason: changeReason || 'Manual profile edit from workspace',
      })
      await loadProfiles()
      await loadVersions(selectedProfileId)
      setNotice('Manual profile edit saved as a new active version.')
      setChangeReason('')
    } catch (err) {
      setError(err.message || 'Failed to save manual profile version.')
    } finally {
      setSaving(false)
    }
  }, [selectedProfileId, editorValue, selectedVersion, changeReason, loadProfiles, loadVersions])

  if (loading) {
    return <div className="profile-workspace-page"><div className="profile-empty-state">Loading profiles...</div></div>
  }

  return (
    <div className="profile-workspace-page">
      <div className="profile-page-header">
        <div>
          <h1>Profile Workspace</h1>
          <p>View, activate, edit, and version `profile.md` for SOP rewrite/improve actions.</p>
        </div>
        <div className="profile-header-stats">
          <div className="profile-stat-chip">
            <Layers3 size={15} />
            <span>{profiles.length} Profile{profiles.length === 1 ? '' : 's'}</span>
          </div>
          <div className="profile-stat-chip">
            <History size={15} />
            <span>{versions.length} Version{versions.length === 1 ? '' : 's'}</span>
          </div>
          <div className="profile-stat-chip success">
            <CheckCircle2 size={15} />
            <span>Active v{activeVersionNumber || '—'}</span>
          </div>
        </div>
      </div>

      {error ? <div className="profile-banner profile-banner-error">{error}</div> : null}
      {notice ? <div className="profile-banner profile-banner-success">{notice}</div> : null}

      <div className="profile-workspace-grid">
        <aside className="profile-list-panel">
          <div className="profile-panel-title"><FileText size={16} /> Profiles</div>
          <div className="profile-list">
            {profiles.length ? profiles.map((profile) => {
              const isActive = String(profile.current_version_id || '') === String(selectedVersionId || '')
              return (
                <button
                  key={profile.id}
                  type="button"
                  className={`profile-list-item${String(profile.id) === String(selectedProfileId) ? ' active' : ''}`}
                  onClick={() => setSelectedProfileId(profile.id)}
                >
                  <div className="profile-list-name">{profile.name}</div>
                  <div className="profile-list-meta">
                    {profile.company_name || 'No company name'}
                    {isActive ? ' • Active version selected' : ''}
                  </div>
                  <div className="profile-list-submeta">
                    {profile.total_sops_analyzed || 0} SOPs analyzed
                  </div>
                </button>
              )
            }) : (
              <div className="profile-panel-empty">
                No profiles found yet.
              </div>
            )}
          </div>
        </aside>

        <section className="profile-editor-panel">
          <div className="profile-editor-toolbar">
            <div className="profile-editor-title">
              <span>{selectedProfile?.name || 'No profile selected'}</span>
              {selectedVersion ? (
                <small>
                  Version {selectedVersion.version_number} • {formatDate(selectedVersion.created_at)}
                </small>
              ) : (
                <small>Select a profile version to load its `profile.md`.</small>
              )}
            </div>
            <div className="profile-editor-actions">
              <button type="button" className="profile-action-btn secondary" onClick={handleActivateVersion} disabled={saving || !selectedVersionId}>
                <RefreshCcw size={14} />
                Use Selected Version
              </button>
              <button type="button" className="profile-action-btn primary" onClick={handleSaveManualVersion} disabled={saving || !editorValue.trim()}>
                <Save size={14} />
                Save as New Version
              </button>
            </div>
          </div>

          <div className="profile-editor-summary">
            <div className="profile-summary-card">
              <span className="profile-summary-label">Current active version</span>
              <strong>v{activeVersionNumber || '—'}</strong>
            </div>
            <div className="profile-summary-card">
              <span className="profile-summary-label">Selected version</span>
              <strong>v{selectedVersion?.version_number || '—'}</strong>
            </div>
            <div className="profile-summary-card wide">
              <span className="profile-summary-label">Selected change reason</span>
              <strong>{selectedVersion?.change_reason || 'No version selected'}</strong>
            </div>
          </div>

          <div className="profile-editor-meta">
            <input
              className="profile-reason-input"
              value={changeReason}
              onChange={(e) => setChangeReason(e.target.value)}
              placeholder="Change reason for activation or manual edit"
            />
          </div>

          <textarea
            className="profile-md-editor"
            value={editorValue}
            onChange={(e) => setEditorValue(e.target.value)}
            placeholder="Profile markdown content will appear here."
          />
        </section>

        <aside className="profile-version-panel">
          <div className="profile-panel-title"><History size={16} /> Version History</div>
          <div className="profile-version-list">
            {versions.length ? versions.map((version) => {
              const isSelected = String(version.id) === String(selectedVersionId)
              const isCurrent = String(version.id) === String(selectedProfile?.current_version_id || '')
              return (
                <button
                  key={version.id}
                  type="button"
                  className={`profile-version-item${isSelected ? ' selected' : ''}`}
                  onClick={() => setSelectedVersionId(version.id)}
                >
                  <div className="profile-version-line">
                    <strong>v{version.version_number}</strong>
                    {isCurrent ? <span className="profile-version-badge">Active</span> : null}
                  </div>
                  <div className="profile-version-reason">{version.change_reason || 'No reason'}</div>
                  <div className="profile-version-date">{formatDate(version.created_at)}</div>
                </button>
              )
            }) : (
              <div className="profile-panel-empty">
                No versions available for this profile yet.
              </div>
            )}
          </div>
          <div className="profile-version-footer">
            <PencilLine size={14} />
            <span>Manual edits create a new version and update the active DB profile.</span>
          </div>
        </aside>
      </div>
    </div>
  )
}
