import React, { useCallback, useEffect, useState } from 'react'
import {
  FileText,
  RefreshCcw,
  Save,
  CheckCircle2,
  Layers3,
  PencilLine,
  Eye,
  Edit3,
  Database,
  Shield,
  AlertTriangle,
  BookOpen,
  Tag,
  Users,
  Trash2,
} from 'lucide-react'
import {
  listClientProfiles,
  getClientProfile,
  getClientProfileMarkdown,
  saveManualClientProfileVersion,
  deleteProfileMd,
  deleteClientProfile,
} from '../api/profileApi'
import './ProfileWorkspacePage.css'

/* ─── tiny markdown → HTML converter (no external dep) ─────────────────── */
function markdownToHtml(md) {
  if (!md) return ''
  let html = md
    // headings
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^# (.+)$/gm, '<h1>$1</h1>')
    // bold + italic
    .replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    // inline code
    .replace(/`(.+?)`/g, '<code>$1</code>')
    // unordered list items
    .replace(/^[ \t]*- (.+)$/gm, '<li>$1</li>')
    // wrap consecutive <li> blocks in <ul>
    .replace(/(<li>.*<\/li>\n?)+/g, (block) => `<ul>${block}</ul>`)
    // horizontal rule
    .replace(/^---$/gm, '<hr />')
    // blank lines → paragraph breaks
    .replace(/\n\n+/g, '</p><p>')
  return `<p>${html}</p>`
}

function MarkdownPreview({ content }) {
  return (
    <div
      className="profile-md-preview"
      // eslint-disable-next-line react/no-danger
      dangerouslySetInnerHTML={{ __html: markdownToHtml(content) }}
    />
  )
}

/* ─── small stat badge ──────────────────────────────────────────────────── */
function StatBadge({ icon: Icon, label, value, variant = 'default' }) {
  return (
    <div className={`profile-stat-badge profile-stat-badge--${variant}`}>
      <Icon size={12} />
      <span className="profile-stat-badge__label">{label}:</span>
      <strong className="profile-stat-badge__value">{value || 'N/A'}</strong>
    </div>
  )
}

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

/* ═══════════════════════════════════════════════════════════════════════════
   MAIN COMPONENT
═══════════════════════════════════════════════════════════════════════════ */
export default function ProfileWorkspacePage() {
  const [profiles, setProfiles] = useState([])
  const [selectedProfileId, setSelectedProfileId] = useState('')
  const [selectedProfile, setSelectedProfile] = useState(null)
  const [profileMd, setProfileMd] = useState('')
  const [editorValue, setEditorValue] = useState('')
  const [viewMode, setViewMode] = useState('preview') // 'preview' | 'edit'
  const [changeReason, setChangeReason] = useState('')
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [confirmDelete, setConfirmDelete] = useState(false)
  const [notice, setNotice] = useState('')
  const [error, setError] = useState('')

  /* ── load list of all profiles (one per SOP) ── */
  const loadProfiles = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const rows = await listClientProfiles()
      setProfiles(rows)
      if (rows.length > 0) {
        setSelectedProfileId((prev) => prev || String(rows[0].id))
      }
    } catch (err) {
      setError(err.message || 'Failed to load profiles.')
    } finally {
      setLoading(false)
    }
  }, [])

  /* ── load one profile's full detail + active_profile_md ── */
  const loadProfileDetail = useCallback(async (profileId) => {
    if (!profileId) return
    setError('')
    try {
      const detail = await getClientProfile(profileId)
      setSelectedProfile(detail)

      let md = detail.active_profile_md || ''

      // fallback: dedicated profile.md endpoint
      if (!md) {
        try {
          const mdResp = await getClientProfileMarkdown(profileId)
          md = mdResp?.markdown || ''
        } catch {
          // ignore
        }
      }

      setProfileMd(md)
      setEditorValue(md)
    } catch (err) {
      setError(err.message || 'Failed to load profile detail.')
    }
  }, [])

  useEffect(() => { loadProfiles() }, [loadProfiles])

  useEffect(() => {
    if (selectedProfileId) loadProfileDetail(selectedProfileId)
  }, [selectedProfileId, loadProfileDetail])

  /* ── refresh ── */
  const handleRefresh = useCallback(async () => {
    if (!selectedProfileId) return
    await loadProfileDetail(selectedProfileId)
    setNotice('Profile refreshed from database.')
    setTimeout(() => setNotice(''), 3000)
  }, [selectedProfileId, loadProfileDetail])

  /* ── delete client profile completely ── */
  const handleDeleteProfile = useCallback(async () => {
    if (!selectedProfileId) return
    setDeleting(true)
    setError('')
    setNotice('')
    setConfirmDelete(false)
    try {
      await deleteClientProfile(selectedProfileId)
      setProfileMd('')
      setEditorValue('')
      setSelectedProfile(null)
      setSelectedProfileId('')
      setViewMode('preview')
      setNotice('Client profile has been completely deleted.')
      await loadProfiles()
      setTimeout(() => setNotice(''), 6000)
    } catch (err) {
      setError(err.message || 'Failed to delete client profile.')
    } finally {
      setDeleting(false)
    }
  }, [selectedProfileId, loadProfiles])

  /* ── save manual edit ── */
  const handleSaveManual = useCallback(async () => {
    if (!selectedProfileId || !editorValue.trim()) return
    setSaving(true)
    setError('')
    setNotice('')
    try {
      await saveManualClientProfileVersion(selectedProfileId, {
        profile_md: editorValue,
        rules_json: selectedProfile?.active_profile_json || {},
        change_reason: changeReason || 'Manual profile edit from workspace',
      })
      setProfileMd(editorValue)
      setChangeReason('')
      setViewMode('preview')
      setNotice('Profile saved successfully.')
      await loadProfiles()
      setTimeout(() => setNotice(''), 4000)
    } catch (err) {
      setError(err.message || 'Failed to save profile.')
    } finally {
      setSaving(false)
    }
  }, [selectedProfileId, editorValue, selectedProfile, changeReason, loadProfiles])

  /* ── derive quick-stats from active_profile_json ── */
  const pj = selectedProfile?.active_profile_json || {}
  const detectedDomains = (pj.detected_domains || []).join(', ') || 'N/A'
  const detectedDepts = (pj.detected_departments || []).join(', ') || 'N/A'
  const sopTypes = (pj.detected_sop_types || []).join(', ') || 'N/A'
  const formality = pj.preferred_style?.formality || 'N/A'
  const riskLevel = pj.risks_gaps?.risk_score?.level || 'unknown'
  const gapCount = pj.risks_gaps?.gap_count ?? 0
  const rolesDict = (pj.roles_raci?.roles && typeof pj.roles_raci.roles === 'object')
    ? pj.roles_raci.roles : (pj.roles_raci || {})
  const rolesCount = Object.values(rolesDict).filter(v => typeof v === 'object' && v?.detected).length
  const complianceStandards = (pj.compliance_elements?.standards_detected || [])
    .map(s => (typeof s === 'string' ? s : s?.standard)).filter(Boolean).join(', ') || 'None'

  const riskVariant = riskLevel === 'high' ? 'red' : riskLevel === 'medium' ? 'amber' : 'green'
  const gapVariant = gapCount > 5 ? 'red' : gapCount > 0 ? 'amber' : 'green'

  /* ── loading screen ── */
  if (loading) {
    return (
      <div className="profile-workspace-page">
        <div className="profile-empty-state" style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 36, marginBottom: 12 }}>⏳</div>
          Loading SOP profiles…
        </div>
      </div>
    )
  }

  /* ── main UI ── */
  return (
    <div className="profile-workspace-page">

      {/* Header */}
      <div className="profile-page-header">
        <div>
          <h1>SOP Profile Workspace</h1>
          <p>Each uploaded SOP auto-generates its own NLP-detected <code>profile.md</code> stored per SOP in the database.</p>
        </div>
        <div className="profile-header-stats">
          <div className="profile-stat-chip">
            <Layers3 size={15} />
            <span>{profiles.length} SOP Profile{profiles.length === 1 ? '' : 's'}</span>
          </div>
          {selectedProfile && (
            <div className="profile-stat-chip success">
              <CheckCircle2 size={15} />
              <span>{selectedProfile.total_sops_analyzed || 0} SOP{selectedProfile.total_sops_analyzed === 1 ? '' : 's'} analyzed</span>
            </div>
          )}
        </div>
      </div>

      {error  ? <div className="profile-banner profile-banner-error">{error}</div>   : null}
      {notice ? <div className="profile-banner profile-banner-success">{notice}</div> : null}

      <div className="profile-workspace-grid">

        {/* ── LEFT: SOP profile list ── */}
        <aside className="profile-list-panel">
          <div className="profile-panel-title">
            <FileText size={16} /> SOP Profiles
          </div>
          <div className="profile-list">
            {profiles.length ? profiles.map((profile) => {
              const isSelected = String(profile.id) === String(selectedProfileId)
              const hasProfile = Boolean(profile.active_profile_md)
              return (
                <button
                  key={profile.id}
                  type="button"
                  className={`profile-list-item${isSelected ? ' active' : ''}`}
                  onClick={() => setSelectedProfileId(String(profile.id))}
                >
                  <div className="profile-list-name">{profile.name}</div>
                  <div className="profile-list-meta">{profile.company_name || 'No company'}</div>
                  <div className="profile-list-submeta">
                    {profile.total_sops_analyzed || 0} SOP{profile.total_sops_analyzed === 1 ? '' : 's'} analyzed
                    {hasProfile
                      ? <span className="profile-ready-badge"> ✓ Profile ready</span>
                      : <span className="profile-pending-badge"> ⏳ Pending</span>}
                  </div>
                </button>
              )
            }) : (
              <div className="profile-panel-empty">
                No SOP profiles found.<br />
                Upload an SOP to generate its profile.
              </div>
            )}
          </div>
        </aside>

        {/* ── CENTER: profile.md viewer / editor ── */}
        <section className="profile-editor-panel">

          {/* Toolbar */}
          <div className="profile-editor-toolbar">
            <div className="profile-editor-title">
              <span>{selectedProfile?.name || 'No profile selected'}</span>
              {selectedProfile && (
                <small>
                  {selectedProfile.company_name || '—'} &nbsp;·&nbsp;
                  Updated: {formatDate(selectedProfile.updated_at || selectedProfile.created_at)}
                </small>
              )}
            </div>
            <div className="profile-editor-actions">
              <button
                type="button"
                className={`profile-action-btn ${viewMode === 'preview' ? 'primary' : 'secondary'}`}
                onClick={() => { setViewMode('preview'); setEditorValue(profileMd) }}
              >
                <Eye size={14} /> Preview
              </button>
              <button
                type="button"
                className={`profile-action-btn ${viewMode === 'edit' ? 'primary' : 'secondary'}`}
                onClick={() => setViewMode('edit')}
              >
                <Edit3 size={14} /> Edit
              </button>
              <button
                type="button"
                className="profile-action-btn secondary"
                onClick={handleRefresh}
                disabled={saving || deleting || !selectedProfileId}
              >
                <RefreshCcw size={14} /> Refresh
              </button>
              {viewMode === 'edit' && (
                <button
                  type="button"
                  className="profile-action-btn primary"
                  onClick={handleSaveManual}
                  disabled={saving || !editorValue.trim()}
                >
                  <Save size={14} /> {saving ? 'Saving…' : 'Save'}
                </button>
              )}
              {selectedProfile && (
                <button
                  id="delete-profile-btn"
                  type="button"
                  className="profile-action-btn danger"
                  onClick={() => setConfirmDelete(true)}
                  disabled={deleting || saving}
                  title="Delete Client Profile completely from the workspace"
                >
                  <Trash2 size={14} /> {deleting ? 'Deleting…' : 'Delete Profile'}
                </button>
              )}
            </div>
          </div>

          {/* Quick-stats bar */}
          {selectedProfile && (
            <div className="profile-stats-bar">
              <StatBadge icon={BookOpen}      label="Domain"     value={detectedDomains}           variant="blue"  />
              <StatBadge icon={Database}      label="Dept"       value={detectedDepts}             variant="default"/>
              <StatBadge icon={Tag}           label="Type"       value={sopTypes}                  variant="default"/>
              <StatBadge icon={PencilLine}    label="Formality"  value={formality}                 variant="green" />
              <StatBadge icon={Users}         label="Roles"      value={rolesCount}                variant="blue"  />
              <StatBadge icon={Shield}        label="Compliance" value={complianceStandards}       variant="green" />
              <StatBadge icon={AlertTriangle} label="Risk"       value={riskLevel.toUpperCase()}   variant={riskVariant} />
              <StatBadge icon={AlertTriangle} label="Gaps"       value={gapCount}                  variant={gapVariant}  />
            </div>
          )}

          {/* Change-reason input (edit mode only) */}
          {viewMode === 'edit' && (
            <div className="profile-editor-meta">
              <input
                className="profile-reason-input"
                value={changeReason}
                onChange={(e) => setChangeReason(e.target.value)}
                placeholder="Change reason (optional)"
              />
            </div>
          )}

          {/* Content area */}
          {!selectedProfile ? (
            <div className="profile-empty-state" style={{ margin: 16, textAlign: 'center' }}>
              Select an SOP profile from the sidebar.
            </div>
          ) : !profileMd ? (
            <div className="profile-empty-state" style={{ margin: 16, textAlign: 'center' }}>
              <div style={{ fontSize: 32, marginBottom: 10 }}>📄</div>
              No profile.md generated yet for this SOP.<br />
              <small>Re-upload the SOP to trigger NLP analysis.</small>
            </div>
          ) : viewMode === 'preview' ? (
            <MarkdownPreview content={profileMd} />
          ) : (
            <textarea
              className="profile-md-editor"
              value={editorValue}
              onChange={(e) => setEditorValue(e.target.value)}
              placeholder="Profile markdown content."
            />
          )}
        </section>

        {/* ── RIGHT: detected-parameters panel ── */}
        <aside className="profile-version-panel">
          <div className="profile-panel-title">
            <Database size={16} /> Detected Parameters
          </div>

          {selectedProfile ? (
            <div className="profile-params-body">

              {/* Roles */}
              <div className="profile-param-section">
                <div className="profile-param-heading">
                  <Users size={12} /> Roles Detected
                </div>
                {Object.entries(rolesDict)
                  .filter(([, v]) => typeof v === 'object' && v?.detected)
                  .map(([role, spec]) => (
                    <div key={role} className="profile-param-item profile-param-item--green">
                      ✓ {role}
                      <span className="profile-param-sub">({spec.raci_category || 'R'})</span>
                    </div>
                  ))}
                {rolesCount === 0 && <div className="profile-param-empty">None detected</div>}
              </div>

              {/* Compliance */}
              <div className="profile-param-section">
                <div className="profile-param-heading">
                  <Shield size={12} /> Compliance Standards
                </div>
                {(pj.compliance_elements?.standards_detected || []).map((s, i) => {
                  const name = typeof s === 'string' ? s : s?.standard
                  return name
                    ? <div key={i} className="profile-param-item profile-param-item--blue">✓ {name}</div>
                    : null
                })}
                {complianceStandards === 'None' && <div className="profile-param-empty">None detected</div>}
              </div>

              {/* Terminology */}
              <div className="profile-param-section">
                <div className="profile-param-heading">
                  <Tag size={12} /> Key Terms
                </div>
                {[
                  ...(pj.terminology?.acronyms || []).slice(0, 5).map(t => ({ t, type: 'acronym' })),
                  ...(pj.terminology?.controlled_terms || []).slice(0, 5).map(t => ({ t, type: 'controlled' })),
                ].map(({ t, type }, i) => (
                  <div key={i} className="profile-param-item profile-param-item--purple">
                    {t} <span className="profile-param-sub">({type})</span>
                  </div>
                ))}
                {!(pj.terminology?.acronyms?.length) && !(pj.terminology?.controlled_terms?.length) && (
                  <div className="profile-param-empty">None detected</div>
                )}
              </div>

              {/* Critical gaps */}
              <div className="profile-param-section">
                <div className="profile-param-heading">
                  <AlertTriangle size={12} /> Critical Gaps
                </div>
                {(pj.risks_gaps?.critical_focus_areas || []).map((area, i) => (
                  <div key={i} className="profile-param-item profile-param-item--amber">⚠ {area}</div>
                ))}
                {!(pj.risks_gaps?.critical_focus_areas?.length) && (
                  <div className="profile-param-empty">No critical gaps</div>
                )}
              </div>

              {/* Workflows */}
              <div className="profile-param-section">
                <div className="profile-param-heading">
                  <FileText size={12} /> Workflows Detected
                </div>
                {Object.entries(pj.workflow_patterns || {})
                  .filter(([, v]) => v?.detected)
                  .map(([wf]) => (
                    <div key={wf} className="profile-param-item profile-param-item--green">✓ {wf}</div>
                  ))}
                {!Object.values(pj.workflow_patterns || {}).some(v => v?.detected) && (
                  <div className="profile-param-empty">None detected</div>
                )}
              </div>

            </div>
          ) : (
            <div className="profile-panel-empty">
              Select a profile to see detected parameters.
            </div>
          )}

          <div className="profile-version-footer">
            <PencilLine size={14} />
            <span>Each SOP upload auto-generates a profile.md via NLP pipeline.</span>
          </div>
        </aside>

      </div>

      {/* ── Delete confirmation modal ── */}
      {confirmDelete && (
        <div className="profile-modal-overlay" role="dialog" aria-modal="true" aria-labelledby="delete-modal-title">
          <div className="profile-modal">
            <div className="profile-modal-icon"><Trash2 size={28} /></div>
            <h2 id="delete-modal-title" className="profile-modal-title">Delete Profile?</h2>
            <p className="profile-modal-body">
              This will permanently delete the client profile <strong>{selectedProfile?.name}</strong> 
              and all of its associated parameters from the workspace database.
              <br /><br />
              This action cannot be undone.
            </p>
            <div className="profile-modal-actions">
              <button
                id="delete-profile-confirm-btn"
                type="button"
                className="profile-action-btn danger"
                onClick={handleDeleteProfile}
              >
                <Trash2 size={14} /> Yes, delete it
              </button>
              <button
                id="delete-profile-cancel-btn"
                type="button"
                className="profile-action-btn secondary"
                onClick={() => setConfirmDelete(false)}
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
