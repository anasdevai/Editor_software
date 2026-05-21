import React, { useState, useEffect, useCallback, useMemo, lazy, Suspense } from 'react'
import {
  Search, Plus, ChevronDown, ChevronUp, ArrowLeft,
  Sparkles, ExternalLink, Edit3, Filter, Download,
  AlertCircle, FileText, X, FileEdit, List, Loader, Upload, Trash2
} from 'lucide-react'
import { getSOPs, getDocument, queryAI, createDocument, deleteDocument, getChatSessionMessages } from '../api/editorApi'
import {
  buildSOPDisplayLabel,
  prepareNewSOPImport,
  SOP_IMPORT_ACCEPT,
} from '../utils/sopImportService'
import SOPTable from '../components/SOPs/SOPTable'
import StatusBadge from '../components/Common/StatusBadge'
import { getKLAssistantContext } from '../utils/assistantContext'
import './SOPsPage.css'
const EditorPage = lazy(() => import('./EditorPage'))
const KL_WORKSPACE_CONTEXT_KEY = 'kl_assistant_workspace_state_v2'

// ── Quick filter suggestions (UI labels only — not mock data) ──────────────
const quickFilters = [
  'Welche SOPs haben die meisten Abweichungen?',
  'Welche SOP wird am häufigsten verletzt?',
  'Was sind meine kritischen CAPA-Maßnahmen?',
  'Führe mir die Audit-Findings zusammen',
]

// ── Sub-components ─────────────────────────────────────────────────────────

function CategoryBadge({ category }) {
  return <span className="sop-cat-badge sop-cat-blue">{category || 'Quality'}</span>
}

function SOPCard({ sop, onOpen, onOpenNewTab, onEdit, onDelete }) {
  return (
    <div className="sop-context-card">
      <div className="sop-card-header">
        <span className="sop-card-code">{sop.sop_number}</span>
        <StatusBadge status={sop.status} />
      </div>
      <h3 className="sop-card-title">{sop.title}</h3>
      {sop.department && (
        <p className="sop-card-desc">{sop.department}</p>
      )}

      <div className="sop-card-meta">
        <CategoryBadge category={sop.department} />
        {sop.version_number && (
          <span className="sop-card-meta-pill sop-meta-muted">
            v{sop.version_number}
          </span>
        )}
      </div>

      <div className="sop-card-actions">
        <button className="sop-card-btn sop-card-btn-primary" onClick={() => onOpen(sop)}>
          Öffnen
        </button>
        <button 
          className="sop-card-btn sop-card-btn-ghost" 
          onClick={() => onOpenNewTab(sop)}
          title="In neuem Tab öffnen"
        >
          <ExternalLink size={13} />
        </button>
        <button className="sop-card-btn sop-card-btn-ghost" onClick={() => onEdit(sop)}>
          <Edit3 size={13} /> Bearbeiten
        </button>
        <button 
          className="sop-card-btn sop-card-btn-ghost" 
          onClick={(e) => {
            e.stopPropagation();
            onDelete(sop);
          }}
          title="SOP löschen"
          style={{ color: 'var(--status-urgent)' }}
        >
          <Trash2 size={13} />
        </button>
      </div>
    </div>
  )
}

function KISummary({ open, onToggle, query, summaryText, sources, loading, error, historyRows = [] }) {
  return (
    <div className={`sops-ki-summary ${open ? 'sops-ki-open' : ''}`}>
      <button className="ki-summary-header" onClick={onToggle}>
        <div className="ki-header-left">
          <Sparkles size={14} className="ki-sparkle" />
          <span className="ki-title">KI-Zusammenfassung</span>
          <span className="ki-subtitle">„{query || 'Welche SOPs haben die meisten Abweichungen?'}"</span>
        </div>
        <div className="ki-header-right">
          {open ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </div>
      </button>

      {open && (
        <div className="ki-summary-body">
          {historyRows.length > 0 ? (
            <div className="ki-transcript" style={{ marginBottom: 12, maxHeight: 220, overflowY: 'auto', fontSize: 13 }}>
              {historyRows.map((row) => (
                <div
                  key={row.id}
                  style={{
                    marginBottom: 8,
                    padding: 8,
                    borderRadius: 6,
                    background: row.role === 'user' ? 'var(--surface-2, #f4f4f5)' : 'var(--surface-1, #fafafa)',
                  }}
                >
                  <strong>{row.role === 'user' ? 'Sie' : 'KI'}:</strong> {row.content}
                </div>
              ))}
            </div>
          ) : null}
          {loading ? <p className="ki-summary-text">KI analysiert Kontext...</p> : null}
          {error ? <p className="ki-summary-text" style={{ color: 'var(--error)' }}>{error}</p> : null}
          {!loading && !error ? (
            <p className="ki-summary-text" style={{ color: 'var(--text-muted)', fontSize: 13 }}>
              {summaryText || 'Frage stellen, um eine KI-Zusammenfassung aus dem Backend zu laden.'}
            </p>
          ) : null}
          {!loading && sources?.length > 0 ? (
            <div className="ki-summary-source-list">
              {sources.slice(0, 5).map((src, idx) => (
                <span key={`${src?.id || src?.label || 'src'}-${idx}`} className="ki-summary-source-chip">
                  {src?.label || src?.id || `Quelle ${idx + 1}`}
                </span>
              ))}
            </div>
          ) : null}
          <div className="ki-summary-actions">
            <button className="ki-action-btn">
              <Download size={13} /> Exportieren
            </button>
            <button className="ki-action-btn">Weitere Fragen</button>
            <button className="ki-action-btn ki-action-primary">SOPs öffnen</button>
          </div>
        </div>
      )}
    </div>
  )
}

// ── Map raw /api/sops record into a display-ready shape ───────────────────
function mapSOP(s) {
  const cv = s.current_version
  return {
    id: String(s.id),
    sop_number: s.sop_number || String(s.id).slice(0, 8),
    title: s.title || 'Untitled',
    department: s.department || '',
    version_number: cv?.version_number || '1',
    status: cv?.external_status || 'draft',
    // For the table view
    code: s.sop_number || String(s.id).slice(0, 8),
    version: cv?.version_number ? `V ${cv.version_number}` : 'V 1',
    date: s.updated_at ? new Date(s.updated_at).toLocaleDateString('de-DE') : '—',
    owner: cv?.metadata_json?.sopMetadata?.author || s.department || 'System',
    is_active: s.is_active,
    updated_at_raw: s.updated_at || null, // kept for client-side sorting
    current_version: cv || null,
  }
}

function buildInitialSOPMetadata(sop) {
  if (!sop || typeof sop !== 'object') return null

  const cv = sop.current_version || {}
  const rawMetadata = cv.metadata_json && typeof cv.metadata_json === 'object'
    ? cv.metadata_json
    : {}
  const rawSopMetadata = rawMetadata.sopMetadata && typeof rawMetadata.sopMetadata === 'object'
    ? rawMetadata.sopMetadata
    : rawMetadata

  const normalizedSopMetadata = {
    ...rawSopMetadata,
    documentId: rawSopMetadata.documentId
      || rawSopMetadata.sop_id
      || rawSopMetadata.sopId
      || rawSopMetadata.document_id
      || sop.sop_number
      || sop.code
      || sop.id,
    title: rawSopMetadata.title || sop.title || '',
    sopVersion: rawSopMetadata.sopVersion
      || rawSopMetadata.version
      || rawSopMetadata.document_revision
      || rawSopMetadata.revision
      || cv.version_number
      || '',
    docType: rawSopMetadata.docType
      || rawSopMetadata.doc_type
      || rawSopMetadata.documentType
      || sop.document_type
      || '',
    category: rawSopMetadata.category || sop.category || '',
    department: rawSopMetadata.department || sop.department || '',
    author: rawSopMetadata.author || '',
    reviewer: rawSopMetadata.reviewer || '',
    effectiveDate: rawSopMetadata.effectiveDate || rawSopMetadata.effective_date || '',
    reviewDate: rawSopMetadata.reviewDate || rawSopMetadata.review_date || '',
    riskLevel: rawSopMetadata.riskLevel || rawSopMetadata.risk_level || '',
    regulatoryReferences:
      rawSopMetadata.regulatoryReferences
      || rawSopMetadata.regulatory_references
      || [],
  }

  return {
    ...rawMetadata,
    sopMetadata: normalizedSopMetadata,
    sopStatus: rawMetadata.sopStatus || cv.external_status || cv.status || sop.status || '',
  }
}

// ── Main Page ──────────────────────────────────────────────────────────────

export default function SOPsPage() {
  const fileInputRef = React.useRef(null)
  const [importing, setImporting] = useState(false)
  const [importNotice, setImportNotice] = useState(null)

  // ── Document tab system ──────────────────────────────────────────────────
  const [tabs, setTabs] = useState([
    { id: 'sops-list', label: 'SOPs', type: 'list', closeable: false },
  ])
  const [activeTabId, setActiveTabId] = useState('sops-list')

  useEffect(() => {
    const openedTabs = tabs.map((tab) => ({
      id: tab.id,
      label: tab.label,
      type: tab.type,
      docId: tab.docId || null,
    }))
    const activeTab = tabs.find((tab) => tab.id === activeTabId) || null
    try {
      localStorage.setItem(
        KL_WORKSPACE_CONTEXT_KEY,
        JSON.stringify({
          updated_at: new Date().toISOString(),
          active_tab_id: activeTabId,
          active_tab_label: activeTab?.label || '',
          opened_tabs: openedTabs,
        }),
      )
    } catch {
      // ignore storage failures
    }
  }, [tabs, activeTabId])

  const openNewSOPTab = useCallback(() => {
    const tabId = 'editor-new'
    setTabs(prev => {
      if (prev.find(t => t.id === tabId)) return prev
      return [...prev, { id: tabId, label: 'Neue SOP', type: 'editor', docId: null, closeable: true }]
    })
    setActiveTabId(tabId)
  }, [])

  const openSOPEditorTab = useCallback((sopId, sopCode, initialMetadataJson = null, initialStatus = '', initialDocTitle = '') => {
    const tabId = `editor-${sopId}`
    const openRequestKey = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
    console.debug('[SOP Open] openSOPEditorTab payload', {
      tabId,
      sopId,
      sopCode,
      initialStatus,
      initialDocTitle,
      hasInitialMetadataJson: Boolean(initialMetadataJson),
      metadataKeys: initialMetadataJson && typeof initialMetadataJson === 'object' ? Object.keys(initialMetadataJson) : [],
      sopMetadataKeys:
        initialMetadataJson?.sopMetadata && typeof initialMetadataJson.sopMetadata === 'object'
          ? Object.keys(initialMetadataJson.sopMetadata)
          : [],
    })
    setTabs(prev => {
      const existing = prev.find(t => t.id === tabId)
      if (existing) {
        return prev.map((tab) => (
          tab.id === tabId
            ? {
                ...tab,
                label: sopCode || tab.label,
                initialMetadataJson: initialMetadataJson || tab.initialMetadataJson || null,
                initialStatus: initialStatus || tab.initialStatus || '',
                initialDocTitle: initialDocTitle || tab.initialDocTitle || '',
                openRequestKey,
              }
            : tab
        ))
      }
      return [...prev, {
        id: tabId,
        label: sopCode || `SOP-${sopId}`,
        type: 'editor',
        docId: String(sopId),
        initialMetadataJson,
        initialStatus,
        initialDocTitle,
        openRequestKey,
        closeable: true,
      }]
    })
    setActiveTabId(tabId)
  }, [])

  const closeTab = useCallback((tabId, e) => {
    e.stopPropagation()
    setTabs(prev => prev.filter(t => t.id !== tabId))
    setActiveTabId(prev => (prev === tabId ? 'sops-list' : prev))
  }, [])

  // ── Data ─────────────────────────────────────────────────────────────────
  const [viewMode, setViewMode] = useState('knowledge')
  const [searchQuery, setSearchQuery] = useState('')
  const [kiSummaryOpen, setKiSummaryOpen] = useState(true)
  const [searchTerm, setSearchTerm] = useState('')
  const [activeFilterTab, setActiveFilterTab] = useState('Alle')
  const [sortOrder, setSortOrder] = useState('asc') // 'asc' | 'recent' | 'oldest'
  const [isKIAnalyzing, setIsKIAnalyzing] = useState(false)
  const [kiError, setKIError] = useState('')
  const [kiSummaryText, setKISummaryText] = useState('')
  const [kiSources, setKISources] = useState([])
  const KNOWLEDGE_SESSION_LS = 'cybrain_knowledge_chat_session_id'
  const [knowledgeSessionId, setKnowledgeSessionId] = useState(() =>
    typeof window !== 'undefined' ? localStorage.getItem(KNOWLEDGE_SESSION_LS) : null,
  )
  const [knowledgeHistoryRows, setKnowledgeHistoryRows] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [sops, setSops] = useState([])
  const [sopToDelete, setSopToDelete] = useState(null)
  const [isDeletingSOP, setIsDeletingSOP] = useState(false)

  const loadSOPs = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const raw = await getSOPs()
      // Raw data — no hardcoded sort; user-controlled dropdown owns ordering
      const mapped = Array.isArray(raw) ? raw.map(mapSOP) : []
      setSops(mapped)
    } catch (err) {
      console.error('Failed to load SOPs:', err)
      setError('SOPs konnten nicht geladen werden.')
      setSops([])
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    loadSOPs()
  }, [loadSOPs])

  useEffect(() => {
    if (viewMode !== 'knowledge') {
      setKnowledgeHistoryRows([])
      return
    }
    const sid = knowledgeSessionId || localStorage.getItem(KNOWLEDGE_SESSION_LS)
    if (!sid) {
      setKnowledgeHistoryRows([])
      return
    }
    let cancelled = false
    getChatSessionMessages(sid)
      .then((rows) => {
        if (cancelled || !Array.isArray(rows)) return
        setKnowledgeHistoryRows(
          rows.map((m) => ({
            id: m.id,
            role: m.role,
            content: String(m.content || '').slice(0, 4000),
          })),
        )
      })
      .catch((err) => {
        console.error('[chat-history-load] knowledge session', err)
        if (!cancelled) setKnowledgeHistoryRows([])
      })
    return () => {
      cancelled = true
    }
  }, [viewMode, knowledgeSessionId])

  useEffect(() => {
    const onRefreshRequest = (event) => {
      const sopId = event?.detail?.sop_id ? String(event.detail.sop_id) : null
      console.info('[sops-refresh-request]', event?.detail || {})
      loadSOPs()
      if (event?.detail?.reason === 'delete' && sopId) {
        setTabs((prev) => prev.filter((tab) => String(tab.docId || '') !== sopId))
        setActiveTabId((prevActiveId) => (
          String(prevActiveId).includes(sopId) ? 'sops-list' : prevActiveId
        ))
      }
    }
    window.addEventListener('sops-refresh-request', onRefreshRequest)
    return () => window.removeEventListener('sops-refresh-request', onRefreshRequest)
  }, [loadSOPs])

  // ── Filtering ─────────────────────────────────────────────────────────────
  const STATUS_MAP = {
    'Freigegeben': ['effective', 'released'],
    'In Prüfung': ['under_review', 'in_review'],
    'Entwurf': ['draft'],
  }

  const filteredSops = useMemo(() => {
    const normalizedSearch = searchTerm.toLowerCase()
    return sops.filter(sop => {
      const matchesSearch =
        sop.title.toLowerCase().includes(normalizedSearch) ||
        sop.code.toLowerCase().includes(normalizedSearch) ||
        (sop.department || '').toLowerCase().includes(normalizedSearch)
      if (activeFilterTab === 'Alle') return matchesSearch
      const allowed = STATUS_MAP[activeFilterTab] || []
      return matchesSearch && allowed.includes((sop.status || '').toLowerCase())
    })
  }, [sops, searchTerm, activeFilterTab])

  // ── Client-side sorting (works on real backend data) ─────────────────────
  const sortedSops = useMemo(() => {
    return [...filteredSops].sort((a, b) => {
      if (sortOrder === 'recent') {
        return new Date(b.updated_at_raw || 0) - new Date(a.updated_at_raw || 0)
      }
      if (sortOrder === 'oldest') {
        return new Date(a.updated_at_raw || 0) - new Date(b.updated_at_raw || 0)
      }
      // Default: ascending A→Z by SOP number
      return (a.sop_number || a.code || '').localeCompare(b.sop_number || b.code || '', 'de')
    })
  }, [filteredSops, sortOrder])

  // ── Actions ──────────────────────────────────────────────────────────────
  const handleOpen = useCallback(async (sopOrId) => {
    const sop = typeof sopOrId === 'object' ? sopOrId : sops.find(s => s.id === String(sopOrId))
    const id = typeof sopOrId === 'object' ? sopOrId.id : String(sopOrId)
    const code = sop?.sop_number || sop?.code || `SOP-${id}`
    try {
      const doc = await getDocument(id)
      console.debug('[SOP Open] getDocument response', {
        id: doc?.id,
        title: doc?.title,
        status: doc?.status,
        sop_number: doc?.sop_number,
        hasMetadataJson: Boolean(doc?.metadata_json && typeof doc.metadata_json === 'object'),
        metadataKeys: doc?.metadata_json && typeof doc.metadata_json === 'object' ? Object.keys(doc.metadata_json) : [],
        sopMetadataKeys:
          doc?.metadata_json?.sopMetadata && typeof doc.metadata_json.sopMetadata === 'object'
            ? Object.keys(doc.metadata_json.sopMetadata)
            : [],
      })
      const metadataJson = doc?.metadata_json && typeof doc.metadata_json === 'object'
        ? doc.metadata_json
        : buildInitialSOPMetadata(sop)
      const status = doc?.status || sop?.status || ''
      const title = doc?.title || sop?.title || ''
      openSOPEditorTab(id, code, metadataJson, status, title)
    } catch (err) {
      console.error('Failed to preload SOP document metadata, falling back to list data:', err)
      const metadataJson = buildInitialSOPMetadata(sop)
      const status = sop?.status || ''
      openSOPEditorTab(id, code, metadataJson, status, sop?.title || '')
    }
  }, [sops, openSOPEditorTab])

  const handleOpenNewTab = useCallback((sopOrId) => {
    const id = typeof sopOrId === 'object' ? sopOrId.id : String(sopOrId)
    window.open(`/editor/${id}`, '_blank')
  }, [])

  const handleCreate = useCallback(() => {
    openNewSOPTab()
  }, [openNewSOPTab])

  const handleAnalyze = () => {
    const text = searchQuery.trim()
    if (!text) return
    setIsKIAnalyzing(true)
    setKIError('')
    const sid =
      knowledgeSessionId ||
      (typeof window !== 'undefined' ? localStorage.getItem(KNOWLEDGE_SESSION_LS) : null)
    const chatHistoryPayload = knowledgeHistoryRows.map((r) => ({
      role: r.role,
      content: r.content,
    }))
    queryAI(text, {
      category: 'sop',
      surface: 'knowledge_search',
      route: '/knowledge',
      session_id: sid || undefined,
      assistant_context: getKLAssistantContext('/knowledge'),
      chat_history: chatHistoryPayload.length ? chatHistoryPayload : undefined,
    })
      .then(async (res) => {
        setKISummaryText(res?.answer || 'Keine Antwort vom Backend erhalten.')
        setKISources(Array.isArray(res?.sources) ? res.sources : [])
        const nextSid = res?.session_id && String(res.session_id).trim()
        if (nextSid) {
          try {
            localStorage.setItem(KNOWLEDGE_SESSION_LS, nextSid)
          } catch {
            // ignore
          }
          setKnowledgeSessionId(nextSid)
          try {
            const rows = await getChatSessionMessages(nextSid)
            if (Array.isArray(rows)) {
              setKnowledgeHistoryRows(
                rows.map((m) => ({
                  id: m.id,
                  role: m.role,
                  content: String(m.content || '').slice(0, 4000),
                })),
              )
            }
          } catch (e) {
            console.error('[chat-history-load] knowledge after query', e)
          }
        }
      })
      .catch((err) => {
        setKIError(err?.message || 'KI-Analyse fehlgeschlagen.')
        setKISummaryText('')
        setKISources([])
      })
      .finally(() => setIsKIAnalyzing(false))
  }

  const handleQuickFilter = (query) => setSearchQuery(query)
  
  const handleImportClick = () => {
    fileInputRef.current?.click()
  }

  const handleDeleteClick = useCallback((sop) => {
    setSopToDelete(sop)
  }, [])

  const handleDeleteCancel = useCallback(() => {
    if (isDeletingSOP) return
    setSopToDelete(null)
  }, [isDeletingSOP])

  const handleDeleteConfirm = useCallback(async () => {
    if (!sopToDelete) return
    setIsDeletingSOP(true)
    try {
      await deleteDocument(sopToDelete.id)
      // Success: refresh the list
      await loadSOPs()
      // Also close any open tabs for this doc
      setTabs(prev => prev.filter(t => t.docId !== String(sopToDelete.id)))
      setSopToDelete(null)
    } catch (err) {
      console.error('Failed to delete SOP:', err)
      alert(`Fehler beim Löschen: ${err.message}`)
    } finally {
      setIsDeletingSOP(false)
    }
  }, [loadSOPs, sopToDelete])

  const handleFileChange = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return

    setImporting(true)
    setImportNotice(null)
    try {
      const imported = await prepareNewSOPImport(file)
      const createPayload = {
        title: imported.resolvedTitle,
        doc_json: imported.docJson,
        metadata_json: imported.metadataJson,
      }

      if (import.meta?.env?.DEV) {
        console.debug('[SOP Import] createDocument payload', createPayload)
      }

      const newDoc = await createDocument(createPayload)

      const tabLabel = buildSOPDisplayLabel(imported.metadata)
        || newDoc.sop_number
        || imported.resolvedTitle

      openSOPEditorTab(newDoc.id, tabLabel)

      setImportNotice({
        type: 'success',
        message: `Imported ${file.name}${imported.metadata?.documentId ? ` (${imported.metadata.documentId})` : ''}`,
      })
      window.setTimeout(() => setImportNotice(null), 4000)

      // Reset file input
      if (fileInputRef.current) fileInputRef.current.value = ''

      // Optional: reload list
      loadSOPs()
    } catch (err) {
      console.error('SOP import failed:', err, {
        status: err?.status,
        responseBody: err?.responseBody || null,
      })
      const message =
        err?.status === 409
          ? (err.message ||
              'This SOP ID already exists. Please create a new version or choose another SOP ID.')
          : (err?.message || 'Import failed')
      setImportNotice({ type: 'error', message })
    } finally {
      setImporting(false)
    }
  }

  return (
    <div className="sops-tabbed-page">

      {/* ── Document tab bar ──────────────────────────────────────────── */}
      <div className="doc-tab-bar" role="tablist" aria-label="SOP Tabs">
        {tabs.map(tab => (
          <button
            key={tab.id}
            role="tab"
            aria-selected={tab.id === activeTabId}
            className={`doc-tab${tab.id === activeTabId ? ' doc-tab-active' : ''}`}
            onClick={() => setActiveTabId(tab.id)}
            title={tab.label}
          >
            {tab.type === 'editor'
              ? <FileEdit size={13} className="doc-tab-icon" />
              : <List size={13} className="doc-tab-icon" />
            }
            <span className="doc-tab-label" title={tab.label}>{tab.label}</span>
            {tab.closeable && (
              <span
                className="doc-tab-close"
                role="button"
                aria-label={`${tab.label} schließen`}
                onClick={(e) => closeTab(tab.id, e)}
              >
                <X size={11} />
              </span>
            )}
          </button>
        ))}
      </div>

      {/* ── LIST TAB CONTENT ──────────────────────────────────────────── */}
      <div
        role="tabpanel"
        style={{ display: activeTabId === 'sops-list' ? undefined : 'none' }}
      >
        {/* TABLE VIEW */}
        {viewMode === 'table' && (
          <div className="sops-page">
            <header className="sops-header">
              <div className="header-titles">
                <button
                  className="sops-back-btn"
                  onClick={() => setViewMode('knowledge')}
                >
                  <ArrowLeft size={15} /> SOPs
                </button>
                <h2 className="page-title">Standard Operating Procedures</h2>
              </div>
              <div className="header-actions">
                <button className="export-btn">
                  <Download size={18} /> <span>Exportieren</span>
                </button>
                <button className="new-sop-btn" onClick={handleCreate}>
                  <Plus size={18} /> <span>Neue SOP erstellen</span>
                </button>
              </div>
            </header>

            <section className="sops-controls-card">
              <div className="search-box">
                <Search size={18} className="search-icon" />
                <input
                  type="text"
                  placeholder="Suchen nach Titel, Code oder Verantwortlichen..."
                  value={searchTerm}
                  onChange={e => setSearchTerm(e.target.value)}
                />
              </div>
              <div className="filter-group">
                <button className="filter-btn-outline"><Filter size={16} /> <span>Filter</span></button>
                <div className="view-selector">
                  {['Alle', 'Freigegeben', 'In Prüfung', 'Entwurf'].map(tab => (
                    <button
                      key={tab}
                      className={`view-tab ${activeFilterTab === tab ? 'active' : ''}`}
                      onClick={() => setActiveFilterTab(tab)}
                    >
                      {tab}
                    </button>
                  ))}
                </div>
                <select
                  className="sop-sort-select"
                  value={sortOrder}
                  onChange={e => setSortOrder(e.target.value)}
                  aria-label="Sortierung"
                >
                  <option value="asc">A → Z</option>
                  <option value="recent">Neueste zuerst</option>
                  <option value="oldest">Älteste zuerst</option>
                </select>
              </div>
            </section>
            <div className="table-container-card">
              {loading ? (
                <div className="table-loading">
                  <Loader size={20} className="spin" /> Lade SOPs...
                </div>
              ) : error ? (
                <div className="table-loading" style={{ color: 'var(--error)' }}>{error}</div>
              ) : sortedSops.length === 0 ? (
                <div className="table-loading">Keine SOPs gefunden.</div>
              ) : (
                <SOPTable 
                  data={sortedSops} 
                  onRowClick={(sop) => handleOpen(sop)} 
                  onOpenNewTab={handleOpenNewTab} 
                  onDelete={handleDeleteClick}
                />
              )}
            </div>
          </div>
        )}

        {/* KNOWLEDGE VIEW */}
        {viewMode === 'knowledge' && (
          <div className="sops-kb-page">
            {/* Breadcrumb tag */}
            <div className="sops-kb-breadcrumb">
              <span className="sops-bc-tag">
                <Plus size={12} /> SOPs
              </span>
            </div>

            {/* Hero card */}
            <div className="sops-hero-card">
              <h1 className="sops-hero-title">Was möchten Sie über Ihre SOPs wissen?</h1>
              <p className="sops-hero-desc">
                Stellen Sie eine Frage im natürlichen Sprachstil. Die KI analysiert Ihre SOPs, verknüpften
                Abweichungen, CAPAs und Audits und liefert eine strukturierte, handlungsbasierte Antwort.
              </p>

              {/* Query input */}
              <div className="sops-query-wrap">
                <textarea
                  className="sops-query-input"
                  placeholder="z.B. Zeige SOPs mit erhöhten Produktionsrisiken? Welche SOP ist am häufigsten angepasst worden?"
                  value={searchQuery}
                  onChange={e => setSearchQuery(e.target.value)}
                  rows={3}
                  onKeyDown={e => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault()
                      handleAnalyze()
                    }
                  }}
                />
                <button
                  className="sops-analyze-btn"
                  onClick={handleAnalyze}
                  disabled={!searchQuery.trim()}
                >
                  <Sparkles size={15} />
                  KI analysieren
                </button>
              </div>

              {/* Action buttons */}
              <div className="sops-action-row">
                <button
                  className="sops-action-btn"
                  onClick={() => sops.length > 0 && handleOpen(sops[0])}
                  disabled={sops.length === 0}
                >
                  <Edit3 size={14} /> SOP bearbeiten
                </button>
                <button
                  className="sops-action-btn"
                  onClick={() => sops.length > 0 && handleOpenNewTab(sops[0])}
                  disabled={sops.length === 0}
                  title="In neuem Tab öffnen"
                >
                  <ExternalLink size={14} /> In neuem Tab
                </button>
                <button className="sops-action-btn" onClick={() => setViewMode('table')}>
                  <FileText size={14} /> Alle SOPs anzeigen
                </button>
                <button className="sops-action-btn sops-action-primary" onClick={handleCreate}>
                  <Plus size={14} /> Neue SOP
                </button>
                <button 
                  className={`sops-action-btn sops-action-primary ${importing ? 'loading' : ''}`} 
                  onClick={handleImportClick}
                  disabled={importing}
                  aria-busy={importing}
                >
                  {importing ? <Loader size={14} className="spin" /> : <Upload size={14} />}
                  <span>{importing ? 'Importiere...' : 'Import SOP'}</span>
                </button>
                {importNotice ? (
                  <div
                    role="status"
                    aria-live="polite"
                    className={`sops-import-notice sops-import-notice-${importNotice.type}`}
                  >
                    {importNotice.type === 'error' ? <AlertCircle size={14} /> : null}
                    <span>{importNotice.message}</span>
                    <button
                      type="button"
                      className="sops-import-notice-dismiss"
                      onClick={() => setImportNotice(null)}
                      aria-label="Dismiss"
                    >
                      <X size={12} />
                    </button>
                  </div>
                ) : null}
                <input
                  type="file"
                  ref={fileInputRef}
                  style={{ display: 'none' }}
                  accept={SOP_IMPORT_ACCEPT}
                  onChange={handleFileChange}
                />
              </div>

              {/* Quick filter chips */}
              <div className="sops-quick-chips">
                {quickFilters.map(f => (
                  <button key={f} className="sops-quick-chip" onClick={() => handleQuickFilter(f)}>
                    {f}
                  </button>
                ))}
              </div>
            </div>

            {/* KI Summary */}
            <KISummary
              open={kiSummaryOpen}
              onToggle={() => setKiSummaryOpen(v => !v)}
              query={searchQuery}
              summaryText={kiSummaryText}
              sources={kiSources}
              loading={isKIAnalyzing}
              error={kiError}
              historyRows={knowledgeHistoryRows}
            />
            {/* Relevant SOPs from backend */}
            <div className="sops-section-title-row">
              <h2 className="sops-section-title">Relevante SOPs im aktuellen Kontext</h2>
              <select
                className="sop-sort-select"
                value={sortOrder}
                onChange={e => setSortOrder(e.target.value)}
                aria-label="Sortierung"
              >
                <option value="asc">A → Z</option>
                <option value="recent">Neueste zuerst</option>
                <option value="oldest">Älteste zuerst</option>
              </select>
            </div>

            <div className="sops-cards-grid">
              {loading && (
                <div style={{ padding: '24px', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: 8 }}>
                  <Loader size={16} /> Lade SOPs...
                </div>
              )}
              {!loading && error && (
                <div style={{ padding: '24px', color: 'var(--error)' }}>{error}</div>
              )}
              {!loading && !error && sops.length === 0 && (
                <div style={{ padding: '24px', color: 'var(--text-muted)' }}>
                  Keine SOPs in der Datenbank gefunden.
                </div>
              )}
              {!loading && sortedSops.map(sop => (
                <SOPCard
                  key={sop.id}
                  sop={sop}
                  onOpen={handleOpen}
                  onOpenNewTab={handleOpenNewTab}
                  onEdit={handleOpen}
                  onDelete={handleDeleteClick}
                />
              ))}
            </div>
          </div>
        )}
      </div>

      {/* ── EDITOR TABS CONTENT ──────────────────────────────────────────── */}
      {tabs.filter(t => t.type === 'editor' && t.id === activeTabId).map(tab => (
        <div
          key={tab.id}
          role="tabpanel"
          className="editor-tab-wrapper"
        >
          <Suspense
            fallback={
              <div style={{ padding: '24px', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: 8 }}>
                <Loader size={16} className="spin" /> Lade Editor...
              </div>
            }
          >
            <EditorPage
              isEmbedded
              initialDocId={tab.docId !== undefined ? tab.docId : null}
              initialMetadataJson={tab.initialMetadataJson || null}
              initialStatus={tab.initialStatus || ''}
              initialDocTitle={tab.initialDocTitle || ''}
              openRequestKey={tab.openRequestKey || ''}
              embedTabId={tab.id}
              onImportMetadataApplied={({ tabId, tabLabel }) => {
                if (!tabId || !tabLabel) return
                setTabs((prev) =>
                  prev.map((row) => (row.id === tabId ? { ...row, label: tabLabel } : row)),
                )
              }}
            />
          </Suspense>
        </div>
      ))}

      {sopToDelete && (
        <div className="sop-delete-modal-overlay" role="presentation">
          <div className="sop-delete-modal" role="dialog" aria-modal="true" aria-labelledby="sop-delete-title">
            <h3 id="sop-delete-title" className="sop-delete-title">
              SOP wirklich löschen?
            </h3>
            <p className="sop-delete-message">
              Sind Sie sicher, dass Sie "{sopToDelete.title}" ({sopToDelete.sop_number}) löschen möchten?
              Alle Versionen werden dauerhaft entfernt.
            </p>
            <div className="sop-delete-actions">
              <button
                type="button"
                className="sop-delete-btn sop-delete-btn-cancel"
                onClick={handleDeleteCancel}
                disabled={isDeletingSOP}
              >
                Cancel
              </button>
              <button
                type="button"
                className="sop-delete-btn sop-delete-btn-confirm"
                onClick={handleDeleteConfirm}
                disabled={isDeletingSOP}
              >
                {isDeletingSOP ? 'Deleting...' : 'OK'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
