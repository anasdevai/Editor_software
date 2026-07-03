import React, { Suspense, lazy, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useParams } from 'react-router-dom'
import { debounce } from 'lodash'
import { useEditor } from '@tiptap/react'
import { Extension } from '@tiptap/core'
import StarterKit from '@tiptap/starter-kit'
import HardBreak from '@tiptap/extension-hard-break'
import Link from '@tiptap/extension-link'
import Underline from '@tiptap/extension-underline'
import { Table } from '@tiptap/extension-table'
import { TableRow } from '@tiptap/extension-table-row'
import { TableCell } from '@tiptap/extension-table-cell'
import { TableHeader } from '@tiptap/extension-table-header'
import UniqueID from '@tiptap/extension-unique-id'
import {
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Eye,
  Maximize2,
} from 'lucide-react'

import RelatedContextSidebar from '../components/Editor/SOP/RelatedContextSidebar'
import SOPMetadataPanel from '../components/Editor/SOP/SOPMetadataPanel'
import LinkModal from '../components/Common/LinkModal'
import LinkingModal from '../components/Common/LinkingModal'
import EditorToolbarSection from '../components/Editor/EditorToolbarSection'
import EditorDocStatsBar from '../components/Editor/EditorDocStatsBar'
import EditorTypingSurface from '../components/Editor/EditorTypingSurface'
import AIWidget from '../components/Dashboard/AIWidget'
import FloatingAskAIButton from '../components/Common/FloatingAskAIButton'
import { useLanguage } from '../context/LanguageContext'
import {
  createDocument,
  createVersion,
  getDocument,
  getRelatedContext,
  getVersion,
  getVersions,
  updateDocument,
  updateVersionStatus,
} from '../api/editorApi'
import { DEFAULT_SOP_VERSION_METADATA, SOP_ORDER, SOP_STATES } from '../utils/sopConstants'
import {
  applySOPImportMetadata,
  buildSOPDisplayLabel,
  normalizeSOPTitleForDisplay,
  prepareEditorSOPImport,
} from '../utils/sopImportService'
import { InlineAiSuggestion } from '../extensions/InlineAiSuggestion'
import {
  KL_ASSISTANT_CONTEXT_REFRESH_DONE,
  KL_ASSISTANT_CONTEXT_REFRESH_REQUEST,
  notifySopEditorContextChanged,
} from '../utils/editorAiBridge'
import { parseSopDocument } from '../utils/targeting/sopParser'
import '../assets/styles/global.css'

const PreviewModal = lazy(() => import('../components/Common/PreviewModal'))
const SideBySideViewer = lazy(() => import('../components/Editor/Diff/SideBySideViewer'))

class EditorSurfaceErrorBoundary extends React.Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false }
  }

  static getDerivedStateFromError() {
    return { hasError: true }
  }

  componentDidCatch(error) {
    console.error('Editor surface crashed:', error)
  }

  render() {
    if (this.state.hasError) {
      return <div className="editor-loading">Editor failed to load. Please refresh.</div>
    }
    return this.props.children
  }
}

const EMPTY_DOC = {
  type: 'doc',
  content: [],
}

const STORAGE_KEY = 'current_document_id'
const KL_EDITOR_CONTEXT_KEY = 'kl_assistant_editor_state_v2'

const isLikelySectionHeading = (line) => {
  const text = String(line || '').trim()
  if (!text) return false
  if (/^\d+(?:\.\d+)*[)\].:-]?\s+\S/u.test(text) && text.length < 180) return true
  if (/^(purpose|scope|procedure|responsibilities|definitions|approval|revision history|zweck|geltungsbereich|verfahren|verantwortlichkeiten)\b/i.test(text)) return true
  if (/^[A-ZÄÖÜ][A-ZÄÖÜ0-9\s/&()-]{4,}$/u.test(text) && text.length < 120) return true
  return false
}

const deriveEditorSelectionSnapshot = (editorInstance) => {
  if (!editorInstance || editorInstance.isDestroyed) {
    return {
      selectedText: '',
      selectedRange: { from: 0, to: 0, empty: true },
      selectedSection: { name: '', type: '', scope: 'cursor_context', text_excerpt: '' },
    }
  }

  const { state } = editorInstance
  const { selection } = state
  const docSize = state.doc.content.size
  const from = Number(selection?.from) || 0
  const to = Number(selection?.to) || from
  const empty = !selection || selection.empty
  const selectedText = empty
    ? ''
    : (state.doc.textBetween(from, to, '\n') || '').trim()
  const prefixText = state.doc.textBetween(0, Math.max(to, 0), '\n') || ''
  const prefixLines = prefixText
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
  let sectionName = ''
  for (let i = prefixLines.length - 1; i >= 0; i -= 1) {
    if (isLikelySectionHeading(prefixLines[i])) {
      sectionName = prefixLines[i]
      break
    }
  }
  if (!sectionName && !empty) {
    sectionName = selectedText.split(/\r?\n/).find((line) => String(line || '').trim()) || ''
  }
  if (!sectionName) {
    try {
      sectionName = selection?.$from?.parent?.textContent || ''
    } catch {
      sectionName = ''
    }
  }

  const blockText = (() => {
    try {
      return (selection?.$from?.parent?.textContent || '').trim()
    } catch {
      return ''
    }
  })()

  const scope = empty ? 'cursor_context' : 'selection'
  const excerpt = selectedText || blockText || sectionName
  const sectionType = /table/i.test(selection?.$from?.parent?.type?.name || '')
    ? 'Table'
    : /heading/i.test(selection?.$from?.parent?.type?.name || '')
      ? 'Heading'
      : 'Paragraph'

  return {
    selectedText,
    selectedRange: {
      from,
      to: Math.min(to, docSize),
      empty,
    },
    selectedSection: {
      name: String(sectionName || '').trim().slice(0, 180),
      type: sectionType,
      scope,
      text_excerpt: String(excerpt || '').trim().slice(0, 1600),
    },
  }
}

const EditorShortcuts = Extension.create({
  name: 'editorShortcuts',
  addKeyboardShortcuts() {
    const handleEmptyBlockBackspace = () => {
      const { state } = this.editor
      const { selection } = state
      if (!selection.empty) return false
      const { $from } = selection
      const parent = $from.parent
      if (!parent?.isTextblock || parent.content.size > 0) return false
      return this.editor.commands.joinBackward() || this.editor.commands.liftEmptyBlock()
    }

    const handleEmptyBlockDelete = () => {
      const { state } = this.editor
      const { selection } = state
      if (!selection.empty) return false
      const { $from } = selection
      const parent = $from.parent
      if (!parent?.isTextblock || parent.content.size > 0) return false
      return this.editor.commands.joinForward() || this.editor.commands.liftEmptyBlock()
    }

    return {
      'Mod-b': () => this.editor.chain().focus().toggleBold().run(),
      'Mod-i': () => this.editor.chain().focus().toggleItalic().run(),
      'Mod-u': () => this.editor.chain().focus().toggleUnderline().run(),
      Enter: () => this.editor.chain().focus().splitBlock().run(),
      'Shift-Enter': () => this.editor.chain().focus().setHardBreak().run(),
      'Mod-Shift-1': () => this.editor.chain().focus().toggleHeading({ level: 1 }).run(),
      'Mod-Shift-2': () => this.editor.chain().focus().toggleHeading({ level: 2 }).run(),
      'Mod-Shift-3': () => this.editor.chain().focus().toggleHeading({ level: 3 }).run(),
      'Mod-Alt-1': () => this.editor.chain().focus().toggleHeading({ level: 1 }).run(),
      'Mod-Alt-2': () => this.editor.chain().focus().toggleHeading({ level: 2 }).run(),
      'Mod-Alt-3': () => this.editor.chain().focus().toggleHeading({ level: 3 }).run(),
      'Mod-Alt-t': () => this.editor.chain().focus().insertTable({ rows: 3, cols: 3, withHeaderRow: true }).run(),
      Backspace: handleEmptyBlockBackspace,
      Delete: handleEmptyBlockDelete,
    }
  },
})

const normalizeMeta = (rawMeta) => {
  if (!rawMeta || typeof rawMeta !== 'object') {
    return { ...DEFAULT_SOP_VERSION_METADATA }
  }

  if (rawMeta.sopMetadata !== undefined) {
    return {
      ...DEFAULT_SOP_VERSION_METADATA,
      ...rawMeta,
      sopStatus: normalizeWorkflowStatus(rawMeta.sopStatus || rawMeta.status) || DEFAULT_SOP_VERSION_METADATA.sopStatus,
      sopMetadata: {
        ...DEFAULT_SOP_VERSION_METADATA.sopMetadata,
        ...(rawMeta.sopMetadata || {}),
      },
      auditTrail: Array.isArray(rawMeta.auditTrail) ? rawMeta.auditTrail : [],
    }
  }

  return {
    ...DEFAULT_SOP_VERSION_METADATA,
    sopStatus: normalizeWorkflowStatus(rawMeta.sopStatus || rawMeta.status) || DEFAULT_SOP_VERSION_METADATA.sopStatus,
    sopMetadata: {
      ...DEFAULT_SOP_VERSION_METADATA.sopMetadata,
      ...rawMeta,
    },
    auditTrail: Array.isArray(rawMeta.auditTrail) ? rawMeta.auditTrail : [],
  }
}

const formatTimestamp = (value) => {
  if (!value) return ''
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return ''
  return date.toLocaleString('en-GB', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  })
}

const buildVersionLabel = (version) => {
  if (!version) return 'SOP-NEW'
  const number = version.versionNumber || 1
  const stamp = version.timestamp || ''
  return stamp ? `v${number} (${stamp})` : `v${number}`
}

const normalizeWorkflowStatus = (value) => {
  const raw = String(value || '').trim()
  if (!raw) return ''
  const compact = raw.toLowerCase().replace(/[\s-]+/g, '_')
  const aliases = {
    in_review: 'under_review',
    underreview: 'under_review',
    changesrequested: 'changes_requested',
  }
  return aliases[compact] || compact
}

const createAuditEntry = (action, fromStatus, toStatus, note, version) => ({
  id: `audit_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
  action,
  fromStatus,
  toStatus,
  note,
  actor: 'Author',
  version,
  createdAt: new Date().toISOString(),
})

const getLocalizedStatusLabel = (status, t) => {
  const statusMap = {
    draft: t.draft,
    under_review: t.underReview,
    changes_requested: t.changesRequested,
    accepted: t.accepted,
    rejected: t.rejected,
    effective: t.effective,
    obsolete: t.obsolete,
  }
  return statusMap[status] || status
}

const hasMeaningfulDraft = (docJson, sopMetadata = {}) => {
  const text = JSON.stringify(docJson || EMPTY_DOC)
  const hasDocContent = Boolean(text && text !== JSON.stringify(EMPTY_DOC))
  const hasMetadataContent = Object.entries(sopMetadata).some(([key, value]) => {
    if (key === 'title') return value && value !== 'SOP-NEW'
    if (key === 'references') return Array.isArray(value) && value.length > 0
    return Boolean(value)
  })

  return hasDocContent || hasMetadataContent
}

const mapVersion = (version) => ({
  id: version.id,
  versionNumber: version.version_number || 1,
  json: version.doc_json || EMPTY_DOC,
  metadata: normalizeMeta(version.metadata_json),
  status: normalizeWorkflowStatus(version.status) || 'draft',
  timestamp: formatTimestamp(version.created_at),
})

const normalizeEditorMetadataTitle = (inputMetadata) => {
  const metadataObj = inputMetadata && typeof inputMetadata === 'object' ? { ...inputMetadata } : {}
  const documentId = String(metadataObj.documentId || '').trim()
  const title = String(metadataObj.title || '').trim()
  if (!title) return metadataObj
  return {
    ...metadataObj,
    title: normalizeSOPTitleForDisplay(title, documentId),
  }
}

const EditorPage = ({
  isEmbedded = false,
  initialDocId = null,
  initialMetadataJson = null,
  initialStatus = '',
  initialDocTitle = '',
  openRequestKey = '',
  embedTabId = null,
  onImportMetadataApplied = null,
}) => {
  const { language, setLanguage, t } = useLanguage()
  const { id: urlDocId } = useParams()
  const [documentId, setDocumentId] = useState(initialDocId || urlDocId || null)
  const [versions, setVersions] = useState([])
  const [currentVersionId, setCurrentVersionId] = useState(null)
  const [latestVersionId, setLatestVersionId] = useState(null)
  const [metadata, setMetadata] = useState({
    ...DEFAULT_SOP_VERSION_METADATA.sopMetadata,
    title: 'SOP-NEW',
  })
  const [sopStatus, setSopStatus] = useState(DEFAULT_SOP_VERSION_METADATA.sopStatus)
  const [auditTrail, setAuditTrail] = useState([])
  const [versionNote, setVersionNote] = useState('')
  const [isSaving, setIsSaving] = useState(false)
  const [lastSaved, setLastSaved] = useState(null)
  const [saveNotice, setSaveNotice] = useState('')
  const [isLinkModalOpen, setIsLinkModalOpen] = useState(false)
  const [isLinkingModalOpen, setIsLinkingModalOpen] = useState(false)
  const [linkModalInitialUrl, setLinkModalInitialUrl] = useState('')
  const [isPreviewOpen, setIsPreviewOpen] = useState(false)
  const [isImporting, setIsImporting] = useState(false)
  const [importNotice, setImportNotice] = useState('')
  const [isLoadingDocument, setIsLoadingDocument] = useState(false)
  const [isSidebarOpen, setIsSidebarOpen] = useState(false)
  const [aiWidgetOpen, setAiWidgetOpen] = useState(false)
  /** While the bubble AI action is in-flight or the comparison modal is open, skip autosave triggers. */
  const [aiPreviewSessionActive, setAiPreviewSessionActive] = useState(false)
  const [compareBaseVersionId, setCompareBaseVersionId] = useState('')
  const [compareTargetVersionId, setCompareTargetVersionId] = useState('')
  const [diffVersions, setDiffVersions] = useState({ oldVersion: null, newVersion: null })
  const [relatedContextRefreshToken, setRelatedContextRefreshToken] = useState(0)
  const [relatedContextSnapshot, setRelatedContextSnapshot] = useState(null)
  const hydrationRef = useRef(false)
  const saveInFlightRef = useRef(false)
  const [isEditorMounted, setIsEditorMounted] = useState(false)

  useEffect(() => {
    if (!initialMetadataJson || typeof initialMetadataJson !== 'object') return
    const normalizedMeta = normalizeMeta(initialMetadataJson)
    const hydrated = {
      ...DEFAULT_SOP_VERSION_METADATA.sopMetadata,
      ...(normalizedMeta?.sopMetadata || {}),
    }
    if (!hydrated.title && initialDocTitle) {
      hydrated.title = initialDocTitle
    }
    console.debug('[SOP Open] initial metadata hydration', {
      initialDocId,
      embedTabId,
      openRequestKey,
      initialStatus,
      initialDocTitle,
      metadataKeys: Object.keys(initialMetadataJson),
      sopMetadataKeys:
        normalizedMeta?.sopMetadata && typeof normalizedMeta.sopMetadata === 'object'
          ? Object.keys(normalizedMeta.sopMetadata)
          : [],
      hydratedPreview: {
        documentId: hydrated.documentId,
        title: hydrated.title,
        sopVersion: hydrated.sopVersion,
        docType: hydrated.docType,
        category: hydrated.category,
        department: hydrated.department,
      },
    })
    setMetadata(normalizeEditorMetadataTitle(hydrated))
    setSopStatus(
      normalizeWorkflowStatus(initialStatus || normalizedMeta?.sopStatus)
      || DEFAULT_SOP_VERSION_METADATA.sopStatus,
    )
    setAuditTrail(Array.isArray(normalizedMeta?.auditTrail) ? normalizedMeta.auditTrail : [])
    setVersionNote(normalizedMeta?.versionNote || '')
  }, [initialMetadataJson, initialStatus, initialDocId, embedTabId, initialDocTitle, openRequestKey])
  const editorExtensions = useMemo(() => ([
    StarterKit.configure({
      hardBreak: false,
      link: false,
      underline: false,
    }),
    HardBreak,
    Underline,
    Link.configure({
      openOnClick: true,
      autolink: true,
      linkOnPaste: true,
    }),
    Table.configure({ resizable: true }),
    TableRow,
    TableHeader,
    TableCell,
    UniqueID.configure({
      attributeName: 'id',
      types: ['heading', 'paragraph', 'table'],
    }),
    EditorShortcuts,
    InlineAiSuggestion,
  ]), [])

  const editor = useEditor({
    extensions: editorExtensions,
    content: EMPTY_DOC,
    immediatelyRender: false,
    shouldRerenderOnTransaction: false,
    onCreate: () => setIsEditorMounted(true),
    onDestroy: () => setIsEditorMounted(false),
    editorProps: {
      attributes: {
        class: 'tiptap tiptap-sop',
      },
    },
  })

  const handleAiPreviewSessionChange = useCallback((active) => {
    setAiPreviewSessionActive(Boolean(active))
  }, [])

  const currentVersion = versions.find((item) => item.id === currentVersionId) || null
  const isHistoricalView = Boolean(latestVersionId && currentVersionId && latestVersionId !== currentVersionId)
  const currentVersionLabel = buildVersionLabel(currentVersion)
  const cleanMetadataTitle = normalizeSOPTitleForDisplay(
    (metadata?.title || '').trim(),
    (metadata?.documentId || '').trim(),
  )
  const docRevision = (metadata?.sopVersion || '').trim()
  const headerParts = [
    (metadata?.documentId || '').trim(),
    cleanMetadataTitle,
    docRevision ? `v${docRevision.replace(/^v/i, '')}` : '',
  ].filter(Boolean)
  const breadcrumbLabel = headerParts.length ? headerParts.join(' · ') : currentVersionLabel

  const applyVersionState = useCallback((versionRecord, fallbackTitle = '') => {
    if (!editor || !versionRecord || !isEditorMounted || editor.isDestroyed) return

    hydrationRef.current = true
    const normalized = {
      ...DEFAULT_SOP_VERSION_METADATA.sopMetadata,
      ...(versionRecord.metadata?.sopMetadata || {}),
    }

    if (!normalized.title && fallbackTitle) {
      normalized.title = fallbackTitle
    }

    setMetadata(normalizeEditorMetadataTitle(normalized))
    setSopStatus(
      normalizeWorkflowStatus(versionRecord.status || versionRecord.metadata?.sopStatus)
      || DEFAULT_SOP_VERSION_METADATA.sopStatus,
    )
    setAuditTrail(versionRecord.metadata?.auditTrail || [])
    setVersionNote(versionRecord.metadata?.versionNote || '')
    editor.commands.setContent(versionRecord.json || EMPTY_DOC, false)

    window.setTimeout(() => {
      hydrationRef.current = false
    }, 0)
  }, [editor, isEditorMounted])

  const hydrateFromDocument = useCallback(async (docId) => {
    if (!docId || !editor || !isEditorMounted || editor.isDestroyed) return

    setIsLoadingDocument(true)

    try {
      const [doc, dbVersions] = await Promise.all([
        getDocument(docId),
        getVersions(docId),
      ])
      console.debug('[SOP Open] hydrateFromDocument response', {
        requestedDocId: docId,
        returnedDocId: doc?.id,
        title: doc?.title,
        status: doc?.status,
        hasMetadataJson: Boolean(doc?.metadata_json && typeof doc.metadata_json === 'object'),
        metadataKeys: doc?.metadata_json && typeof doc.metadata_json === 'object' ? Object.keys(doc.metadata_json) : [],
        sopMetadataKeys:
          doc?.metadata_json?.sopMetadata && typeof doc.metadata_json.sopMetadata === 'object'
            ? Object.keys(doc.metadata_json.sopMetadata)
            : [],
      })

      const nextVersions = dbVersions.map(mapVersion)
      const currentDocVersion = {
        id: doc.current_version_id,
        version_number: doc.version_number,
        doc_json: doc.doc_json || EMPTY_DOC,
        metadata_json: doc.metadata_json || {},
        status: doc.current_version?.status || doc.status || DEFAULT_SOP_VERSION_METADATA.sopStatus,
        created_at: doc.updated_at || doc.created_at,
      }
      console.debug('[SOP Status Debug] API status sources', {
        docId: doc?.id,
        currentVersionStatus: doc?.current_version?.status || null,
        sopStatus: doc?.status || null,
        metadataJsonStatus: doc?.metadata_json?.status || null,
        metadataSopStatus: doc?.metadata_json?.sopStatus || null,
        chosenStatusBeforeNormalize: currentDocVersion.status,
      })

      const normalizedCurrent = mapVersion(currentDocVersion)
      const mergedVersions = nextVersions.some((item) => item.id === normalizedCurrent.id)
        ? nextVersions.map((item) => (item.id === normalizedCurrent.id ? normalizedCurrent : item))
        : [...nextVersions, normalizedCurrent]

      setVersions(mergedVersions)
      setCurrentVersionId(normalizedCurrent.id)
      setLatestVersionId(normalizedCurrent.id)
      setCompareBaseVersionId(normalizedCurrent.id)
      setCompareTargetVersionId(normalizedCurrent.id)
      applyVersionState(normalizedCurrent, doc.title || '')
      
      // CRITICAL: Always use the UUID from the backend for subsequent API calls (like Related Context)
      if (doc.id) {
        setDocumentId(doc.id)
      }
    } finally {
      setIsLoadingDocument(false)
    }
  }, [editor, applyVersionState, isEditorMounted])

  useEffect(() => {
    console.debug('[SOP Status Debug] final rendered status', {
      documentId,
      sopStatus,
      normalizedSopStatus: normalizeWorkflowStatus(sopStatus),
    })
  }, [documentId, sopStatus])

  useEffect(() => {
    if (!documentId) return
    try {
      localStorage.setItem(STORAGE_KEY, documentId)
      notifySopEditorContextChanged()
    } catch {
      // ignore
    }
  }, [documentId])

  useEffect(() => {
    if (!initialDocId || !openRequestKey || !editor || !isEditorMounted || editor.isDestroyed) return
    // Explicit re-hydration when user clicks Open/Open again for the same tab.
    // This updates metadata for mounted editor tabs without affecting normal typing flow.
    hydrateFromDocument(initialDocId).catch((error) => {
      console.error('Failed to refresh document on open request:', error)
    })
  }, [initialDocId, openRequestKey, editor, isEditorMounted, hydrateFromDocument])

  useEffect(() => {
    if (!editor || !isEditorMounted || editor.isDestroyed) return

    const storedId = localStorage.getItem(STORAGE_KEY)
    const targetId = initialDocId || urlDocId || storedId

    if (!targetId) {
      editor.commands.setContent(EMPTY_DOC, false)
      setDocumentId(null)
      return
    }

    hydrateFromDocument(targetId)
      .then(() => {
        // Hydration sets the correct UUID via setDocumentId(doc.id)
        localStorage.setItem(STORAGE_KEY, targetId)
      })
      .catch((error) => {
        console.error('Failed to load editor document:', error)
      })
  }, [editor, isEditorMounted, initialDocId, urlDocId, hydrateFromDocument])

  useEffect(() => {
    if (!documentId) {
      setRelatedContextSnapshot(null)
      return
    }
    let cancelled = false
    getRelatedContext(documentId)
      .then((ctx) => {
        if (!cancelled) {
          setRelatedContextSnapshot(ctx)
        }
      })
      .catch(() => {
        if (!cancelled) {
          setRelatedContextSnapshot(null)
        }
      })
    return () => {
      cancelled = true
    }
  }, [documentId, relatedContextRefreshToken])

  const persistKlEditorContext = useCallback(() => {
    if (!editor || editor.isDestroyed) return
    const selectionSnapshot = deriveEditorSelectionSnapshot(editor)
    const parsedSop = parseSopDocument(editor)
    const liveTargets = {
      schema_version: 1,
      document_id: String(documentId || ''),
      document_size: parsedSop.docSize || 0,
      captured_at: new Date().toISOString(),
      schema: parsedSop.schema || '',
      sections: (parsedSop.sections || []).map((section, index) => ({
        id: section.id,
        type: 'section',
        label: section.title || `Section ${index + 1}`,
        order: index + 1,
        from: section.position?.from,
        to: section.position?.to,
        content: section.content || '',
      })),
      tables: (parsedSop.tables || []).map((table, index) => ({
        id: table.id,
        type: 'table',
        label: table.caption || table.title || `Table ${index + 1}`,
        order: index + 1,
        parent_id: table.parent_id || null,
        owning_section: table.owningSection || '',
        from: table.position?.from,
        to: table.position?.to,
        content: table.content || '',
      })),
      paragraphs: (parsedSop.paragraphs || []).map((paragraph, index) => ({
        id: paragraph.id,
        type: 'paragraph',
        label: `P${index + 1}${paragraph.parent_title ? ` · ${paragraph.parent_title}` : ''}`,
        order: index + 1,
        parent_id: paragraph.parent_id || null,
        owning_section: paragraph.parent_title || '',
        from: paragraph.position?.from,
        to: paragraph.position?.to,
        content: paragraph.content || '',
      })),
    }
    const linked = {
      deviations: Array.isArray(relatedContextSnapshot?.related_deviations) ? relatedContextSnapshot.related_deviations : [],
      capas: Array.isArray(relatedContextSnapshot?.related_capas) ? relatedContextSnapshot.related_capas : [],
      audits: Array.isArray(relatedContextSnapshot?.related_audit_findings) ? relatedContextSnapshot.related_audit_findings : [],
      decisions: Array.isArray(relatedContextSnapshot?.related_decisions) ? relatedContextSnapshot.related_decisions : [],
      related_sops: Array.isArray(relatedContextSnapshot?.related_sops) ? relatedContextSnapshot.related_sops : [],
    }
    const payload = {
      updated_at: new Date().toISOString(),
      sop: {
        id: documentId || '',
        sop_number: metadata?.documentId || '',
        documentId: metadata?.documentId || '',
        title: metadata?.title || '',
        version: metadata?.sopVersion || '',
        current_version_id: currentVersionId || '',
        status: sopStatus || '',
        references: Array.isArray(metadata?.references) ? metadata.references : [],
        metadata: {
          ...metadata,
        },
        metadata_json: {
          sopStatus,
          sopMetadata: {
            ...metadata,
          },
          auditTrail: Array.isArray(auditTrail) ? auditTrail.slice(-12) : [],
          versionNote: versionNote || '',
        },
      },
      linked,
      editor_text: editor.getText() || '',
      selected_text: selectionSnapshot.selectedText,
      selected_range: selectionSnapshot.selectedRange,
      selected_section_name: selectionSnapshot.selectedSection.name,
      selected_section: selectionSnapshot.selectedSection,
      live_editor_targets: liveTargets,
    }
    try {
      localStorage.setItem(KL_EDITOR_CONTEXT_KEY, JSON.stringify(payload))
      notifySopEditorContextChanged()
    } catch {
      // ignore storage failures
    }
  }, [documentId, metadata, sopStatus, editor, relatedContextSnapshot, currentVersionId, auditTrail, versionNote])

  useEffect(() => {
    persistKlEditorContext()
  }, [documentId, metadata, sopStatus, relatedContextSnapshot, persistKlEditorContext])

  useEffect(() => {
    if (!editor || !isEditorMounted || editor.isDestroyed) return undefined
    const syncContext = debounce(() => {
      persistKlEditorContext()
    }, 1200)
    editor.on('update', syncContext)
    editor.on('selectionUpdate', syncContext)
    return () => {
      editor.off('update', syncContext)
      editor.off('selectionUpdate', syncContext)
      syncContext.cancel()
    }
  }, [editor, isEditorMounted, persistKlEditorContext])

  useEffect(() => {
    if (typeof window === 'undefined') return undefined
    const onRefreshRequest = (event) => {
      const requestId = event?.detail?.requestId
      persistKlEditorContext()
      window.dispatchEvent(
        new CustomEvent(KL_ASSISTANT_CONTEXT_REFRESH_DONE, {
          detail: { requestId, ok: true },
        }),
      )
    }
    window.addEventListener(KL_ASSISTANT_CONTEXT_REFRESH_REQUEST, onRefreshRequest)
    return () => window.removeEventListener(KL_ASSISTANT_CONTEXT_REFRESH_REQUEST, onRefreshRequest)
  }, [persistKlEditorContext])

  const persistDocument = useCallback(async ({ showSavingIndicator = true } = {}) => {
    if (!editor || !isEditorMounted || editor.isDestroyed || isHistoricalView || hydrationRef.current) return
    if (saveInFlightRef.current) return

    const currentJson = editor.getJSON()
    if (!documentId && !hasMeaningfulDraft(currentJson, metadata)) return

    saveInFlightRef.current = true
    if (showSavingIndicator) {
      setIsSaving(true)
    }

    const payload = {
      title: metadata.title || 'SOP-NEW',
      doc_type: 'sop',
      doc_json: currentJson,
      metadata_json: {
        sopStatus,
        sopMetadata: metadata,
        auditTrail,
        versionNote,
      },
    }

    try {
      let response
      let createdNewDocument = false

      if (!documentId) {
        response = await createDocument(payload)
        createdNewDocument = true
        setDocumentId(response.id)
        setCurrentVersionId(response.current_version_id)
        localStorage.setItem(STORAGE_KEY, response.id)
      } else {
        response = await updateDocument(documentId, payload)
      }

      setLastSaved(new Date())

      // Only hydrate after initial create. Re-hydrating on every autosave causes
      // visible editor reflow/flicker and cursor jumps.
      if (createdNewDocument && response?.id) {
        await hydrateFromDocument(response.id)
      }
    } catch (error) {
      console.error('Save failed:', error)
      setSaveNotice(error?.message || 'Save failed. Please try again.')
      window.setTimeout(() => setSaveNotice(''), 2600)
      if (error?.status === 409) {
        window.alert(
          error.message ||
            'This SOP ID already exists. Please create a new version or choose another SOP ID.',
        )
      }
    } finally {
      saveInFlightRef.current = false
      if (showSavingIndicator) {
        setIsSaving(false)
      }
    }
  }, [editor, isEditorMounted, metadata, sopStatus, auditTrail, versionNote, documentId, hydrateFromDocument, isHistoricalView])

  const debouncedSave = useMemo(
    () => debounce(() => {
      persistDocument({ showSavingIndicator: false })
    }, 1600),
    [persistDocument]
  )

  const handleAfterAiBridgeApply = useCallback((detail = {}) => {
    try {
      debouncedSave.flush()
    } catch {
      // ignore
    }
    const suggestionId = String(detail?.suggestion_id || '').trim()
    if (suggestionId && documentId && editor && isEditorMounted && !editor.isDestroyed) {
      createVersion(documentId, {
        doc_json: editor.getJSON(),
        metadata_json: {
          sopStatus: SOP_STATES.DRAFT,
          sopMetadata: metadata,
          auditTrail,
          versionNote,
        },
        status: SOP_STATES.DRAFT,
        suggestion_id: suggestionId,
        change_justification: `Accepted ${detail?.action || 'AI'} suggestion from editor`,
      })
        .then(async (result) => {
          if (result?.id) {
            setCurrentVersionId(result.id)
            setLatestVersionId(result.id)
            await hydrateFromDocument(documentId)
          }
        })
        .catch((err) => {
          console.error('[kl-editor-action] create version after AI apply', err)
          persistDocument({ showSavingIndicator: false }).catch((saveErr) => {
            console.error('[kl-editor-action] persist after AI apply fallback', saveErr)
          })
        })
      return
    }
    persistDocument({ showSavingIndicator: false }).catch((err) => {
      console.error('[kl-editor-action] persist after AI apply', err)
    })
  }, [auditTrail, debouncedSave, documentId, editor, hydrateFromDocument, isEditorMounted, metadata, persistDocument, versionNote])

  useEffect(() => {
    if (!editor || !isEditorMounted || editor.isDestroyed) return

    const handleUpdate = () => {
      if (hydrationRef.current || isHistoricalView || aiPreviewSessionActive) return
      debouncedSave()
    }

    editor.on('update', handleUpdate)

    return () => {
      editor.off('update', handleUpdate)
      debouncedSave.cancel()
    }
  }, [editor, isEditorMounted, debouncedSave, isHistoricalView, aiPreviewSessionActive])

  useEffect(() => {
    if (hydrationRef.current || isHistoricalView || aiPreviewSessionActive) return
    debouncedSave()
  }, [metadata, sopStatus, auditTrail, versionNote, debouncedSave, isHistoricalView, aiPreviewSessionActive])

  useEffect(() => {
    if (!editor || !isEditorMounted || editor.isDestroyed) return
    editor.setEditable(!isHistoricalView)
  }, [editor, isEditorMounted, isHistoricalView])

  useEffect(() => {
    const onKeyDown = (event) => {
      if (!(event.ctrlKey || event.metaKey) || event.key.toLowerCase() !== 's') return
      event.preventDefault()
      if (isHistoricalView) return
      if (!editor || !isEditorMounted || editor.isDestroyed) {
        setSaveNotice('Editor is not ready yet.')
        window.setTimeout(() => setSaveNotice(''), 1800)
        return
      }
      if (!documentId && !metadata?.title && !editor.getText().trim()) {
        setSaveNotice('Nothing to save yet.')
        window.setTimeout(() => setSaveNotice(''), 1800)
        return
      }
      debouncedSave.cancel()
      persistDocument({ showSavingIndicator: true }).catch((error) => {
        console.error('Ctrl+S save failed:', error)
        setSaveNotice(error?.message || 'Save failed.')
        window.setTimeout(() => setSaveNotice(''), 2400)
      })
    }

    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [debouncedSave, persistDocument, isHistoricalView, editor, isEditorMounted, documentId, metadata?.title])

  useEffect(() => {
    document.body.style.overflow = aiWidgetOpen ? 'hidden' : ''
    return () => { document.body.style.overflow = '' }
  }, [aiWidgetOpen])

  const handleMetadataChange = (key, value) => {
    setMetadata((prev) => ({
      ...prev,
      [key]: value,
    }))
  }

  const handleMetadataPanelChange = (nextMetadata) => {
    console.debug('[OCR] metadata passed to SOPMetadataPanel', nextMetadata)
    setMetadata(nextMetadata)
  }

  const addReference = () => {
    const value = window.prompt('Enter SOP reference')
    if (!value?.trim()) return

    setMetadata((prev) => ({
      ...prev,
      references: [...(Array.isArray(prev.references) ? prev.references : []), value.trim()],
    }))
  }

  const removeReference = (index) => {
    setMetadata((prev) => ({
      ...prev,
      references: (Array.isArray(prev.references) ? prev.references : []).filter((_, itemIndex) => itemIndex !== index),
    }))
  }

  const submitForReview = () => {
    const versionNumber = currentVersion?.versionNumber || 1
    const nextAuditTrail = [
      ...auditTrail,
      createAuditEntry('submit_review', sopStatus, SOP_STATES.UNDER_REVIEW, versionNote || 'Submitted for review', versionNumber),
    ]
    setAuditTrail(nextAuditTrail)
    setSopStatus(SOP_STATES.UNDER_REVIEW)

    if (documentId && currentVersionId) {
      updateVersionStatus(documentId, currentVersionId, {
        status: SOP_STATES.UNDER_REVIEW,
        metadata_json: {
          sopStatus: SOP_STATES.UNDER_REVIEW,
          sopMetadata: metadata,
          auditTrail: nextAuditTrail,
          versionNote,
        },
      }).catch((error) => {
        console.error('Failed to update version status:', error)
      })
    }
  }

  const createNewVersionHandler = async () => {
    if (!editor || !isEditorMounted || editor.isDestroyed) return

    if (!documentId) {
      await persistDocument()
    }

    const activeDocumentId = documentId || localStorage.getItem(STORAGE_KEY)
    if (!activeDocumentId) return

    try {
      const result = await createVersion(activeDocumentId, {
        doc_json: editor.getJSON(),
        metadata_json: {
          sopStatus: SOP_STATES.DRAFT,
          sopMetadata: metadata,
          auditTrail: [],
          versionNote,
        },
        status: SOP_STATES.DRAFT,
      })

      await hydrateFromDocument(activeDocumentId)
      if (result?.id) {
        const loaded = await getVersion(activeDocumentId, result.id)
        setCurrentVersionId(loaded.id)
        const loadedMeta = normalizeMeta(loaded.metadata_json)
        setLatestVersionId(loaded.id)
        applyVersionState({
          id: loaded.id,
          json: loaded.doc_json || EMPTY_DOC,
          metadata: loadedMeta,
          versionNumber: loaded.version_number || currentVersion?.versionNumber || 1,
          timestamp: formatTimestamp(loaded.created_at),
        }, metadata.title || '')
      }
    } catch (error) {
      console.error('Failed to create new version:', error)
    }
  }

  const loadVersionHandler = useCallback(async (versionId) => {
    if (!documentId || !versionId) return

    try {
      const loaded = await getVersion(documentId, versionId)
      const loadedMeta = normalizeMeta(loaded.metadata_json)
      setCurrentVersionId(loaded.id)
      applyVersionState({
        id: loaded.id,
        json: loaded.doc_json || EMPTY_DOC,
        metadata: loadedMeta,
        versionNumber: loaded.version_number || 1,
        timestamp: formatTimestamp(loaded.created_at),
      }, metadata.title || '')
    } catch (error) {
      console.error('Failed to load version:', error)
    }
  }, [documentId, applyVersionState, metadata.title])

  const aiSopContext = useMemo(
    () => ({
      ...metadata,
      title: metadata?.title?.trim() || 'Untitled SOP',
      documentId: metadata?.documentId || documentId || 'SOP-NEW',
      sop_entity_id: documentId || null,
    }),
    [
      documentId,
      metadata?.title,
      metadata?.documentId,
      metadata?.department,
      metadata?.category,
      metadata?.sopVersion,
      metadata?.docType,
    ],
  )

  const openLinkModal = () => {
    if (!editor || !isEditorMounted || editor.isDestroyed) return
    setLinkModalInitialUrl(editor.getAttributes('link')?.href || '')
    setIsLinkModalOpen(true)
  }

  const handleLinkSave = (url) => {
    if (!editor || !isEditorMounted || editor.isDestroyed) return

    if (url) {
      editor.chain().focus().extendMarkRange('link').setLink({ href: url }).run()
    } else {
      editor.chain().focus().extendMarkRange('link').unsetLink().run()
    }

    setIsLinkModalOpen(false)
  }

  const triggerImport = async (event) => {
    const file = event.target.files?.[0]
    if (!file || !editor || !isEditorMounted || editor.isDestroyed) return

    const previousDoc = editor.getJSON()
    setIsImporting(true)
    setImportNotice('')
    try {
      // Let loading state render before extraction/mapping work starts.
      await new Promise((resolve) => window.requestAnimationFrame(resolve))
      const imported = await prepareEditorSOPImport(file)
      console.debug('[OCR] metadata received', imported.metadata || imported.response?.sop_metadata || {})
      // Yield one frame so UI can paint loading state before heavy content insert.
      await new Promise((resolve) => window.requestAnimationFrame(resolve))
      try {
        editor.commands.setContent(imported.docJson || imported.html, false)
      } catch (setContentError) {
        if (!imported.html) throw setContentError
        console.warn('[SOP Import] TipTap JSON import failed; retrying with HTML fallback.', setContentError)
        editor.commands.setContent(imported.html, false)
      }

      const ui = imported.metadata || {}
      if (ui && typeof ui === 'object') {
        console.debug('[OCR] metadata passed to SOPMetadataPanel', ui)
        const normalized = applySOPImportMetadata(DEFAULT_SOP_VERSION_METADATA.sopMetadata, ui)
        setMetadata((prev) => ({
          ...prev,
          ...normalized,
        }))
        const extractedStatus = (ui.sopStatus || ui.status || '').trim()
        if (extractedStatus) {
          setSopStatus(normalizeWorkflowStatus(extractedStatus) || extractedStatus)
        }
        const tabLabel = buildSOPDisplayLabel(ui)
        if (tabLabel && typeof onImportMetadataApplied === 'function') {
          onImportMetadataApplied({
            tabId: embedTabId,
            tabLabel,
            documentId,
          })
        }
      }

      setImportNotice('SOP imported successfully')
      window.setTimeout(() => setImportNotice(''), 2600)
    } catch (error) {
      console.error('Import failed:', error)
      setImportNotice(`Import failed: ${error?.message || 'Unknown error'}`)
      // Never leave editor in a broken visual state after failed import.
      // Restore previous content first; if that fails, fallback to empty doc.
      try {
        editor.commands.setContent(previousDoc || EMPTY_DOC, false)
      } catch {
        editor.commands.setContent(EMPTY_DOC, false)
      }
    } finally {
      setIsImporting(false)
      event.target.value = ''
    }
  }

  const insertPlaceholder = () => {
    if (!editor || !isEditorMounted || editor.isDestroyed || isHistoricalView) return
    const key = window.prompt('Enter placeholder name', 'DocumentOwner')
    if (!key?.trim()) return
    editor.chain().focus().insertContent(`{{${key.trim()}}}`).run()
  }

  const openCompareViewer = useCallback(async () => {
    if (!documentId || !compareBaseVersionId || !compareTargetVersionId) return

    try {
      const [baseVersion, targetVersion] = await Promise.all([
        getVersion(documentId, compareBaseVersionId),
        getVersion(documentId, compareTargetVersionId),
      ])

      const mapLoadedVersion = (version) => ({
        id: version.id,
        versionNumber: version.version_number || 1,
        json: version.doc_json || EMPTY_DOC,
      })

      setDiffVersions({
        oldVersion: mapLoadedVersion(baseVersion),
        newVersion: mapLoadedVersion(targetVersion),
      })
    } catch (error) {
      console.error('Failed to compare versions:', error)
    }
  }, [documentId, compareBaseVersionId, compareTargetVersionId])

  const openCompareFromAssistant = useCallback(async () => {
    if (!documentId) {
      const msg = 'Bitte öffnen Sie zuerst eine SOP im Editor.'
      window.alert(msg)
      throw new Error(msg)
    }
    if (!versions.length) {
      const msg = 'Keine Versionen zum Vergleichen.'
      window.alert(msg)
      throw new Error(msg)
    }
    const sorted = [...versions].sort(
      (a, b) => (Number(a.versionNumber) || 0) - (Number(b.versionNumber) || 0),
    )
    const base = sorted[0]
    const target = sorted[sorted.length - 1]
    if (!base?.id || !target?.id || base.id === target.id) {
      const msg =
        'Nur eine Version vorhanden. Bitte im Editor zwei Versionen in der Werkzeugleiste wählen und den Versionsvergleich dort starten.'
      window.alert(msg)
      throw new Error(msg)
    }
    try {
      const [baseVersion, targetVersion] = await Promise.all([
        getVersion(documentId, base.id),
        getVersion(documentId, target.id),
      ])
      setDiffVersions({
        oldVersion: {
          id: base.id,
          versionNumber: base.versionNumber || 1,
          json: baseVersion.doc_json || EMPTY_DOC,
        },
        newVersion: {
          id: target.id,
          versionNumber: target.versionNumber || 1,
          json: targetVersion.doc_json || EMPTY_DOC,
        },
      })
    } catch (error) {
      console.error('Failed to compare versions from assistant:', error)
      const msg = error?.message || 'Versionsvergleich fehlgeschlagen.'
      window.alert(msg)
      throw error instanceof Error ? error : new Error(msg)
    }
  }, [documentId, versions])

  if (!editor || !isEditorMounted || editor.isDestroyed) {
    return <div className="editor-loading">Loading editor...</div>
  }

  const references = Array.isArray(metadata.references) ? metadata.references : []
  const statusLabel = getLocalizedStatusLabel(sopStatus, t)
  const versionSelectValue = currentVersionId || (versions[0]?.id ?? '')
  const compareBaseValue = compareBaseVersionId || currentVersionId || (versions[0]?.id ?? '')
  const compareTargetValue = compareTargetVersionId || currentVersionId || (versions[0]?.id ?? '')
  return (
    <div className={`editor-page-container figma-editor-page${isEmbedded ? ' editor-embedded' : ''}`}>
      <div className="figma-shell">
        <aside className="figma-left-rail" aria-hidden="true">
          <div className="figma-rail-top"><div className="figma-rail-dot" /><div className="figma-rail-dot" /><div className="figma-rail-dot" /><div className="figma-rail-dot" /></div>
          <div className="figma-rail-bottom"><div className="figma-rail-dot" /><div className="figma-rail-dot" /><div className="figma-rail-dot" /></div>
        </aside>
        <div className="figma-workspace">
          <div className="figma-header-strip">
            <div className="figma-breadcrumb"><span>{breadcrumbLabel}</span></div>
            <div className="figma-header-actions">
              <button type="button" className="figma-sidebar-toggle" onClick={() => setIsSidebarOpen((prev) => !prev)} aria-expanded={isSidebarOpen} aria-controls="sop-metadata-sidebar" title={isSidebarOpen ? t.hideMetadataPanel : t.showMetadataPanel}>
                {isSidebarOpen ? <ChevronRight size={14} /> : <ChevronLeft size={14} />}
                <span>{isSidebarOpen ? t.hideMetadataPanel : t.showMetadataPanel}</span>
              </button>
              <button type="button" className="figma-icon-chip"><Eye size={15} /></button>
              <button type="button" className="figma-icon-chip"><Maximize2 size={15} /></button>
            </div>
          </div>
          <div className={`figma-layout${isSidebarOpen ? '' : ' sidebar-collapsed'}`}>
            <main className="figma-main-column">
              <EditorToolbarSection
                versionSelectValue={versionSelectValue}
                versions={versions}
                loadVersionHandler={loadVersionHandler}
                buildVersionLabel={buildVersionLabel}
                onPreviewOpen={() => setIsPreviewOpen(true)}
                onCreateVersion={createNewVersionHandler}
                onSave={() => persistDocument()}
                isHistoricalView={isHistoricalView}
                isSaving={isSaving}
                t={t}
                editor={editor}
                openLinkModal={openLinkModal}
                compareBaseValue={compareBaseValue}
                compareTargetValue={compareTargetValue}
                setCompareBaseVersionId={setCompareBaseVersionId}
                setCompareTargetVersionId={setCompareTargetVersionId}
                openCompareViewer={openCompareViewer}
                documentId={documentId}
                compareBaseVersionId={compareBaseVersionId}
                compareTargetVersionId={compareTargetVersionId}
                isImporting={isImporting}
                triggerImport={triggerImport}
                insertPlaceholder={insertPlaceholder}
                language={language}
                setLanguage={setLanguage}
              />
              {importNotice ? (
                <section className="figma-import-notice" role="status" aria-live="polite">
                  <CheckCircle2 size={15} />
                  <span>{importNotice}</span>
                </section>
              ) : null}
              {saveNotice ? (
                <section className="figma-import-notice" role="status" aria-live="polite">
                  <CheckCircle2 size={15} />
                  <span>{saveNotice}</span>
                </section>
              ) : null}
              <section className="figma-editor-canvas">
                {isHistoricalView ? <span className="editor-stage-hint">{t.historicalVersionLoaded}</span> : null}
                {isLoadingDocument ? <span className="editor-stage-hint">{t.loading}</span> : null}
                <EditorSurfaceErrorBoundary>
                  <EditorTypingSurface
                    editor={editor}
                    isEditable={!isHistoricalView && isEditorMounted}
                    aiSopContext={aiSopContext}
                    documentId={documentId}
                    onPreviewSessionChange={handleAiPreviewSessionChange}
                    onAfterApply={handleAfterAiBridgeApply}
                    onVersionCompareRequest={openCompareFromAssistant}
                  />
                </EditorSurfaceErrorBoundary>
              </section>
            </main>
            <aside id="sop-metadata-sidebar" className={`sop-sidebar figma-sidebar${isSidebarOpen ? '' : ' collapsed'}`} aria-hidden={!isSidebarOpen}>
            <SOPMetadataPanel
              metadata={metadata}
              onChange={handleMetadataPanelChange}
              status={sopStatus}
              onStatusChange={(value) => setSopStatus(normalizeWorkflowStatus(value) || value)}
              isReadOnly={isHistoricalView}
            />

            <div className="sidebar-card">
              <div className="sidebar-section-kicker">{t.referenceManagement}</div>
              <h3 className="sidebar-title">{t.sopReferences}</h3>
              <div className="reference-entry-row">
                <button type="button" className="sidebar-mini-btn success" onClick={addReference} disabled={isHistoricalView}>{t.add}</button>
              </div>
              <div className="sidebar-list">
                {references.length === 0 ? (
                  <p className="sidebar-empty-text">{t.noReferencesAdded}</p>
                ) : (
                  references.map((item, index) => (
                    <div key={`${item}-${index}`} className="reference-row">
                      <span>{item}</span>
                      <button type="button" className="sidebar-mini-btn danger" onClick={() => removeReference(index)} disabled={isHistoricalView}>{t.remove}</button>
                    </div>
                  ))
                )}
              </div>
            </div>

            <div className="sidebar-card">
              <div className="sidebar-section-kicker">{t.workflowAction}</div>
              <h3 className="sidebar-title">{t.sopActions}</h3>
              <div className="status-note">{t.currentStatus}: {statusLabel}</div>
              <textarea
                className="sidebar-textarea"
                value={versionNote}
                onChange={(event) => setVersionNote(event.target.value)}
                placeholder={t.sopNotePlaceholder}
                disabled={isHistoricalView}
              />
              <button type="button" className="sidebar-primary-btn" onClick={submitForReview} disabled={isHistoricalView}>{t.submitForReview}</button>
            </div>

            <div className="sidebar-card">
              <div className="sidebar-section-kicker">{t.lifecycle}</div>
              <h3 className="sidebar-title">{t.sopLifecycle}</h3>
              <div className="lifecycle-stack">
                {SOP_ORDER.map((stateId) => (
                  <div key={stateId} className={`lifecycle-pill${sopStatus === stateId ? ' active' : ''}`}>
                    <span>{getLocalizedStatusLabel(stateId, t)}</span>
                    {sopStatus === stateId ? <span>({t.current})</span> : null}
                  </div>
                ))}
              </div>
            </div>

            <div className="sidebar-card">
              <div className="sidebar-section-kicker">{t.changeLog}</div>
              <h3 className="sidebar-title">{t.sopAuditTrail}</h3>
              {auditTrail.length === 0 ? (
                <p className="sidebar-empty-text">{t.noAuditEntries}</p>
              ) : (
                <div className="audit-stack">
                  {auditTrail.slice().reverse().map((entry) => (
                    <div key={entry.id} className="audit-item">
                      <div className="audit-item-top">
                        <strong>v{entry.version || currentVersion?.versionNumber || 1}</strong>
                        <span>{getLocalizedStatusLabel(entry.toStatus, t)}</span>
                      </div>
                      <p>{entry.note || 'Workflow update'}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <RelatedContextSidebar
              sopId={documentId}
              onLinkClick={() => setIsLinkingModalOpen(true)}
              refreshToken={relatedContextRefreshToken}
            />
          </aside>
          </div>
        </div>
      </div>
        <EditorDocStatsBar
          editor={editor}
          lastSaved={lastSaved}
          isSaving={isSaving}
          profile="sop"
          onProfileChange={() => {}}
          workflowStatus={sopStatus}
        />

      <LinkModal
        isOpen={isLinkModalOpen}
        onClose={() => setIsLinkModalOpen(false)}
        onSave={handleLinkSave}
        initialUrl={linkModalInitialUrl}
      />

      <LinkingModal
        isOpen={isLinkingModalOpen}
        onClose={() => setIsLinkingModalOpen(false)}
        sourceId={documentId}
        sourceType="sop"
        onLinkCreated={() => {
          setRelatedContextRefreshToken((prev) => prev + 1)
        }}
      />

      {isPreviewOpen ? (
        <Suspense fallback={null}>
          <PreviewModal
            isOpen={isPreviewOpen}
            onClose={() => setIsPreviewOpen(false)}
            editor={editor}
            versionId={currentVersion?.versionNumber || 1}
          />
        </Suspense>
      ) : null}

      {diffVersions.oldVersion && diffVersions.newVersion ? (
        <Suspense fallback={null}>
          <SideBySideViewer
            oldVersion={diffVersions.oldVersion}
            newVersion={diffVersions.newVersion}
            onClose={() => setDiffVersions({ oldVersion: null, newVersion: null })}
          />
        </Suspense>
      ) : null}

      {!isEmbedded ? (
        <>
          <aside className={`ai-assistant-sidebar editor-ai-assistant-sidebar${aiWidgetOpen ? ' ai-sidebar-open' : ''}`}>
            <AIWidget />
          </aside>

          {aiWidgetOpen ? (
            <div
              className="ai-widget-overlay"
              onClick={() => setAiWidgetOpen(false)}
              aria-hidden="true"
            />
          ) : null}

          <FloatingAskAIButton onClick={() => setAiWidgetOpen((prev) => !prev)} isOpen={aiWidgetOpen} />
        </>
      ) : null}
    </div>
  )
}

export default EditorPage
