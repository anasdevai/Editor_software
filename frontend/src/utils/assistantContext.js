const KL_EDITOR_CONTEXT_KEY = 'kl_assistant_editor_state_v2'
const KL_WORKSPACE_CONTEXT_KEY = 'kl_assistant_workspace_state_v2'
const ASSISTANT_STATE_RESET_MARKER = 'kl_assistant_state_reset_v2_applied'

const LEGACY_KEYS_TO_CLEAR = [
  'chat_page_conversations_v3_reset',
  'chat_page_active_conversation_v3_reset',
  'ai_widget_messages_by_path_v3_reset',
  'kl_assistant_editor_state_v1',
  'kl_assistant_workspace_state_v1',
  'current_document_id',
]

function readLocalJson(key, fallback) {
  try {
    const raw = localStorage.getItem(key)
    if (!raw) return fallback
    const parsed = JSON.parse(raw)
    return parsed && typeof parsed === 'object' ? parsed : fallback
  } catch {
    return fallback
  }
}

function sanitizeText(value, maxLen = 3000) {
  const text = String(value || '').trim()
  if (!text) return ''
  return text.length > maxLen ? `${text.slice(0, maxLen)}...` : text
}

function trimEntityList(items, limit = 6) {
  if (!Array.isArray(items)) return []
  return items.slice(0, limit).map((entry) => {
    if (!entry || typeof entry !== 'object') return {}
    return {
      id: entry.id || '',
      sop_number: entry.sop_number || entry.ref_number || '',
      deviation_number: entry.deviation_number || '',
      capa_number: entry.capa_number || '',
      audit_number: entry.audit_number || entry.finding_number || '',
      decision_number: entry.decision_number || '',
      title: sanitizeText(entry.title || entry.label || '', 120),
      status: sanitizeText(entry.status || entry.external_status || '', 40),
    }
  })
}

export function getKLAssistantContext(pathname = '/') {
  const editorState = readLocalJson(KL_EDITOR_CONTEXT_KEY, {})
  const workspaceState = readLocalJson(KL_WORKSPACE_CONTEXT_KEY, {})
  const activeDocumentId = localStorage.getItem('current_document_id') || ''
  const activeSopId = activeDocumentId || editorState?.sop?.id || ''
  const activeTabId = workspaceState?.active_tab_id || ''
  const workspaceEditorTabActive =
    typeof activeTabId === 'string' && activeTabId.startsWith('editor-')
  const onEditorSurface =
    pathname.startsWith('/editor') ||
    ((pathname === '/sops' || pathname.startsWith('/sops/')) && workspaceEditorTabActive)

  return {
    route: pathname,
    current_document_id: activeSopId,
    active_sop_id: onEditorSurface ? activeSopId : '',
    editor_surface_active: Boolean(activeSopId && onEditorSurface),
    current_sop: {
      id: activeSopId,
      sop_number: editorState?.sop?.sop_number || editorState?.sop?.documentId || '',
      title: editorState?.sop?.title || '',
      version: editorState?.sop?.version || '',
      status: editorState?.sop?.status || '',
      references: Array.isArray(editorState?.sop?.references) ? editorState.sop.references : [],
    },
    linked_context: {
      deviations: trimEntityList(editorState?.linked?.deviations, 6),
      capas: trimEntityList(editorState?.linked?.capas, 6),
      audits: trimEntityList(editorState?.linked?.audits, 6),
      decisions: trimEntityList(editorState?.linked?.decisions, 6),
      related_sops: trimEntityList(editorState?.linked?.related_sops, 6),
    },
    opened_tabs: trimEntityList(workspaceState?.opened_tabs, 8),
    active_tab_id: workspaceState?.active_tab_id || '',
    active_tab_label: workspaceState?.active_tab_label || '',
    editor_excerpt: sanitizeText(editorState?.editor_text, 5000),
    context_updated_at: editorState?.updated_at || workspaceState?.updated_at || null,
  }
}

export function getAssistantContextStorageKeys() {
  return {
    editor: KL_EDITOR_CONTEXT_KEY,
    workspace: KL_WORKSPACE_CONTEXT_KEY,
  }
}

export function resetAssistantStateOnce() {
  if (typeof window === 'undefined') return
  try {
    if (localStorage.getItem(ASSISTANT_STATE_RESET_MARKER) === '1') return
    LEGACY_KEYS_TO_CLEAR.forEach((key) => localStorage.removeItem(key))
    localStorage.setItem(ASSISTANT_STATE_RESET_MARKER, '1')
  } catch {
    // Ignore storage access failures.
  }
}
