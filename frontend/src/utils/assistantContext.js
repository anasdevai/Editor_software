const KL_EDITOR_CONTEXT_KEY = 'kl_assistant_editor_state_v2'
const KL_WORKSPACE_CONTEXT_KEY = 'kl_assistant_workspace_state_v2'
const KL_ASSISTANT_ACTION_MEMORY_KEY = 'kl_assistant_action_memory_v1'
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

function sanitizeObjectStrings(input, maxLen = 240) {
  if (!input || typeof input !== 'object') return {}
  const out = {}
  Object.entries(input).forEach(([key, value]) => {
    if (value == null) return
    if (Array.isArray(value)) {
      out[key] = value.slice(0, 12).map((entry) => sanitizeText(entry, maxLen))
      return
    }
    if (typeof value === 'object') {
      out[key] = sanitizeObjectStrings(value, maxLen)
      return
    }
    out[key] = sanitizeText(value, maxLen)
  })
  return out
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
  const actionMemory = readLocalJson(KL_ASSISTANT_ACTION_MEMORY_KEY, {})
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
      current_version_id: editorState?.sop?.current_version_id || '',
      status: editorState?.sop?.status || '',
      references: Array.isArray(editorState?.sop?.references) ? editorState.sop.references : [],
      metadata: sanitizeObjectStrings(editorState?.sop?.metadata || {}, 220),
      metadata_json: sanitizeObjectStrings(editorState?.sop?.metadata_json || {}, 220),
    },
    selected_text: sanitizeText(editorState?.selected_text || '', 2400),
    selected_range:
      editorState?.selected_range && typeof editorState.selected_range === 'object'
        ? {
            from: Number(editorState.selected_range.from) || 0,
            to: Number(editorState.selected_range.to) || 0,
            empty: Boolean(editorState.selected_range.empty),
          }
        : null,
    selected_section: {
      name: sanitizeText(
        editorState?.selected_section?.name
        || editorState?.selected_section_name
        || '',
        160,
      ),
      type: sanitizeText(editorState?.selected_section?.type || '', 80),
      scope: sanitizeText(editorState?.selected_section?.scope || '', 80),
      text_excerpt: sanitizeText(editorState?.selected_section?.text_excerpt || '', 1200),
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
    last_action:
      actionMemory?.last_action && typeof actionMemory.last_action === 'object'
        ? {
            action: sanitizeText(actionMemory.last_action.action || '', 60),
            target_scope: sanitizeText(actionMemory.last_action.target_scope || '', 80),
            section_name: sanitizeText(actionMemory.last_action.section_name || '', 160),
            sop_id: sanitizeText(actionMemory.last_action.sop_id || '', 80),
            sop_number: sanitizeText(actionMemory.last_action.sop_number || '', 80),
            sop_title: sanitizeText(actionMemory.last_action.sop_title || '', 180),
            request_prompt: sanitizeText(actionMemory.last_action.request_prompt || '', 500),
            original_text_excerpt: sanitizeText(actionMemory.last_action.original_text_excerpt || '', 1600),
            suggested_text_excerpt: sanitizeText(actionMemory.last_action.suggested_text_excerpt || '', 1600),
            status: sanitizeText(actionMemory.last_action.status || '', 60),
            source: sanitizeText(actionMemory.last_action.source || '', 60),
            updated_at: actionMemory.last_action.updated_at || null,
          }
        : null,
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
    localStorage.removeItem(KL_ASSISTANT_ACTION_MEMORY_KEY)
    localStorage.setItem(ASSISTANT_STATE_RESET_MARKER, '1')
  } catch {
    // Ignore storage access failures.
  }
}

export function saveAssistantLastAction(action) {
  if (typeof window === 'undefined' || !action || typeof action !== 'object') return
  const safe = {
    updated_at: new Date().toISOString(),
    action: sanitizeText(action.action || '', 60),
    target_scope: sanitizeText(action.target_scope || '', 80),
    section_name: sanitizeText(action.section_name || '', 160),
    sop_id: sanitizeText(action.sop_id || '', 80),
    sop_number: sanitizeText(action.sop_number || '', 80),
    sop_title: sanitizeText(action.sop_title || '', 180),
    request_prompt: sanitizeText(action.request_prompt || '', 500),
    original_text_excerpt: sanitizeText(action.original_text_excerpt || '', 1600),
    suggested_text_excerpt: sanitizeText(action.suggested_text_excerpt || '', 1600),
    status: sanitizeText(action.status || '', 60),
    source: sanitizeText(action.source || '', 60),
  }
  try {
    localStorage.setItem(
      KL_ASSISTANT_ACTION_MEMORY_KEY,
      JSON.stringify({
        updated_at: safe.updated_at,
        last_action: safe,
      }),
    )
  } catch {
    // Ignore storage errors.
  }
}

export function clearAssistantLastAction() {
  if (typeof window === 'undefined') return
  try {
    localStorage.removeItem(KL_ASSISTANT_ACTION_MEMORY_KEY)
  } catch {
    // Ignore storage errors.
  }
}
