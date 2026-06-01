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

function deriveLineRows(text) {
  return String(text || '')
    .split(/\r?\n/)
    .map((line, index) => ({
      line_id: `ln_${index + 1}`,
      text: sanitizeText(line, 600),
      line_number: index + 1,
    }))
    .filter((row) => row.text)
}

function deriveSopSections(text) {
  const rows = deriveLineRows(text)
  if (!rows.length) return []

  const isHeading = (value) => {
    const line = String(value || '').trim()
    const cleanLine = line.replace(/^[^\p{L}\p{N}]+/u, '').trim()
    if (!line) return false
    if (/^(deviations?|abweichungen?|capas?|capa|audits?|audit findings?|entscheidungen?|decisions?)\b/i.test(cleanLine)) return true
    if (/\bzugeh(?:ö|oe|Ã¶)rig\s+zu\s+SOP-/i.test(cleanLine)) return true
    if (/^\d+(?:\.\d+)*[)\].:-]?\s+\S/u.test(line) && line.length < 180) return true
    if (/^(purpose|scope|procedure|responsibilities|definitions|approval|revision history|zweck|geltungsbereich|verfahren|verantwortlichkeiten)\b/i.test(line)) return true
    if (/^[A-ZÄÖÜ][A-ZÄÖÜ0-9\s/&()-]{4,}$/u.test(line) && line.length < 120) return true
    return false
  }

  const sections = []
  let current = null
  rows.forEach((row) => {
    if (isHeading(row.text)) {
      if (current) sections.push(current)
      current = {
        id: `sec_${sections.length + 1}`,
        label: row.text,
        order: sections.length + 1,
        content: row.text,
        lines: [row],
      }
      return
    }
    if (!current) {
      current = {
        id: 'sec_1',
        label: 'Document',
        order: 1,
        content: '',
        lines: [],
      }
    }
    current.lines.push(row)
    current.content = `${current.content ? `${current.content}\n` : ''}${row.text}`.trim()
  })
  if (current) sections.push(current)
  return sections.slice(0, 30).map((section) => ({
    ...section,
    content: sanitizeText(section.content, 2400),
    lines: section.lines.slice(0, 80),
  }))
}

function buildSelectedSectionPayload(editorState, sections) {
  const selected = editorState?.selected_section || {}
  const selectedName = sanitizeText(selected?.name || editorState?.selected_section_name || '', 160)
  const selectedText = sanitizeText(editorState?.selected_text || '', 2400)
  const matched = selectedName
    ? sections.find((section) => section.label.toLowerCase() === selectedName.toLowerCase())
    : null
  const content = selectedText || sanitizeText(matched?.content || selected?.text_excerpt || '', 2400)
  return {
    id: matched?.id || null,
    label: selectedName || matched?.label || null,
    content: content || null,
  }
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
  const editorText = sanitizeText(editorState?.editor_text, 14000)
  const sections = deriveSopSections(editorText)
  const selectedSectionPayload = buildSelectedSectionPayload(editorState, sections)
  const metadata = sanitizeObjectStrings(editorState?.sop?.metadata || {}, 220)
  const metadataJson = sanitizeObjectStrings(editorState?.sop?.metadata_json || {}, 220)

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
      owner: metadata?.owner || metadata?.author || metadata?.department || '',
      current_version_id: editorState?.sop?.current_version_id || '',
      status: editorState?.sop?.status || '',
      created_at: metadata?.createdAt || metadata?.created_at || null,
      updated_at: editorState?.updated_at || null,
      tags: Array.isArray(metadata?.tags) ? metadata.tags : [],
      compliance_standards: Array.isArray(metadata?.regulatoryReferences) ? metadata.regulatoryReferences : [],
      references: Array.isArray(editorState?.sop?.references) ? editorState.sop.references : [],
      metadata,
      metadata_json: metadataJson,
      sections,
      full_text: editorText,
      word_count: editorText ? editorText.split(/\s+/).filter(Boolean).length : 0,
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
      id: selectedSectionPayload.id,
      name: sanitizeText(
        editorState?.selected_section?.name
        || editorState?.selected_section_name
        || '',
        160,
      ),
      label: selectedSectionPayload.label,
      content: selectedSectionPayload.content,
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
    editor_excerpt: sanitizeText(editorState?.editor_text, 10000),
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
    last_focus:
      actionMemory?.last_focus && typeof actionMemory.last_focus === 'object'
        ? {
            target_scope: sanitizeText(actionMemory.last_focus.target_scope || '', 80),
            section_name: sanitizeText(actionMemory.last_focus.section_name || '', 160),
            sop_id: sanitizeText(actionMemory.last_focus.sop_id || '', 80),
            sop_number: sanitizeText(actionMemory.last_focus.sop_number || '', 80),
            sop_title: sanitizeText(actionMemory.last_focus.sop_title || '', 180),
            source: sanitizeText(actionMemory.last_focus.source || '', 60),
            updated_at: actionMemory.last_focus.updated_at || null,
          }
        : null,
    active_scope:
      actionMemory?.active_scope && typeof actionMemory.active_scope === 'object'
        ? actionMemory.active_scope
        : null,
    instruction_memory: Array.isArray(actionMemory?.instruction_memory)
      ? actionMemory.instruction_memory.slice(-12)
      : [],
    conversation_history: Array.isArray(actionMemory?.conversation_history)
      ? actionMemory.conversation_history.slice(-24)
      : [],
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

export function saveAssistantSessionSnapshot(snapshot = {}) {
  if (typeof window === 'undefined' || !snapshot || typeof snapshot !== 'object') return
  const existing = readLocalJson(KL_ASSISTANT_ACTION_MEMORY_KEY, {})
  try {
    localStorage.setItem(
      KL_ASSISTANT_ACTION_MEMORY_KEY,
      JSON.stringify({
        ...existing,
        updated_at: new Date().toISOString(),
        active_scope: snapshot.active_scope || existing.active_scope || null,
        instruction_memory: snapshot.instruction_memory || existing.instruction_memory || [],
        conversation_history: snapshot.conversation_history || existing.conversation_history || [],
      }),
    )
    window.dispatchEvent(new CustomEvent('sop-editor-context-changed'))
  } catch {
    // ignore
  }
}

/**
 * Remember the user's intended target for this chat session (before the editor finishes).
 * Enables follow-ups like "rewrite in 6 lines" without repeating the section name.
 */
export function recordAssistantTurnPlan({ action, targetScope, sectionName, requestPrompt, sopId } = {}) {
  if (typeof window === 'undefined') return
  const existing = readLocalJson(KL_ASSISTANT_ACTION_MEMORY_KEY, {})
  const section = sanitizeText(sectionName || '', 160)
  const scope = sanitizeText(targetScope || (section ? 'section' : ''), 80)
  const act = sanitizeText(action || '', 60)
  if (!act && !section && !scope) return
  saveAssistantLastAction({
    action: act || existing?.last_action?.action || '',
    target_scope: scope || existing?.last_action?.target_scope || '',
    section_name: section || existing?.last_action?.section_name || '',
    sop_id: sopId || existing?.last_action?.sop_id || getActiveEditorDocumentIdFromStorage(),
    request_prompt: sanitizeText(requestPrompt || '', 500),
    status: 'planned',
    source: 'sidebar_turn_plan',
  })
}

function getActiveEditorDocumentIdFromStorage() {
  try {
    return String(localStorage.getItem('current_document_id') || '').trim()
  } catch {
    return ''
  }
}

export function saveAssistantLastAction(action) {
  if (typeof window === 'undefined' || !action || typeof action !== 'object') return
  const existing = readLocalJson(KL_ASSISTANT_ACTION_MEMORY_KEY, {})
  const safe = {
    updated_at: new Date().toISOString(),
    action: sanitizeText(action.action || '', 60),
    target_scope: sanitizeText(action.target_scope || '', 80),
    section_name: sanitizeText(action.section_name || '', 160),
    section_id: sanitizeText(action.section_id || action.section_name || '', 160),
    sop_id: sanitizeText(action.sop_id || '', 80),
    sop_number: sanitizeText(action.sop_number || '', 80),
    sop_title: sanitizeText(action.sop_title || '', 180),
    request_prompt: sanitizeText(action.request_prompt || '', 500),
    original_text_excerpt: sanitizeText(action.original_text_excerpt || '', 1600),
    suggested_text_excerpt: sanitizeText(action.suggested_text_excerpt || '', 1600),
    status: sanitizeText(action.status || '', 60),
    source: sanitizeText(action.source || '', 60),
  }
  const activeScope = {
    section_id: safe.section_id || existing?.active_scope?.section_id || null,
    section_label: safe.section_name || existing?.active_scope?.section_label || null,
    last_action: safe.action || null,
    last_result: safe.suggested_text_excerpt || existing?.active_scope?.last_result || null,
    last_result_length: safe.suggested_text_excerpt
      ? safe.suggested_text_excerpt.split(/\s+/).filter(Boolean).length
      : Number(existing?.active_scope?.last_result_length || 0),
  }
  try {
    localStorage.setItem(
      KL_ASSISTANT_ACTION_MEMORY_KEY,
      JSON.stringify({
        updated_at: safe.updated_at,
        last_focus: existing?.last_focus || null,
        last_action: safe,
        active_scope: activeScope,
        instruction_memory: existing?.instruction_memory || [],
        conversation_history: existing?.conversation_history || [],
      }),
    )
    window.dispatchEvent(new CustomEvent('sop-editor-context-changed'))
  } catch {
    // Ignore storage errors.
  }
}

export function saveAssistantLastFocus(focus) {
  if (typeof window === 'undefined' || !focus || typeof focus !== 'object') return
  const existing = readLocalJson(KL_ASSISTANT_ACTION_MEMORY_KEY, {})
  const safe = {
    updated_at: new Date().toISOString(),
    target_scope: sanitizeText(focus.target_scope || '', 80),
    section_name: sanitizeText(focus.section_name || '', 160),
    sop_id: sanitizeText(focus.sop_id || '', 80),
    sop_number: sanitizeText(focus.sop_number || '', 80),
    sop_title: sanitizeText(focus.sop_title || '', 180),
    source: sanitizeText(focus.source || '', 60),
  }
  if (!safe.section_name && safe.target_scope !== 'full_document') return
  try {
    localStorage.setItem(
      KL_ASSISTANT_ACTION_MEMORY_KEY,
      JSON.stringify({
        updated_at: safe.updated_at,
        last_action: existing?.last_action || null,
        last_focus: safe,
      }),
    )
    window.dispatchEvent(new CustomEvent('sop-editor-context-changed'))
  } catch {
    // Ignore storage errors.
  }
}

export function clearAssistantLastAction() {
  if (typeof window === 'undefined') return
  try {
    localStorage.removeItem(KL_ASSISTANT_ACTION_MEMORY_KEY)
    window.dispatchEvent(new CustomEvent('sop-editor-context-changed'))
  } catch {
    // Ignore storage errors.
  }
}
