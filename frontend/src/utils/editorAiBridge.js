/**
 * Editor ↔ KL/KI Assistant bridge.
 *
 * The assistant chat surfaces (AIWidget, ChatPage) dispatch a request event when
 * the user types an "editor action" intent (rewrite, improve, gap check, read).
 * The active EditorPage subscribes via {@link EditorAIBridge} and executes the
 * action against the open SOP, applying the result into the live editor.
 *
 * A `requestId` is round-tripped so the calling assistant can correlate the
 * eventual {@link EDITOR_AI_ACTION_RESULT_EVENT} status with the original
 * pending chat message.
 */

export const EDITOR_AI_ACTION_REQUEST_EVENT = 'kl-assistant-editor-action-request'
export const EDITOR_AI_ACTION_RESULT_EVENT = 'kl-assistant-editor-action-result'

/** Must match ``triggered_by`` sent to ``/api/ai/action`` (see ``[ai-action-prompt-source]`` logs). */
export const AI_ACTION_TRIGGERED_BY = Object.freeze({
  EDITOR_BUBBLE: 'editor_bubble',
  KL_ASSISTANT: 'kl_assistant',
  ACTIONS_TAB: 'actions_tab',
})

/** Status values reported back by the editor bridge to the assistant. */
export const EDITOR_AI_ACTION_STATUS = Object.freeze({
  APPLIED: 'applied',
  CANCELLED: 'cancelled',
  DISPLAYED: 'displayed',
  NOT_AVAILABLE: 'not_available',
  ERROR: 'error',
})

/** Editor-side actions that can be requested via the bridge. */
export const EDITOR_AI_ACTIONS = Object.freeze({
  REWRITE: 'rewrite',
  IMPROVE: 'improve',
  GAP_CHECK: 'gap_check',
  READ: 'read',
  SUMMARIZE: 'summarize',
  ANALYZE: 'analyze',
  COMPARE: 'compare',
})

const REWRITE_PATTERNS = [
  /\brewrite\b/i,
  /\bre[-\s]?write\b/i,
  /\brewriting\b/i,
  /\bumschreiben\b/i,
  /\bneu\s+schreiben\b/i,
  /\bneu\s+formulieren\b/i,
  /\bumformulieren\b/i,
  /\bschreibe\s+(die|diese)\s+sop\b/i,
]

const IMPROVE_PATTERNS = [
  /\bimprove\b/i,
  /\benhance\b/i,
  /\bpolish\b/i,
  /\boptimi[sz]e\b/i,
  /\brefine\b/i,
  /\bverbessere?n?\b/i,
  /\boptimieren?\b/i,
  /\bverfeinern\b/i,
  /\b(make|machen)\s+(it|sie|es|this)?\s*(besser|better|formal(?:er)?)\b/i,
  /\bmore\s+formal\b/i,
  /\bformeller\b/i,
  /\bformal(?:er|ere)?\s+machen\b/i,
  /\bimprovement\s+suggestions?\b/i,
  /\bverbesserungsvorschläge?\b/i,
  /\bgenerate\s+improvement\b/i,
]

const GAP_PATTERNS = [
  /\bgap\s*check\b/i,
  /\bgap\s*analysis\b/i,
  /\bwhat\s+(?:is|are)\s+the\s+gaps?\b/i,
  /\bwhat\s+gaps?\s+(?:are|exist)\b/i,
  /\bgaps?\s+in\s+(?:this\s+)?sop\b/i,
  /\bcompliance\s+(check|gap|review|audit)\b/i,
  /\bl(?:ü|ue|u)cken[\s-]?(analyse|pr(?:ü|ue|u)fung|check)?\b/i,
  /\b(?:finde|zeige|identifiziere|pr(?:ü|ue)fe)\s+(?:die\s+)?l(?:ü|ue|u)cken\b/i,
  /\bqa\s*review\b/i,
  /\bwelche\s+l(?:ü|ue|u)cken\b/i,
  /\bidentify\s+risks?\b/i,
  /\brisks?\s+and\s+gaps?\b/i,
  /\brisiken\s+und\s+l(?:ü|ue|u)cken\b/i,
  /\b(?:sop|dokument).*\bl(?:ü|ue|u)cken\b/i,
  /\bl(?:ü|ue|u)cken\b.*\b(?:sop|dokument)\b/i,
]

const READ_PATTERNS = [
  /\bread\s+(this|the|current)?\s*sop\b/i,
  /\b(show|display|open)\s+(this|the|current)?\s*sop\b/i,
  /\blese?\s+(die|diese|aktuelle)?\s*sop\b/i,
  /\bzeige?\s+(die|diese|aktuelle)?\s*sop\b/i,
  /\b(öffne|oeffne)\s+(die|diese|aktuelle)?\s*sop\b/i,
  /\binhalt\s+(der\s+)?(aktuellen\s+)?sop\b/i,
]

const SUMMARIZE_PATTERNS = [
  /\bsummarize\b/i,
  /\bsummary\b/i,
  /\bexecutive\s+summary\b/i,
  /\bzusammenfass/i,
  /\bkurzfassung\b/i,
  /\bfasse\s+zusammen\b/i,
  /\bverkürz/i,
  /\bsummarize\s+this\s+sop\s+in\s+\d+\s+words?\b/i,
  /\bzusammenfass(?:en|ung).*\b\d+\s+wörter\b/i,
  /\b(?:summarize|zusammenfass).*\b(?:section|abschnitt)\b/i,
  /\b(?:abschnitt|section)\s+.*\b(?:summarize|zusammenfass)\b/i,
]

const ANALYZE_PATTERNS = [
  /\banalyze\b/i,
  /\banalyse\b/i,
  /\bcompliance\s+analysis\b/i,
  /\bcompliance\s+of\s+this\s+sop\b/i,
  /\bcompliance\s+review\b/i,
]

const COMPARE_PATTERNS = [
  /\bcompare\s+(sop\s+)?versions?\b/i,
  /\bversion\s+compare\b/i,
  /\bversionsvergleich\b/i,
  /\bcompare\s+versions?\s+for\b/i,
]

/**
 * Legacy regex intent detector (bubble menu / offline fallback only).
 * Prefer {@link classifyAssistantMessage} from assistantIntentRouter.js for chat routing.
 * Returns one of EDITOR_AI_ACTIONS values, or null when no intent matches.
 */
export function detectEditorIntent(rawText) {
  const text = String(rawText || '').trim()
  if (!text) return null
  for (const pattern of GAP_PATTERNS) {
    if (pattern.test(text)) return EDITOR_AI_ACTIONS.GAP_CHECK
  }
  for (const pattern of COMPARE_PATTERNS) {
    if (pattern.test(text)) return EDITOR_AI_ACTIONS.COMPARE
  }
  for (const pattern of SUMMARIZE_PATTERNS) {
    if (pattern.test(text)) return EDITOR_AI_ACTIONS.SUMMARIZE
  }
  for (const pattern of ANALYZE_PATTERNS) {
    if (pattern.test(text)) return EDITOR_AI_ACTIONS.ANALYZE
  }
  for (const pattern of REWRITE_PATTERNS) {
    if (pattern.test(text)) return EDITOR_AI_ACTIONS.REWRITE
  }
  for (const pattern of IMPROVE_PATTERNS) {
    if (pattern.test(text)) return EDITOR_AI_ACTIONS.IMPROVE
  }
  for (const pattern of READ_PATTERNS) {
    if (pattern.test(text)) return EDITOR_AI_ACTIONS.READ
  }
  return null
}

/** Map quick-action chip id (AIWidget) to a concrete {@link EDITOR_AI_ACTIONS} value, or null. */
export function editorActionFromChipId(chipId) {
  const id = String(chipId || '').trim().toLowerCase()
  const map = {
    rewrite: EDITOR_AI_ACTIONS.REWRITE,
    improve: EDITOR_AI_ACTIONS.IMPROVE,
    gap: EDITOR_AI_ACTIONS.GAP_CHECK,
    summarize: EDITOR_AI_ACTIONS.SUMMARIZE,
    analyze: EDITOR_AI_ACTIONS.ANALYZE,
    compare: EDITOR_AI_ACTIONS.COMPARE,
  }
  return map[id] || null
}

/**
 * Active SOP document UUID for editor bridge: prefers `current_document_id`, then
 * KL assistant editor context written by {@link EditorPage}.
 */
export function getActiveEditorDocumentId() {
  if (typeof window === 'undefined') return ''
  try {
    const direct = String(localStorage.getItem('current_document_id') || '').trim()
    if (direct) return direct
    const raw = localStorage.getItem('kl_assistant_editor_state_v2')
    if (!raw) return ''
    const parsed = JSON.parse(raw)
    const sid = parsed?.sop?.id
    return sid && String(sid).trim() ? String(sid).trim() : ''
  } catch {
    return ''
  }
}

/**
 * True when the supplied route is the SOP editor surface (e.g. `/editor`,
 * `/editor/:id`). Used by chat surfaces to decide whether to bridge into the
 * editor or fall back to the regular RAG chat flow.
 */
export function isEditorRoute(pathname) {
  const path = String(pathname || '')
  return path === '/editor' || path.startsWith('/editor/')
}

/**
 * True when an SOP is open in the editor (dedicated /editor route OR embedded tab on /sops).
 * The sidebar Actions flow depends on this — not only on the URL path.
 */
export function hasActiveSopEditor(pathname) {
  const docId = getActiveEditorDocumentId()
  if (!docId) return false
  const path = String(pathname || '')
  if (isEditorRoute(path)) return true
  if (path === '/sops' || path.startsWith('/sops')) return true
  return false
}

export const SOP_EDITOR_CONTEXT_EVENT = 'sop-editor-context-changed'
export const KL_ASSISTANT_CONTEXT_REFRESH_REQUEST = 'kl-assistant-context-refresh-request'
export const KL_ASSISTANT_CONTEXT_REFRESH_DONE = 'kl-assistant-context-refresh-done'

export function notifySopEditorContextChanged() {
  if (typeof window === 'undefined') return
  window.dispatchEvent(new CustomEvent(SOP_EDITOR_CONTEXT_EVENT))
}

/**
 * Ask the open editor to flush the latest SOP text/selection into assistant context
 * before classify-intent or /api/ai/query (avoids stale 1.2s debounced snapshots).
 * @param {number} [timeoutMs]
 * @returns {Promise<boolean>} true when editor acknowledged refresh
 */
export function requestAssistantEditorContextRefresh(timeoutMs = 500) {
  if (typeof window === 'undefined') {
    return Promise.resolve(false)
  }
  if (!getActiveEditorDocumentId()) {
    return Promise.resolve(false)
  }

  const requestId = `ctx-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`

  return new Promise((resolve) => {
    const timer = window.setTimeout(() => {
      window.removeEventListener(KL_ASSISTANT_CONTEXT_REFRESH_DONE, onDone)
      resolve(false)
    }, timeoutMs)

    const onDone = (event) => {
      const d = event.detail || {}
      if (d.requestId !== requestId) return
      window.clearTimeout(timer)
      window.removeEventListener(KL_ASSISTANT_CONTEXT_REFRESH_DONE, onDone)
      resolve(d.ok !== false)
    }

    window.addEventListener(KL_ASSISTANT_CONTEXT_REFRESH_DONE, onDone)
    window.dispatchEvent(
      new CustomEvent(KL_ASSISTANT_CONTEXT_REFRESH_REQUEST, {
        detail: { requestId },
      }),
    )
  })
}

/** Lightweight, opaque request id. */
export function makeEditorAiRequestId() {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return `eai-${crypto.randomUUID()}`
  }
  return `eai-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`
}

/**
 * Dispatch an editor-action request from a chat surface.
 *
 * @param {object} opts
 * @param {string} opts.action      One of EDITOR_AI_ACTIONS values.
 * @param {string} [opts.prompt]    Original user prompt for logging.
 * @param {string} [opts.requestId] Optional, generated when omitted.
 * @param {string} [opts.source]    Origin tag (e.g. 'kl_assistant', 'chat_page').
 * @returns {string} The request id that result events will echo back.
 */
export function dispatchEditorAiActionRequest({
  action,
  prompt = '',
  userPrompt = '',
  sectionHint = '',
  targetScope = '',
  lineNumber = null,
  recordId = '',
  preferFullSection = false,
  requestId,
  source = 'kl_assistant',
} = {}) {
  if (typeof window === 'undefined') return ''
  const id = requestId || makeEditorAiRequestId()
  console.info('[kl-editor-bridge-dispatch]', { action, requestId: id, source, promptLen: String(prompt || '').length })
  window.dispatchEvent(
    new CustomEvent(EDITOR_AI_ACTION_REQUEST_EVENT, {
      detail: {
        action,
        prompt,
        userPrompt: String(userPrompt || '').trim(),
        sectionHint: String(sectionHint || '').trim(),
        targetScope: String(targetScope || '').trim().toLowerCase(),
        lineNumber: lineNumber ?? null,
        recordId: String(recordId || '').trim(),
        preferFullSection: Boolean(preferFullSection),
        requestId: id,
        source,
      },
    }),
  )
  return id
}

/** Subscribe to incoming editor-action requests. Returns an unsubscribe fn. */
export function subscribeEditorAiActionRequest(handler) {
  if (typeof window === 'undefined' || typeof handler !== 'function') return () => {}
  const listener = (event) => {
    try {
      handler(event.detail || {})
    } catch (err) {
      console.error('[editor-ai-bridge] request handler failed', err)
    }
  }
  window.addEventListener(EDITOR_AI_ACTION_REQUEST_EVENT, listener)
  return () => window.removeEventListener(EDITOR_AI_ACTION_REQUEST_EVENT, listener)
}

/**
 * Dispatch a status update from the editor bridge back to the requesting
 * assistant surface.
 */
export function dispatchEditorAiActionResult(detail) {
  if (typeof window === 'undefined') return
  window.dispatchEvent(
    new CustomEvent(EDITOR_AI_ACTION_RESULT_EVENT, {
      detail: detail || {},
    }),
  )
}

/** Subscribe to status updates emitted by the editor bridge. */
export function subscribeEditorAiActionResult(handler) {
  if (typeof window === 'undefined' || typeof handler !== 'function') return () => {}
  const listener = (event) => {
    try {
      handler(event.detail || {})
    } catch (err) {
      console.error('[editor-ai-bridge] result handler failed', err)
    }
  }
  window.addEventListener(EDITOR_AI_ACTION_RESULT_EVENT, listener)
  return () => window.removeEventListener(EDITOR_AI_ACTION_RESULT_EVENT, listener)
}

const STATUS_MESSAGES_DE = {
  rewrite_applied: 'Die SOP wurde im Editor neu geschrieben. Bitte prüfen und speichern.',
  rewrite_cancelled: 'Rewrite verworfen. Der Editor bleibt unverändert.',
  improve_applied: 'Verbesserungen wurden im Editor übernommen. Bitte prüfen und speichern.',
  improve_cancelled: 'Verbesserung verworfen. Der Editor bleibt unverändert.',
  gap_check_applied: 'Gap-Check-Findings als Anhang in die SOP eingefügt. Bitte prüfen und speichern.',
  gap_check_cancelled: 'Gap Check geschlossen. Der Editor bleibt unverändert.',
  summarize_applied: 'Zusammenfassung wurde in die SOP übernommen. Bitte prüfen und speichern.',
  summarize_cancelled: 'Zusammenfassung verworfen. Der Editor bleibt unverändert.',
  analyze_applied: 'Analyse wurde in die SOP übernommen. Bitte prüfen und speichern.',
  analyze_cancelled: 'Analyse verworfen. Der Editor bleibt unverändert.',
  read_displayed: 'Die aktive SOP ist bereits im Editor geladen.',
  compare_displayed: 'Versionsvergleich wurde geöffnet.',
  not_available: 'Keine SOP im Editor geöffnet. Bitte öffne zuerst eine SOP, bevor du Editor-Aktionen nutzt.',
  error: 'Editor-Aktion fehlgeschlagen.',
}

/**
 * Build a short, user-facing chat status line for a bridge result. Returned
 * text is German to match the existing assistant UI strings; callers may
 * append additional details (e.g. error.message).
 */
export function describeEditorAiResult(detail) {
  if (!detail || typeof detail !== 'object') return STATUS_MESSAGES_DE.error
  const { action, status, message } = detail
  if (status === EDITOR_AI_ACTION_STATUS.NOT_AVAILABLE) {
    return STATUS_MESSAGES_DE.not_available
  }
  if (status === EDITOR_AI_ACTION_STATUS.ERROR) {
    return message ? `${STATUS_MESSAGES_DE.error} ${message}` : STATUS_MESSAGES_DE.error
  }
  if (action === EDITOR_AI_ACTIONS.READ) return STATUS_MESSAGES_DE.read_displayed
  if (action === EDITOR_AI_ACTIONS.COMPARE && status === EDITOR_AI_ACTION_STATUS.DISPLAYED) {
    return STATUS_MESSAGES_DE.compare_displayed
  }
  const key = `${action}_${status === EDITOR_AI_ACTION_STATUS.APPLIED ? 'applied' : 'cancelled'}`
  return STATUS_MESSAGES_DE[key] || STATUS_MESSAGES_DE.error
}
