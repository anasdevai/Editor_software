/**
 * Bridge between the sidebar Actions tab and the live TipTap editor (snapshot,
 * inline diff decorations, accept/reject commits).
 */

export const EDITOR_SNAPSHOT_REQUEST_EVENT = 'editor-actions-snapshot-request'
export const EDITOR_SNAPSHOT_RESPONSE_EVENT = 'editor-actions-snapshot-response'
export const EDITOR_INLINE_SUGGESTION_SHOW_EVENT = 'editor-actions-inline-show'
export const EDITOR_INLINE_SUGGESTION_CLEAR_EVENT = 'editor-actions-inline-clear'
export const EDITOR_INLINE_SUGGESTION_APPLY_EVENT = 'editor-actions-inline-apply'
export const ACTIONS_TAB_RUN_EVENT = 'actions-tab-run-request'
export const EDITOR_SCROLL_TO_RANGE_EVENT = 'editor-actions-scroll-to-range'
export const EDITOR_GAP_APPEND_EVENT = 'editor-actions-gap-append'

/** Ask the mounted editor (EditorAIBridge) whether the current text selection is non-empty. */
export const EDITOR_SELECTION_QUERY_EVENT = 'kl-editor-selection-query'
export const EDITOR_SELECTION_RESPONSE_EVENT = 'kl-editor-selection-response'

const SNAPSHOT_TIMEOUT_MS = 4000

/**
 * @param {number} [timeoutMs]
 * @returns {Promise<boolean>}
 */
export function queryEditorHasNonEmptySelection(timeoutMs = 150) {
  if (typeof window === 'undefined') {
    return Promise.resolve(false)
  }
  const requestId = `sel-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`
  return new Promise((resolve) => {
    const timer = window.setTimeout(() => {
      window.removeEventListener(EDITOR_SELECTION_RESPONSE_EVENT, onResponse)
      resolve(false)
    }, timeoutMs)

    const onResponse = (event) => {
      const d = event.detail || {}
      if (d.requestId !== requestId) return
      window.clearTimeout(timer)
      window.removeEventListener(EDITOR_SELECTION_RESPONSE_EVENT, onResponse)
      resolve(Boolean(d.hasSelection))
    }

    window.addEventListener(EDITOR_SELECTION_RESPONSE_EVENT, onResponse)
    window.dispatchEvent(
      new CustomEvent(EDITOR_SELECTION_QUERY_EVENT, {
        detail: { requestId },
      }),
    )
  })
}

export function requestEditorSnapshot({
  prompt = '',
  userPrompt = '',
  sectionHint = '',
  targetScope = '',
  lineNumber = null,
  recordId = '',
  preferFullSection = false,
} = {}) {
  if (typeof window === 'undefined') {
    return Promise.reject(new Error('Editor snapshot is only available in the browser.'))
  }

  const requestId = `snap-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`

  return new Promise((resolve, reject) => {
    const timer = window.setTimeout(() => {
      window.removeEventListener(EDITOR_SNAPSHOT_RESPONSE_EVENT, onResponse)
      reject(new Error('Editor did not respond. Open an SOP in the editor first.'))
    }, SNAPSHOT_TIMEOUT_MS)

    const onResponse = (event) => {
      const detail = event.detail || {}
      if (detail.requestId !== requestId) return
      window.clearTimeout(timer)
      window.removeEventListener(EDITOR_SNAPSHOT_RESPONSE_EVENT, onResponse)
      if (detail.ok === false) {
        reject(new Error(detail.message || detail.error || 'Editor snapshot unavailable.'))
        return
      }
      resolve(detail)
    }

    window.addEventListener(EDITOR_SNAPSHOT_RESPONSE_EVENT, onResponse)
    window.dispatchEvent(
      new CustomEvent(EDITOR_SNAPSHOT_REQUEST_EVENT, {
        detail: {
          requestId,
          prompt: String(prompt || ''),
          userPrompt: String(userPrompt || '').trim(),
          sectionHint: String(sectionHint || '').trim(),
          targetScope: String(targetScope || '').trim().toLowerCase(),
          lineNumber: lineNumber ?? null,
          recordId: String(recordId || '').trim(),
          preferFullSection: Boolean(preferFullSection),
        },
      }),
    )
  })
}

export function showEditorInlineSuggestion(detail) {
  if (typeof window === 'undefined') return
  window.dispatchEvent(
    new CustomEvent(EDITOR_INLINE_SUGGESTION_SHOW_EVENT, {
      detail: detail || {},
    }),
  )
}

export function clearEditorInlineSuggestion(requestId) {
  if (typeof window === 'undefined') return
  window.dispatchEvent(
    new CustomEvent(EDITOR_INLINE_SUGGESTION_CLEAR_EVENT, {
      detail: { requestId },
    }),
  )
}

export function applyEditorInlineSuggestion(requestId) {
  if (typeof window === 'undefined') return
  window.dispatchEvent(
    new CustomEvent(EDITOR_INLINE_SUGGESTION_APPLY_EVENT, {
      detail: { requestId },
    }),
  )
}

export function dispatchActionsTabRun({
  action,
  prompt = '',
  userPrompt = '',
  sectionHint = '',
  targetScope = '',
  lineNumber = null,
  recordId = null,
  preferFullSection = false,
  sourceContentOverride = null,
} = {}) {
  if (typeof window === 'undefined') return
  window.dispatchEvent(
    new CustomEvent(ACTIONS_TAB_RUN_EVENT, {
      detail: {
        action,
        prompt,
        userPrompt: String(userPrompt || '').trim(),
        sectionHint: String(sectionHint || recordId || '').trim(),
        targetScope: String(targetScope || '').trim().toLowerCase(),
        lineNumber: lineNumber ?? null,
        recordId: recordId ? String(recordId).trim() : '',
        preferFullSection: Boolean(preferFullSection),
        sourceContentOverride: sourceContentOverride || null,
      },
    }),
  )
}

export function scrollEditorToRange(from, to) {
  if (typeof window === 'undefined') return
  window.dispatchEvent(
    new CustomEvent(EDITOR_SCROLL_TO_RANGE_EVENT, {
      detail: { from, to },
    }),
  )
}

export function appendGapFindingsToEditor(html) {
  if (typeof window === 'undefined') return
  window.dispatchEvent(
    new CustomEvent(EDITOR_GAP_APPEND_EVENT, {
      detail: { html: html || '' },
    }),
  )
}

export function subscribeActionsTabRun(handler) {
  return subscribeEditorEvent(ACTIONS_TAB_RUN_EVENT, handler)
}

export function subscribeEditorSnapshotRequest(handler) {
  if (typeof window === 'undefined' || typeof handler !== 'function') return () => {}
  const listener = (event) => {
    try {
      handler(event.detail || {})
    } catch (err) {
      console.error('[editor-actions-bridge] snapshot handler failed', err)
    }
  }
  window.addEventListener(EDITOR_SNAPSHOT_REQUEST_EVENT, listener)
  return () => window.removeEventListener(EDITOR_SNAPSHOT_REQUEST_EVENT, listener)
}

export function dispatchEditorSnapshotResponse(detail) {
  if (typeof window === 'undefined') return
  window.dispatchEvent(
    new CustomEvent(EDITOR_SNAPSHOT_RESPONSE_EVENT, {
      detail: detail || {},
    }),
  )
}

function subscribeEditorEvent(eventName, handler) {
  if (typeof window === 'undefined' || typeof handler !== 'function') return () => {}
  const listener = (event) => {
    try {
      handler(event.detail || {})
    } catch (err) {
      console.error(`[editor-actions-bridge] ${eventName} handler failed`, err)
    }
  }
  window.addEventListener(eventName, listener)
  return () => window.removeEventListener(eventName, listener)
}

export function subscribeEditorInlineSuggestionShow(handler) {
  return subscribeEditorEvent(EDITOR_INLINE_SUGGESTION_SHOW_EVENT, handler)
}

export function subscribeEditorInlineSuggestionClear(handler) {
  return subscribeEditorEvent(EDITOR_INLINE_SUGGESTION_CLEAR_EVENT, handler)
}

export function subscribeEditorInlineSuggestionApply(handler) {
  return subscribeEditorEvent(EDITOR_INLINE_SUGGESTION_APPLY_EVENT, handler)
}
