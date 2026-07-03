/**
 * Thin sidebar transport — all intent/scope logic lives in Python (classify-intent).
 */

import { classifyAssistantMessage } from './assistantIntentRouter'
import { getKLAssistantContext } from './assistantContext'
import { hasActiveSopEditor, requestAssistantEditorContextRefresh } from './editorAiBridge'

/**
 * Flush live editor context, call backend classify-intent, return response as-is.
 */
export async function prepareSidebarTurn({
  message,
  pathname = '/',
  recentMessages = [],
}) {
  const text = String(message || '').trim()
  const hasActiveSop = hasActiveSopEditor(pathname)

  if (hasActiveSop) {
    try {
      await requestAssistantEditorContextRefresh(400)
    } catch {
      // Editor not mounted; use last persisted context.
    }
  }

  const assistantContext = getKLAssistantContext(pathname)
  const classification = await classifyAssistantMessage({
    message: text,
    pathname,
    recentMessages,
    assistantContextOverride: assistantContext,
  })

  return {
    message: text,
    pathname,
    hasActiveSop,
    assistantContext,
    classification,
    runEditor: Boolean(classification.run_editor_action),
    runQuery: Boolean(classification.run_query),
    actionPrompt: classification.enriched_instruction || text,
  }
}

/** Pass backend target_resolution to editor snapshot (TipTap position mapping only). */
export function buildTargetOptionsFromClassification(classification, userMessage = '') {
  const tr =
    classification?.target_resolution && typeof classification.target_resolution === 'object'
      ? classification.target_resolution
      : {}

  return {
    userPrompt: String(userMessage || '').trim(),
    targetId: String(tr.target_id || '').trim(),
    targetType: String(tr.target_type || '').trim().toLowerCase(),
    targetLabel: String(tr.target_label || tr.section_hint || classification?.section_hint || '').trim(),
    owningSection: String(tr.owning_section || '').trim(),
    sectionHint: String(tr.section_hint || classification?.section_hint || classification?.record_id || '').trim(),
    targetScope: String(tr.target_scope || classification?.target_scope || 'selection').trim(),
    lineNumber: tr.line_number ?? classification?.line_number ?? null,
    recordId: String(tr.record_id || classification?.record_id || '').trim(),
    preferFullSection: Boolean(tr.prefer_full_section),
    sourceContentOverride: classification?.source_content_override || null,
  }
}
