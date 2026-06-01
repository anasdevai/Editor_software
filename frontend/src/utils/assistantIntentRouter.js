/**
 * Sidebar transport for backend classify-intent — no keyword or alias logic here.
 */

import { classifyAssistantIntent } from '../api/editorApi'
import { queryEditorHasNonEmptySelection } from './editorActionsBridge'
import { getKLAssistantContext, saveAssistantLastAction, saveAssistantSessionSnapshot } from './assistantContext'
import { EDITOR_AI_ACTIONS, hasActiveSopEditor } from './editorAiBridge'

const ACTION_MAP = {
  rewrite: EDITOR_AI_ACTIONS.REWRITE,
  improve: EDITOR_AI_ACTIONS.IMPROVE,
  gap_check: EDITOR_AI_ACTIONS.GAP_CHECK,
  summarize: EDITOR_AI_ACTIONS.SUMMARIZE,
  analyze: EDITOR_AI_ACTIONS.ANALYZE,
  compare: EDITOR_AI_ACTIONS.COMPARE,
  read: EDITOR_AI_ACTIONS.READ,
}

const INLINE_CONTENT_ACTIONS = new Set([
  EDITOR_AI_ACTIONS.REWRITE,
  EDITOR_AI_ACTIONS.IMPROVE,
])

/**
 * @typedef {object} AssistantIntentClassification
 * @property {'chat'|'editor_action'|'clarify'|'follow_up_action'} flow
 * @property {string|null} action
 * @property {string|null} target_scope
 * @property {string|null} section_hint
 * @property {string|null} sidebar_intent — rag | sop_query | action | followup | clarify (from backend)
 * @property {boolean} run_editor_action
 * @property {boolean} run_query
 * @property {string|null} enriched_instruction
 * @property {object|null} target_resolution
 */

export async function classifyAssistantMessage({
  message,
  pathname = '/',
  recentMessages = [],
  assistantContextOverride = null,
}) {
  const text = String(message || '').trim()
  const assistantContext = assistantContextOverride || getKLAssistantContext(pathname)
  const hasActiveSop = hasActiveSopEditor(pathname)

  let hasEditorSelection = false
  if (hasActiveSop && typeof window !== 'undefined') {
    try {
      hasEditorSelection = await queryEditorHasNonEmptySelection(180)
    } catch {
      hasEditorSelection = false
    }
  }

  try {
    const result = await classifyAssistantIntent({
      message: text,
      route: pathname,
      has_active_sop: hasActiveSop,
      has_editor_selection: hasEditorSelection,
      recent_messages: Array.isArray(recentMessages) ? recentMessages.slice(-8) : [],
      assistant_context: assistantContext,
    })
    const normalized = normalizeClassification(result)
    if (result?.session_snapshot) {
      saveAssistantSessionSnapshot(result.session_snapshot)
    } else if (result?.active_scope) {
      saveAssistantSessionSnapshot({
        active_scope: result.active_scope,
        instruction_memory: result.instruction_memory || [],
        conversation_history: result.conversation_history || [],
      })
    }
    const scope = normalized.updated_active_scope || result?.active_scope
    if (
      normalized.run_editor_action
      && normalized.flow !== 'chat'
      && scope
      && typeof scope === 'object'
    ) {
      const sectionLabel = String(scope.section_label || normalized.section_hint || '').trim()
      if (sectionLabel || scope.last_action) {
        saveAssistantLastAction({
          action: String(scope.last_action || normalized.action || '').trim(),
          target_scope: String(normalized.target_scope || scope.target_scope || 'section').trim(),
          section_name: sectionLabel,
          section_id: String(scope.section_id || sectionLabel).trim(),
          request_prompt: text,
          status: 'classified',
          source: 'sidebar_classify',
        })
      }
    }
    return normalized
  } catch (err) {
    console.warn('[assistant-intent] classification failed, defaulting to chat', err)
    return {
      flow: 'chat',
      action: null,
      target_scope: null,
      section_hint: null,
      linked_entity_types: [],
      constraints: {},
      clarification_question: null,
      confidence: 0,
      reasoning: 'classifier_unavailable',
      sidebar_intent: 'rag',
      run_editor_action: false,
      run_query: true,
      enriched_instruction: text,
      target_resolution: null,
    }
  }
}

function normalizeClassification(raw) {
  const rawFlow = String(raw?.flow || '').trim().toLowerCase()
  const flow = ['chat', 'editor_action', 'clarify', 'follow_up_action'].includes(rawFlow) ? rawFlow : 'chat'
  const resolved = raw?.resolved_scope && typeof raw.resolved_scope === 'object' ? raw.resolved_scope : null
  const targetResolution =
    raw?.target_resolution && typeof raw.target_resolution === 'object' ? raw.target_resolution : null

  let targetScope = raw?.target_scope || targetResolution?.target_scope || null
  let sectionHint = raw?.section_hint || targetResolution?.section_hint || null
  if (resolved?.target_scope && !targetScope) targetScope = resolved.target_scope
  if (resolved?.section_label && !sectionHint) sectionHint = resolved.section_label
  if (targetScope === 'previous_suggestion') targetScope = 'section'

  const runEditor = Boolean(raw?.run_editor_action)
  let runQuery = Boolean(raw?.run_query)
  const sidebarIntent = String(raw?.sidebar_intent || '').trim().toLowerCase()
  if (flow === 'chat' || sidebarIntent === 'sop_query' || sidebarIntent === 'rag') {
    runQuery = true
  }
  if (runEditor && (flow === 'editor_action' || flow === 'follow_up_action')) {
    runQuery = false
  }

  return {
    flow,
    action: raw?.action || null,
    target_scope: targetScope,
    section_hint: sectionHint,
    line_number: raw?.line_number ?? targetResolution?.line_number ?? resolved?.line_number ?? null,
    record_id: raw?.record_id || targetResolution?.record_id || resolved?.record_id || null,
    linked_entity_types: Array.isArray(raw?.linked_entity_types) ? raw.linked_entity_types : [],
    constraints: raw?.constraints && typeof raw.constraints === 'object' ? raw.constraints : {},
    previous_action: raw?.previous_action && typeof raw.previous_action === 'object' ? raw.previous_action : null,
    clarification_question: raw?.clarification_question || null,
    confidence: typeof raw?.confidence === 'number' ? raw.confidence : 0.5,
    reasoning: raw?.reasoning || raw?.reason || null,
    frustration_signal: raw?.frustration_signal && typeof raw.frustration_signal === 'object' ? raw.frustration_signal : null,
    source_content_override: raw?.source_content_override && typeof raw.source_content_override === 'object'
      ? raw.source_content_override
      : null,
    repetition_instruction: raw?.repetition_instruction || null,
    updated_active_scope: raw?.updated_active_scope || raw?.active_scope || null,
    sidebar_intent: raw?.sidebar_intent || null,
    run_editor_action: flow === 'chat' ? false : runEditor,
    run_query: flow === 'chat' ? true : (runEditor ? false : runQuery),
    enriched_instruction: raw?.enriched_instruction || null,
    target_resolution: targetResolution,
    chat_submode: raw?.chat_submode || null,
    assistant_message: raw?.assistant_message || null,
    query_analysis: raw?.query_analysis || null,
  }
}

export function mapClassificationToEditorAction(classification) {
  const key = String(classification?.action || '').trim().toLowerCase()
  if (key === 'read') return null
  return ACTION_MAP[key] || null
}

/**
 * Map backend classification to editor execution plan (no local intent heuristics).
 */
export function planEditorActionExecution(classification, opts = {}) {
  const intent = opts.explicitAction || mapClassificationToEditorAction(classification)
  const tr = classification?.target_resolution || {}
  const sectionHint = String(
    tr.section_hint || classification?.section_hint || classification?.record_id || '',
  ).trim()
  const scope = String(
    tr.target_scope || classification?.target_scope || (sectionHint ? 'section' : 'selection'),
  ).trim().toLowerCase()

  const snapshotOptions = {
    sectionHint,
    targetScope: scope,
    preferFullSection: Boolean(tr.prefer_full_section),
  }

  if (!intent) {
    return {
      intent: null,
      inlineAction: null,
      useInline: false,
      useBridge: false,
      snapshotOptions,
    }
  }

  const c = classification?.constraints || {}
  let inlineAction = intent
  if (intent === EDITOR_AI_ACTIONS.REWRITE && (c.tone === 'formal' || c.detail_level)) {
    inlineAction = EDITOR_AI_ACTIONS.IMPROVE
  }

  const useBridge =
    intent === EDITOR_AI_ACTIONS.COMPARE
    || intent === EDITOR_AI_ACTIONS.READ
    || intent === EDITOR_AI_ACTIONS.ANALYZE

  const useInline = INLINE_CONTENT_ACTIONS.has(intent) && !useBridge

  return {
    intent,
    inlineAction,
    useInline,
    useBridge,
    snapshotOptions,
  }
}

/** Use server-built instruction; fallback to raw message only if missing. */
export function buildEnrichedActionPrompt(message, classification = {}) {
  const fromServer = String(classification?.enriched_instruction || '').trim()
  if (fromServer) return fromServer
  return String(message || '').trim()
}
