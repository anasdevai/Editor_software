/**
 * Semantic intent routing for the unified KL/KI Assistant chat panel.
 * Uses backend LLM classification — not fixed keyword matching.
 */

import { classifyAssistantIntent } from '../api/editorApi'
import { queryEditorHasNonEmptySelection } from './editorActionsBridge'
import { getKLAssistantContext, saveAssistantSessionSnapshot } from './assistantContext'
import { EDITOR_AI_ACTIONS, hasActiveSopEditor } from './editorAiBridge'
import { wantsSectionScopeIntent } from './editorTargetResolver.js'

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
  EDITOR_AI_ACTIONS.SUMMARIZE,
])

/**
 * @typedef {object} AssistantIntentClassification
 * @property {'chat'|'editor_action'|'clarify'} flow
 * @property {string|null} action
 * @property {string|null} target_scope
 * @property {string|null} section_hint
 * @property {string[]} linked_entity_types
 * @property {object} constraints
 * @property {object|null} previous_action
 * @property {string|null} clarification_question
 * @property {number} confidence
 */

/**
 * @typedef {object} EditorActionExecutionPlan
 * @property {string|null} intent — bridge action id
 * @property {string|null} inlineAction — action passed to inline /api/ai/action runner
 * @property {boolean} useInline — show inline diff + Accept/Reject in editor
 * @property {boolean} useBridge — legacy modal bridge (full-doc analysis only)
 * @property {{ sectionHint: string, targetScope: string }} snapshotOptions
 */

/**
 * Call backend semantic intent classifier.
 * @returns {Promise<AssistantIntentClassification>}
 */
export async function classifyAssistantMessage({ message, pathname = '/', recentMessages = [] }) {
  const text = String(message || '').trim()
  const assistantContext = getKLAssistantContext(pathname)
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
    }
  }
}

function normalizeClassification(raw) {
  const rawFlow = String(raw?.flow || '').trim().toLowerCase()
  const flow = ['chat', 'editor_action', 'clarify', 'follow_up_action'].includes(rawFlow) ? rawFlow : 'chat'
  const resolved = raw?.resolved_scope && typeof raw.resolved_scope === 'object' ? raw.resolved_scope : null
  let targetScope = raw?.target_scope || null
  let sectionHint = raw?.section_hint || null
  if (resolved?.target_scope) targetScope = resolved.target_scope
  if (resolved?.section_label && !sectionHint) sectionHint = resolved.section_label
  if (targetScope === 'previous_suggestion') targetScope = 'section'
  return {
    flow,
    action: raw?.action || null,
    target_scope: targetScope,
    section_hint: sectionHint,
    line_number: raw?.line_number || resolved?.line_number || null,
    record_id: raw?.record_id || resolved?.record_id || null,
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
  }
}

/** Map classifier action id to editor bridge action constant. */
export function mapClassificationToEditorAction(classification) {
  const key = String(classification?.action || '').trim().toLowerCase()
  if (key === 'read') return null
  return ACTION_MAP[key] || null
}

/**
 * Decide inline editor diff vs modal bridge from semantic classification (not keywords).
 * @returns {EditorActionExecutionPlan}
 */
export function planEditorActionExecution(classification, opts = {}) {
  const intent = opts.explicitAction || mapClassificationToEditorAction(classification)
  const rawScope = String(classification?.target_scope || '').toLowerCase()
  const sectionHint = String(
    classification?.section_hint || classification?.record_id || '',
  ).trim()
  const userMessage = String(opts.userMessage || '').trim()
  const scope =
    rawScope === 'full_document'
      ? 'full_document'
      : sectionHint || classification?.record_id
      ? 'section'
      : wantsSectionScopeIntent(userMessage)
      ? 'section'
      : rawScope === 'previous_suggestion' || rawScope === 'current_section'
      ? 'section'
      : rawScope
  const snapshotOptions = {
    sectionHint,
    targetScope: scope,
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

/**
 * Enrich the user instruction with extracted constraints and target hints
 * for the existing editor snapshot / action handlers.
 */
export function buildEnrichedActionPrompt(message, classification = {}) {
  const base = String(message || '').trim()
  if (!base) return ''

  const hints = []
  const c = classification.constraints || {}

  hints.push('Output language: keep the same language as the target SOP text/profile unless the user explicitly requests another language.')
  if (c.tone) hints.push(`Tone: ${c.tone}`)
  if (c.word_count) hints.push(`Target length: about ${c.word_count} words`)
  if (c.line_count) hints.push(`Keep the answer to roughly ${c.line_count} lines (short lines / bullets acceptable).`)
  if (c.length === 'shorter') hints.push('Make the result shorter than the source.')
  if (c.length === 'longer') hints.push('Expand the result with more detail.')
  if (c.language) hints.push(`Output language: ${c.language}`)
  if (c.detail_level) hints.push(`Detail level: ${c.detail_level}`)
  if (c.format) hints.push(`Format: ${c.format}`)

  if (classification.section_hint) {
    hints.push(`Target section: ${classification.section_hint}`)
    hints.push(
      'Apply to the complete section body under that heading (all paragraphs until the next section), not the heading line alone.',
    )
  }
  if (classification.previous_action && typeof classification.previous_action === 'object') {
    const prev = classification.previous_action
    if (prev.action) hints.push(`Previous assistant action: ${prev.action}`)
    if (prev.section_name && !classification.section_hint) {
      hints.push(`Continue working on the same target section: ${prev.section_name}`)
    }
    if (prev.target_scope) hints.push(`Previous target scope: ${prev.target_scope}`)
    if (prev.request_prompt) hints.push(`Previous instruction: ${prev.request_prompt}`)
  }
  if (classification.target_scope === 'full_document') {
    hints.push('Apply to the entire SOP document.')
  }
  if (classification.target_scope === 'selection') {
    hints.push('Apply only to the current editor selection.')
  }
  if (classification.line_number) {
    hints.push(`Target line number: ${classification.line_number}`)
  }
  if (classification.record_id) {
    hints.push(`Target record entry: ${classification.record_id}`)
  }
  if (classification.source_content_override?.enabled && classification.source_content_override?.content) {
    hints.push('Operate on the previous assistant output for this section, not the original SOP source.')
  }
  if (classification.repetition_instruction) {
    hints.push(classification.repetition_instruction)
  }
  if (classification.frustration_signal?.detected) {
    if (classification.frustration_signal.target_word_count) {
      hints.push(`Target word count: about ${classification.frustration_signal.target_word_count} words`)
    }
    hints.push('User is refining the previous result — keep the same section target.')
  }
  if (classification.target_scope === 'linked_context' && classification.linked_entity_types?.length) {
    hints.push(`Focus on linked records: ${classification.linked_entity_types.join(', ')}`)
  }

  if (!hints.length) return base
  return `${base}\n\n[Assistant constraints]\n${hints.join('\n')}`
}
