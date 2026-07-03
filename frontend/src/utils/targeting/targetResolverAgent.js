import {
  buildEditorSectionIndex,
  buildEditorTableIndex,
  resolveTargetInEditor,
} from '../editorTargetResolver.js'
import { classifyTargetIntent } from './intentClassifier.js'
import { parseSopDocument } from './sopParser.js'

const STRUCTURAL_SCOPES = new Set(['section', 'subsection', 'table', 'table_section', 'full_document'])

const normalize = (value = '') =>
  String(value || '')
    .replace(/[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}]/gu, ' ')
    .replace(/^\s*\d+(?:\.\d+)*[.)\]:-]?\s*/, '')
    .replace(/[^\p{L}\p{N}\s.-]/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .toLowerCase()

const promptNamesStructuralTarget = (value = '') =>
  /\b(?:section|sections|heading|table|tabelle|abschnitt|kapitel)\b/i.test(String(value || ''))
  || /["'][^"']{2,160}["']/.test(String(value || ''))
  || /\b\d+(?:\.\d+)+\b/.test(String(value || ''))

const explicitlyUsesSelection = (value = '') =>
  /\b(?:selected\s+(?:text|word|paragraph|section|sentence|line|content)|selection|highlighted|marked\s+text|this\s+selection)\b/i
    .test(String(value || ''))

const requestsUnsupportedTableSpan = (value = '') =>
  /\b(?:row|cell|zeile|zelle)\b/i.test(String(value || ''))

const rangesOverlap = (a = {}, b = {}) =>
  Number.isFinite(a.from)
  && Number.isFinite(a.to)
  && Number.isFinite(b.from)
  && Number.isFinite(b.to)
  && a.from < b.to
  && b.from < a.to

const tokenSet = (value = '') =>
  new Set(normalize(value).split(/\s+/).filter((token) => token.length >= 3))

const overlapScore = (query, label) => {
  const q = tokenSet(query)
  const l = tokenSet(label)
  if (!q.size || !l.size) return 0
  let hits = 0
  for (const token of q) if (l.has(token)) hits += 1
  return hits / Math.max(q.size, l.size)
}

const exactLabelScore = (query, label) => {
  const q = normalize(query)
  const l = normalize(label)
  if (!q || !l) return 0

  if (q === l) return 1
  if (q.includes(l) || l.includes(q)) return 0.88
  return overlapScore(q, l)
}

const isTableScope = (scope = '') => ['table', 'table_section'].includes(String(scope || '').toLowerCase())

function scoreTableCandidate(userQuery = '', table = {}) {
  const label = [table.caption, table.owningSection].filter(Boolean).join(' - ')
  return Math.max(
    exactLabelScore(userQuery, table.caption || ''),
    exactLabelScore(userQuery, table.owningSection || '') * 0.96,
    exactLabelScore(userQuery, label),
    overlapScore(userQuery, `${label} ${table.text || ''}`) * 0.82,
  )
}

function resolveTableTargetFromIndex(editor, userQuery = '') {
  const tables = buildEditorTableIndex(editor)
    .map((table) => ({ table, score: scoreTableCandidate(userQuery, table) }))
    .filter(({ score }) => score >= 0.34)
    .sort((a, b) => b.score - a.score || a.table.from - b.table.from)
  if (!tables.length) return null
  const { table, score } = tables[0]
  const text = editor?.state?.doc?.textBetween?.(table.from, table.to, '\n')?.trim() || table.text || ''
  return {
    from: table.from,
    to: table.to,
    text,
    isFullDoc: false,
    sectionName: table.caption || table.owningSection || `Table ${table.index + 1}`,
    sectionType: 'Table',
    confidence: Math.max(0.84, Number(score.toFixed(2))),
    resolved_target_type: 'table',
    match_reason: 'table_index_semantic_match',
  }
}

function candidateList(editor, userQuery = '', { tableOnly = false, sectionOnly = false } = {}) {
  const sections = tableOnly ? [] : buildEditorSectionIndex(editor).map((section) => {
    const title = section.sectionName || section.title || section.heading || ''
    const score = Math.max(
      exactLabelScore(userQuery, title),
      overlapScore(userQuery, `${title} ${section.text || ''}`) * 0.7,
    )
    return {
      id: section.id || `sec_${section.from}`,
      label: title,
      type: 'section',
      score: Number(score.toFixed(3)),
      from: section.from,
      to: section.to,
    }
  })

  const tables = sectionOnly ? [] : buildEditorTableIndex(editor).map((table) => {
    const label = [table.caption, table.owningSection].filter(Boolean).join(' - ')
    const score = scoreTableCandidate(userQuery, table)
    return {
      id: table.id || `table_${table.from}`,
      label: label || `Table at ${table.from}`,
      type: 'table',
      score: Number(score.toFixed(3)),
      from: table.from,
      to: table.to,
    }
  })

  return [...sections, ...tables]
    .filter((candidate) => candidate.score > 0.18)
    .sort((a, b) => b.score - a.score || a.from - b.from)
    .slice(0, 6)
}

function targetIdFor(target) {
  if (target?.target_id) return target.target_id
  const base = target?.resolved_target_type || target?.sectionType || 'target'
  const label = normalize(target?.sectionName || target?.text || '')
    .replace(/\s+/g, '_')
    .slice(0, 60)
  return `${base}_${label || target?.from || 'unknown'}`
}

function findTreeNodeById(tree, targetId = '') {
  const id = String(targetId || '').trim()
  if (!id || !tree?.nodes?.length) return null
  return tree.nodes.find((node) => String(node.id || '') === id) || null
}

function nodeRange(node, docSize = 0) {
  const rawFrom = node?.position?.from ?? node?.from
  const rawTo = node?.position?.to ?? node?.to
  if (!Number.isFinite(Number(rawFrom))) return null
  const size = Math.max(0, Number(docSize) || 0)
  const from = Math.max(0, Math.min(Number(rawFrom), size || Number(rawFrom)))
  const to = Number.isFinite(Number(rawTo))
    ? Math.max(from, Math.min(Number(rawTo), size || Number(rawTo)))
    : Math.max(from, size)
  return to > from ? { from, to } : null
}

function targetFromTreeNode(editor, node, targetAnalysis = {}) {
  const doc = editor?.state?.doc
  const range = nodeRange(node, doc?.content?.size || 0)
  if (!doc || !range) return null
  const rawType = String(node.type || node.target_type || targetAnalysis?.target_type || '').toLowerCase()
  const resolvedType = rawType === 'full_document' ? 'full_document'
    : rawType === 'table' ? 'table'
      : rawType === 'paragraph' ? 'paragraph'
        : rawType === 'selection' ? 'selection'
          : 'section'
  const text = doc.textBetween(range.from, range.to, '\n')?.trim() || node.content || ''
  return {
    from: range.from,
    to: range.to,
    text,
    isFullDoc: resolvedType === 'full_document',
    sectionName: node.title || node.caption || node.label || targetAnalysis?.target_label || 'Selected target',
    sectionType: resolvedType === 'table' ? 'Table'
      : resolvedType === 'paragraph' ? 'Paragraph'
        : resolvedType === 'selection' ? 'Selection'
          : resolvedType === 'full_document' ? 'Full Document'
            : 'Heading',
    confidence: Math.max(Number(targetAnalysis?.confidence || 0), 0.72),
    resolved_target_type: resolvedType,
    target_id: node.id,
    match_reason: 'deep_agent_candidate_id_match',
  }
}

function targetIdTypeMismatch({ node, effectiveScope, intent, combinedPrompt }) {
  const nodeType = String(node?.type || node?.target_type || '').toLowerCase()
  if (!nodeType) return false
  if ((intent.names_table || isTableScope(effectiveScope) || /\b(?:table|tabelle)\b/i.test(combinedPrompt)) && nodeType !== 'table') {
    return true
  }
  if (
    effectiveScope === 'section'
    && !['section', 'table_section'].includes(nodeType)
    && !/\b(?:paragraph|sentence|selected|selection|highlighted)\b/i.test(combinedPrompt)
  ) {
    return true
  }
  return false
}

function containsRealTable(tree, target) {
  if (!target) return false
  return tree.tables.some((table) => rangesOverlap(target, table.position))
}

function validateTarget({ target, resolution, intent, userQuery, explicitSelection }) {
  if (!target?.text || target.from == null || target.to == null) {
    return {
      ok: false,
      reason: 'Could not find that heading, table, or paragraph in the open SOP.',
      code: 'target_not_found',
    }
  }

  const structural =
    STRUCTURAL_SCOPES.has(resolution.scope)
    || promptNamesStructuralTarget(userQuery)
    || ['section', 'table', 'table_section'].includes(String(resolution.target_type || '').toLowerCase())

  if (structural && !explicitSelection && target.sectionType === 'Word') {
    return {
      ok: false,
      reason: 'The request names a section or table, but the editor resolved only one word.',
      code: 'unsafe_word_target',
    }
  }

  const label = String(resolution.target_label || target.sectionName || '').trim()
  if (
    structural
    && !explicitSelection
    && resolution.scope !== 'table'
    && label
    && String(target.text || '').trim().length <= label.length + 8
  ) {
    return {
      ok: false,
      reason: `The target "${label}" resolved only to its heading, not the full section body.`,
      code: 'heading_only_target',
    }
  }

  if (intent.names_table && resolution.scope === 'table' && target.sectionType !== 'Table') {
    return {
      ok: false,
      reason: 'The request names a table, but the editor did not resolve a real table node.',
      code: 'table_target_missing_table_node',
    }
  }

  return { ok: true }
}

function resolveScope({ intent, targetScope, targetType, targetAnalysis, combinedPrompt, sectionHint = '', targetLabel = '' }) {
  const analysisType = String(targetType || targetAnalysis?.target_type || '').toLowerCase()
  const structuralPrompt = promptNamesStructuralTarget(combinedPrompt) && !explicitlyUsesSelection(combinedPrompt)
  const hasNamedHint = Boolean(String(targetLabel || targetAnalysis?.target_label || sectionHint || '').trim())
  const semanticType = String(targetAnalysis?.target_type || '').toLowerCase()

  if (intent.names_table) return 'table'
  if (structuralPrompt) return intent.names_table ? 'table' : 'section'
  if (['section', 'table', 'table_section', 'paragraph', 'sentence'].includes(semanticType)) return semanticType
  if (hasNamedHint && !intent.names_full_document && analysisType === 'full_document') return 'section'
  if (hasNamedHint && !intent.names_full_document && String(targetScope || '').toLowerCase() === 'full_document') return 'section'
  if (intent.names_full_document) return 'full_document'
  if (analysisType) return analysisType
  if (intent.target_scope && intent.target_scope !== 'unknown') return intent.target_scope
  if (targetScope) return targetScope
  return ''
}

function buildResolution({ target, tree, intent, scope, userQuery, targetAnalysis, reason }) {
  const containsTable = containsRealTable(tree, target)
  const confidence = Number(
    Math.max(
      Number(targetAnalysis?.confidence || 0),
      Number(target?.confidence || 0),
      target?.sectionType === 'Table' ? 0.94 : 0.88,
    ).toFixed(2),
  )

  return {
    intent: intent.intent,
    primary_action: intent.primary_action,
    scope: target?.isFullDoc ? 'full_document' : (scope || target?.resolved_target_type || 'section'),
    target_scope: target?.isFullDoc ? 'full_document' : (scope || target?.resolved_target_type || 'section'),
    target_ids: [targetIdFor(target)],
    range: { from: target.from, to: target.to },
    confidence,
    reason,
    needs_clarification: false,
    candidate_targets: [],
    contains_table: containsTable,
    patch_policy: 'replace_exact_range',
    resolved_heading: target.sectionName || null,
    target_type: target?.resolved_target_type || target?.sectionType || scope || null,
    target_id: targetIdFor(target),
    target_label: targetAnalysis?.target_label || target.sectionName || null,
    length_constraint: intent.length_constraint,
    style_constraint: intent.style_constraint,
    evidence_requirement: intent.evidence_requirement,
  }
}

export const TargetResolverAgent = {
  resolve({
    editor,
    userQuery = '',
    prompt = '',
    userPrompt = '',
    action = '',
    selection = null,
    sectionHint = '',
    targetScope = '',
    targetType = '',
    targetId = '',
    targetLabel = '',
    owningSection = '',
    lineNumber = null,
    recordId = '',
    preferFullSection = false,
    targetAnalysis = null,
  } = {}) {
    const effectiveQuery = String(userQuery || userPrompt || prompt || '').trim()
    const combinedPrompt = `${effectiveQuery}\n${prompt || ''}\n${userPrompt || ''}`.trim()
    const intent = classifyTargetIntent(combinedPrompt, { action, targetScope })
    const tree = parseSopDocument(editor)
    const explicitSelection = intent.explicit_selection
    const scope = resolveScope({ intent, targetScope, targetType, targetAnalysis, combinedPrompt, sectionHint, targetLabel })
    const structuralPrompt = promptNamesStructuralTarget(combinedPrompt) && !explicitSelection
    const effectiveScope = structuralPrompt && scope === 'selection'
      ? (intent.names_table ? 'table' : 'section')
      : scope
    const semanticTargetId = String(targetId || targetAnalysis?.target_id || '').trim()
    const selectionRequestsSection = /\b(?:selected|highlighted|marked)\s+(?:section|heading)\b/i.test(combinedPrompt)
    const candidateOptions = {
      tableOnly: isTableScope(effectiveScope) || intent.names_table,
      sectionOnly: effectiveScope === 'section' && !intent.names_table,
    }

    const resolverSelection =
      explicitSelection || (!structuralPrompt && effectiveScope === 'selection')
        ? selection
        : { empty: true }

    let target = null
    let resolverError = null
    if (intent.names_table && requestsUnsupportedTableSpan(combinedPrompt) && !explicitSelection) {
      const candidates = candidateList(editor, combinedPrompt, { tableOnly: true })
      return {
        target: null,
        tree,
        intent,
        resolution: {
          intent: intent.intent,
          primary_action: intent.primary_action,
          scope: 'table',
          target_scope: 'table',
          target_ids: [],
          confidence: 0.42,
          reason: 'Table row/cell targeting needs a concrete row or cell ID before editing.',
          needs_clarification: true,
          candidate_targets: candidates,
          patch_policy: 'clarify_before_edit',
          code: 'table_span_requires_specific_target',
        },
      }
    }
    if (semanticTargetId) {
      if (semanticTargetId === 'selection' && selection && !selection.empty && explicitSelection) {
        const resolvedSelection = resolveTargetInEditor(editor, {
          prompt: combinedPrompt,
          userPrompt: effectiveQuery,
          selection,
          targetScope: 'selection',
          targetType: 'selection',
          preferFullSection: selectionRequestsSection,
        })
        if (resolvedSelection) {
          target = {
            ...resolvedSelection,
            confidence: Math.max(Number(targetAnalysis?.confidence || 0), 0.92),
            target_id: 'selection',
            match_reason: selectionRequestsSection
              ? 'selected_section_expanded_from_live_heading'
              : 'deep_agent_selection_id_match',
          }
        }
      }
    }
    if (semanticTargetId && !target) {
      const semanticNode = findTreeNodeById(tree, semanticTargetId)
      if (!semanticNode) {
        const candidates = candidateList(editor, combinedPrompt, candidateOptions)
        return {
          target: null,
          tree,
          intent,
          resolution: {
            intent: intent.intent,
            primary_action: intent.primary_action,
            scope: effectiveScope || intent.target_scope || 'unknown',
            target_scope: effectiveScope || intent.target_scope || 'unknown',
            target_ids: [],
            confidence: Math.min(Number(targetAnalysis?.confidence || 0), 0.35),
            reason: `DeepAgent returned target_id "${semanticTargetId}", but it is not present in the live editor snapshot.`,
            needs_clarification: true,
            candidate_targets: candidates,
            patch_policy: 'clarify_before_edit',
            code: 'invalid_deep_agent_target_id',
          },
        }
      }
      if (targetIdTypeMismatch({ node: semanticNode, effectiveScope, intent, combinedPrompt })) {
        const candidates = candidateList(editor, combinedPrompt, candidateOptions)
        return {
          target: null,
          tree,
          intent,
          resolution: {
            intent: intent.intent,
            primary_action: intent.primary_action,
            scope: effectiveScope || intent.target_scope || 'unknown',
            target_scope: effectiveScope || intent.target_scope || 'unknown',
            target_ids: [],
            confidence: Math.min(Number(targetAnalysis?.confidence || 0), 0.38),
            reason: `DeepAgent target_id "${semanticTargetId}" has type "${semanticNode.type}", which does not match this request.`,
            needs_clarification: true,
            candidate_targets: candidates,
            patch_policy: 'clarify_before_edit',
            code: 'deep_agent_target_type_mismatch',
          },
        }
      }
      target = targetFromTreeNode(editor, semanticNode, targetAnalysis)
    }
    if (!target && isTableScope(effectiveScope)) {
      target = resolveTableTargetFromIndex(editor, combinedPrompt)
    }
    try {
      target = target || resolveTargetInEditor(editor, {
        prompt: String(prompt || effectiveQuery || ''),
        userPrompt: String(userPrompt || effectiveQuery || ''),
        selection: resolverSelection,
        sectionHint: String(targetLabel || targetAnalysis?.target_label || sectionHint || ''),
        targetScope: String(effectiveScope || ''),
        targetType: String(effectiveScope || targetType || ''),
        targetLabel: String(targetLabel || targetAnalysis?.target_label || ''),
        owningSection: String(owningSection || targetAnalysis?.owning_section || ''),
        lineNumber,
        recordId: String(recordId || ''),
        preferFullSection: Boolean(preferFullSection || selectionRequestsSection || structuralPrompt || effectiveScope === 'section' || effectiveScope === 'table_section'),
      })
    } catch (err) {
      resolverError = err
    }

    if (target?.sectionType === 'Word' && structuralPrompt) {
      try {
        const retried = resolveTargetInEditor(editor, {
          prompt: String(prompt || effectiveQuery || ''),
          userPrompt: String(userPrompt || effectiveQuery || ''),
          selection: { empty: true },
          sectionHint: String(targetLabel || targetAnalysis?.target_label || sectionHint || target.text || ''),
          targetScope: intent.names_table ? 'table' : 'section',
          targetType: intent.names_table ? 'table' : 'section',
          targetLabel: String(targetLabel || targetAnalysis?.target_label || sectionHint || target.text || ''),
          owningSection: String(owningSection || targetAnalysis?.owning_section || ''),
          lineNumber,
          recordId: String(recordId || ''),
          preferFullSection: true,
        })
        if (retried) target = retried
      } catch (err) {
        resolverError = err
      }
    }

    const initialReason = explicitSelection && selection && !selection.empty
      ? 'Explicit editor selection used.'
      : structuralPrompt
        ? 'Named structural target resolved; stale selection ignored.'
        : 'Target resolved from editor structure.'

    if (!target) {
      const candidates = candidateList(editor, combinedPrompt, candidateOptions)
      return {
        target: null,
        tree,
        intent,
        resolution: {
          intent: intent.intent,
          primary_action: intent.primary_action,
          scope: effectiveScope || intent.target_scope || 'unknown',
          target_scope: effectiveScope || intent.target_scope || 'unknown',
          target_ids: [],
          confidence: Number(resolverError?.confidence || 0),
          reason: resolverError?.message || 'No deterministic target matched.',
          needs_clarification: true,
          candidate_targets: candidates,
          patch_policy: 'clarify_before_edit',
        },
      }
    }

    const resolution = buildResolution({
      target,
      tree,
      intent,
      scope: effectiveScope,
      userQuery: combinedPrompt,
      targetAnalysis,
      reason: initialReason,
    })

    const validation = validateTarget({
      target,
      resolution,
      intent,
      userQuery: combinedPrompt,
      explicitSelection,
    })

    if (!validation.ok) {
      const candidates = candidateList(editor, combinedPrompt, candidateOptions)
      return {
        target,
        tree,
        intent,
        resolution: {
          ...resolution,
          confidence: Math.min(resolution.confidence, 0.45),
          reason: validation.reason,
          needs_clarification: true,
          candidate_targets: candidates,
          patch_policy: 'clarify_before_edit',
          code: validation.code,
        },
      }
    }

    return {
      target: {
        ...target,
        audit: {
          target_type: resolution.target_type,
          target_id: resolution.target_id,
          target_label: resolution.target_label,
          owning_section: targetAnalysis?.owning_section || owningSection || null,
          confidence: resolution.confidence,
          source: 'TargetResolverAgent',
          match_reason: target.match_reason || resolution.reason,
          selection_used: explicitSelection && Boolean(selection && !selection.empty),
        },
      },
      tree,
      intent,
      resolution,
    }
  },
}

export function resolveTargetWithAgent(options = {}) {
  return TargetResolverAgent.resolve(options)
}

export default TargetResolverAgent
