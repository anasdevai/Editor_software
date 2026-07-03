import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import AIComparisonModal from './AIComparisonModal'
import { analyzeSopTarget, performAIAction } from '../../api/editorApi'
import { formatAiSuggestionForUi } from '../../utils/aiOutputFormatter'
import {
  dispatchEditorSnapshotResponse,
  subscribeEditorInlineSuggestionApply,
  subscribeEditorInlineSuggestionClear,
  subscribeEditorInlineSuggestionShow,
  subscribeEditorSnapshotRequest,
  EDITOR_GAP_APPEND_EVENT,
  EDITOR_SCROLL_TO_RANGE_EVENT,
  EDITOR_SELECTION_QUERY_EVENT,
  EDITOR_SELECTION_RESPONSE_EVENT,
} from '../../utils/editorActionsBridge'
import {
  clearInlineAiSuggestion,
  setInlineAiSuggestion,
} from '../../utils/editorInlineSuggestionPlugin'
import { buildEditorSectionIndex, buildEditorTableIndex, collectSectionRanges } from '../../utils/editorTargetResolver'
import { TargetResolverAgent } from '../../utils/targeting/targetResolverAgent'
import { parseSopDocument } from '../../utils/targeting/sopParser'
import { buildDeepAgentTargetContext } from '../../utils/targeting/contextBuilder'
import { htmlContainsTable, resolveTableReplacementRange } from '../../utils/tableReplacementRange'
import {
  AI_ACTION_TRIGGERED_BY,
  EDITOR_AI_ACTIONS,
  EDITOR_AI_ACTION_STATUS,
  dispatchEditorAiActionResult,
  subscribeEditorAiActionRequest,
} from '../../utils/editorAiBridge'

const INLINE_SHOWN_EVENT = 'editor-actions-inline-shown'
const INLINE_APPLIED_EVENT = 'editor-actions-inline-applied'

const ACTION_TEXT_WARNING_CHARS = 7000

const stripHtml = (value) =>
  String(value || '')
    .replace(/<br\s*\/?>/gi, '\n')
    .replace(/<\/p>/gi, '\n')
    .replace(/<\/div>/gi, '\n')
    .replace(/<[^>]+>/g, '')
    .replace(/\n{3,}/g, '\n\n')
    .trim()

const escapeMarkdownCell = (value) =>
  String(value || '')
    .replace(/\s+/g, ' ')
    .replace(/\|/g, '\\|')
    .trim()

const escapeHtmlCell = (value) =>
  String(value || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')

const tableNodeToMarkdown = (tableNode) => {
  const rows = []
  tableNode?.forEach?.((rowNode) => {
    const cells = []
    rowNode?.forEach?.((cellNode) => {
      cells.push(escapeMarkdownCell(cellNode.textContent || ' '))
    })
    if (cells.length) rows.push(cells)
  })
  if (!rows.length) return ''

  const width = Math.max(...rows.map((row) => row.length))
  const normalizedRows = rows.map((row) => Array.from({ length: width }, (_, index) => row[index] || ' '))
  const header = normalizedRows[0]
  const separator = Array.from({ length: width }, () => '---')
  const body = normalizedRows.slice(1)
  return [header, separator, ...body]
    .map((row) => `| ${row.join(' | ')} |`)
    .join('\n')
}

const extractTableContext = (editor, from, to) => {
  const tables = []
  try {
    editor?.state?.doc?.nodesBetween?.(from, to, (node) => {
      if (node.type?.name === 'table') {
        const markdown = tableNodeToMarkdown(node)
        if (markdown) tables.push(markdown)
        return false
      }
      return true
    })
  } catch {
    return []
  }
  return tables
}

const markdownTablesToHtml = (text = '') => {
  const lines = String(text || '').replace(/\r\n/g, '\n').split('\n')
  const html = []
  let index = 0

  const isTableLine = (line) => /^\s*\|.+\|\s*$/.test(line)
  const splitRow = (line) => line.trim().replace(/^\||\|$/g, '').split('|').map((cell) => cell.trim())
  const isSeparator = (line) => splitRow(line).every((cell) => /^:?-{3,}:?$/.test(cell))

  while (index < lines.length) {
    if (isTableLine(lines[index]) && isTableLine(lines[index + 1] || '') && isSeparator(lines[index + 1])) {
      const header = splitRow(lines[index])
      index += 2
      const rows = []
      while (index < lines.length && isTableLine(lines[index])) {
        rows.push(splitRow(lines[index]))
        index += 1
      }
      html.push(
        `<table><thead><tr>${header.map((cell) => `<th>${escapeHtmlCell(cell)}</th>`).join('')}</tr></thead><tbody>${
          rows.map((row) => `<tr>${row.map((cell) => `<td>${escapeHtmlCell(cell)}</td>`).join('')}</tr>`).join('')
        }</tbody></table>`,
      )
      continue
    }
    html.push(lines[index])
    index += 1
  }

  return html.join('\n')
}

const firstMarkdownRowCells = (markdown = '') => {
  const line = String(markdown || '')
    .split(/\r?\n/)
    .find((row) => /^\s*\|.+\|\s*$/.test(row) && !/^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(row))
  if (!line) return []
  return line
    .trim()
    .replace(/^\||\|$/g, '')
    .split('|')
    .map((cell) => cell.replace(/\\\|/g, '|').trim())
    .filter(Boolean)
}

const markdownRows = (markdown = '') =>
  String(markdown || '')
    .split(/\r?\n/)
    .filter((row) => /^\s*\|.+\|\s*$/.test(row))
    .filter((row) => !/^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(row))
    .map((row) => row
      .trim()
      .replace(/^\||\|$/g, '')
      .split('|')
      .map((cell) => cell.replace(/\\\|/g, '|').trim()))

const inferTableWidthFromContext = (tableContext = []) => {
  const first = Array.isArray(tableContext) ? tableContext.find(Boolean) : ''
  const cells = firstMarkdownRowCells(first)
  if (cells.length >= 2) return Math.min(cells.length, 8)
  return 2
}

const tableHtmlFromRows = (rows = []) => {
  if (!Array.isArray(rows) || rows.length < 2) return ''
  const width = Math.max(...rows.map((row) => row.length))
  if (width < 2) return ''
  const normalize = (row) => Array.from({ length: width }, (_, index) => row[index] || '')
  const header = normalize(rows[0])
  const body = rows.slice(1).map(normalize)
  return `<table><thead><tr>${header.map((cell) => `<th>${escapeHtmlCell(cell)}</th>`).join('')}</tr></thead><tbody>${
    body.map((row) => `<tr>${row.map((cell) => `<td>${escapeHtmlCell(cell)}</td>`).join('')}</tr>`).join('')
  }</tbody></table>`
}

const normalizeLooseTableTextLines = (text = '') =>
  stripHtml(text)
    .replace(/\*\*(.*?)\*\*/g, '$1')
    .replace(/__(.*?)__/g, '$1')
    .replace(/^\s*[-*]\s+/gm, '')
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)

const anchoredPlainTableToHtml = (text = '', tableContext = []) => {
  const sourceRows = markdownRows(Array.isArray(tableContext) ? tableContext[0] : '')
  if (sourceRows.length < 2) return ''
  const header = sourceRows[0]
  const width = header.length
  if (width < 2) return ''

  const lines = normalizeLooseTableTextLines(text)
  if (!lines.length) return ''
  const normalizedHeader = header.map((cell) => cell.replace(/\s+/g, ' ').trim()).filter(Boolean)
  const headerLikeIndex = lines.findIndex((line) => (
    normalizedHeader.filter((cell) => line.toLowerCase().includes(cell.toLowerCase())).length >= Math.max(2, Math.ceil(width / 2))
  ))
  const bodyLines = headerLikeIndex >= 0 ? lines.slice(headerLikeIndex + 1) : lines
  if (!bodyLines.length) return ''

  const originalRowStarts = sourceRows
    .slice(1)
    .map((row) => String(row[0] || '').trim())
    .filter(Boolean)
  const rowStartRe = originalRowStarts.length
    ? new RegExp(`^(?:${originalRowStarts.map((value) => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})(?:\\b|\\s|$)`, 'i')
    : null

  const groups = []
  let current = []
  for (const line of bodyLines) {
    if (rowStartRe?.test(line) && current.length) {
      groups.push(current.join(' '))
      current = [line]
    } else {
      current.push(line)
    }
  }
  if (current.length) groups.push(current.join(' '))

  const rows = groups
    .map((group, index) => {
      const original = sourceRows[index + 1] || []
      const firstCell = original[0] || ''
      const rowText = firstCell && group.toLowerCase().startsWith(firstCell.toLowerCase())
        ? group.slice(firstCell.length).trim()
        : group
      if (width === 2) return [firstCell || rowText, firstCell ? rowText : (original[1] || '')]
      const originalMiddle = original.slice(1, width - 1)
      const middlePattern = originalMiddle
        .filter(Boolean)
        .map((value) => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
        .join('|')
      const parts = middlePattern
        ? rowText.split(new RegExp(`\\s+(${middlePattern})\\s+`, 'i')).filter(Boolean)
        : [rowText]
      const cells = [firstCell || original[0] || '']
      let rest = rowText
      for (const marker of originalMiddle) {
        const idx = rest.toLowerCase().indexOf(String(marker).toLowerCase())
        if (idx >= 0) {
          cells.push(rest.slice(0, idx).trim() || marker)
          rest = rest.slice(idx + String(marker).length).trim()
        } else {
          cells.push(marker || '')
        }
      }
      cells.push(rest || original[width - 1] || parts.join(' '))
      return cells.slice(0, width)
    })
    .filter((row) => row.some((cell) => String(cell || '').trim()))

  if (rows.length < 1) return ''
  return tableHtmlFromRows([header, ...rows])
}

const plainLineTableToHtml = (text = '', tableContext = []) => {
  const raw = String(text || '')
  if (/<\/?(?:table|thead|tbody|tr|td|th)\b/i.test(raw)) return raw

  const anchored = anchoredPlainTableToHtml(raw, tableContext)
  if (anchored) return anchored

  const lines = normalizeLooseTableTextLines(raw)

  const width = inferTableWidthFromContext(tableContext)
  if (width < 2 || lines.length < width * 2) return ''

  const rows = []
  for (let index = 0; index < lines.length; index += width) {
    const row = lines.slice(index, index + width)
    if (row.length < width) {
      if (!rows.length) break
      rows[rows.length - 1][width - 1] = `${rows[rows.length - 1][width - 1]} ${row.join(' ')}`.trim()
      break
    }
    rows.push(row)
  }

  if (rows.length < 2) return ''
  const header = rows[0]
  const body = rows.slice(1)
  return tableHtmlFromRows([header, ...body])
}

const normalizeTableSuggestionHtml = (text = '', tableContext = []) => {
  const raw = String(text || '')
  if (!raw.trim()) return ''
  if (/<\/?(?:table|thead|tbody|tr|td|th)\b/i.test(raw)) return raw

  const markdownHtml = markdownTablesToHtml(raw)
  if (/<\/?(?:table|thead|tbody|tr|td|th)\b/i.test(markdownHtml)) return markdownHtml

  const formatted = formatAiSuggestionForUi({ action: EDITOR_AI_ACTIONS.REWRITE, suggestedText: raw, structuredData: {} })
  if (/<\/?(?:table|thead|tbody|tr|td|th)\b/i.test(formatted)) return formatted

  return plainLineTableToHtml(raw, tableContext) || formatted
}

const compactTextForTargetMap = (value, limit = 700) =>
  String(value || '')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, limit)

const wantsFullDocumentTarget = (value = '') =>
  (
    /\b(?:full|whole|entire|complete)\s+(?:sop|document|doc)\b|\b(?:rewrite|improve|summarize|gap\s*check)\s+(?:the\s+|this\s+|current\s+)?(?:full|whole|entire|complete)?\s*(?:sop|document|doc)\b/i
      .test(String(value || ''))
    && !namesStructuralTarget(value)
  )

const compactSectionIndexForTargetAnalysis = (sections = []) =>
  (Array.isArray(sections) ? sections : []).map((section, index) => ({
    id: section.id || `section-${index + 1}`,
    type: 'section',
    target_type: 'section',
    index,
    sectionName: section.sectionName || section.title || section.heading || '',
    title: section.sectionName || section.title || section.heading || '',
    label: section.sectionName || section.title || section.heading || '',
    sectionType: section.sectionType || 'Heading',
    from: section.from,
    to: section.to,
    text: compactTextForTargetMap(section.text, 900),
  }))

const compactTableIndexForTargetAnalysis = (tables = []) =>
  (Array.isArray(tables) ? tables : []).map((table, index) => ({
    id: table.id || `table-${index + 1}`,
    type: 'table',
    target_type: 'table',
    index,
    caption: table.caption || `Table ${index + 1}`,
    label: table.caption || table.owningSection || `Table ${index + 1}`,
    owningSection: table.owningSection || '',
    from: table.from,
    to: table.to,
    rowCount: table.rowCount || 0,
    columnCount: table.columnCount || 0,
    text: compactTextForTargetMap(table.text, 900),
  }))

const compactParagraphIndexForTargetAnalysis = (paragraphs = []) =>
  (Array.isArray(paragraphs) ? paragraphs : []).map((paragraph, index) => ({
    id: paragraph.id || `paragraph-${index + 1}`,
    type: 'paragraph',
    target_type: 'paragraph',
    index,
    label: paragraph.title || `Paragraph ${index + 1}`,
    title: paragraph.title || `Paragraph ${index + 1}`,
    from: paragraph.position?.from ?? paragraph.from,
    to: paragraph.position?.to ?? paragraph.to,
    text: compactTextForTargetMap(paragraph.content || paragraph.text, 900),
  }))

const compactDocumentTreeForTargetAnalysis = (nodes = []) =>
  (Array.isArray(nodes) ? nodes : []).slice(0, 220).map((node, index) => ({
    id: node.id || `node-${index + 1}`,
    type: node.type || 'node',
    target_type: node.type || 'node',
    index,
    label: node.title || node.caption || `Node ${index + 1}`,
    title: node.title || node.caption || `Node ${index + 1}`,
    caption: node.caption || '',
    owningSection: node.owningSection || node.owning_section || '',
    parent_id: node.parent_id || null,
    parent_title: node.parent_title || '',
    from: node.position?.from ?? node.from,
    to: node.position?.to ?? node.to,
    rowCount: node.rowCount || 0,
    columnCount: node.columnCount || 0,
    text: compactTextForTargetMap(node.content || node.text, 700),
  }))

const buildActiveScopeForTargetAnalysis = (parsedSop, selectionPayload = {}) => {
  const anchor = Number(selectionPayload?.from ?? selectionPayload?.to)
  if (!Number.isFinite(anchor)) return {}
  const nodes = Array.isArray(parsedSop?.nodes) ? parsedSop.nodes : []
  const sections = Array.isArray(parsedSop?.sections) ? parsedSop.sections : []
  const containing = nodes
    .filter((node) => {
      const from = Number(node.position?.from ?? node.from)
      const to = Number(node.position?.to ?? node.to)
      return Number.isFinite(from) && Number.isFinite(to) && from <= anchor && anchor <= to
    })
    .sort((a, b) => {
      const aSize = Number(a.position?.to ?? a.to) - Number(a.position?.from ?? a.from)
      const bSize = Number(b.position?.to ?? b.to) - Number(b.position?.from ?? b.from)
      return aSize - bSize
    })
  const block = containing.find((node) => node.id !== 'doc_root') || null
  const section = sections
    .filter((node) => {
      const from = Number(node.position?.from ?? node.from)
      const to = Number(node.position?.to ?? node.to)
      return Number.isFinite(from) && Number.isFinite(to) && from <= anchor && anchor <= to
    })
    .sort((a, b) => Number(b.position?.from ?? b.from) - Number(a.position?.from ?? a.from))[0]
  return {
    cursor_block_id: block?.id || null,
    cursor_block_type: block?.type || null,
    cursor_block_label: block?.title || block?.caption || null,
    section_id: section?.id || block?.parent_id || null,
    sectionName: section?.title || block?.parent_title || null,
    from: block?.position?.from ?? null,
    to: block?.position?.to ?? null,
  }
}

const findParsedTargetNodeById = (editor, targetId = '') => {
  const id = String(targetId || '').trim()
  if (!id || id === 'selection' || id === 'doc_root') return null
  const parsed = parseSopDocument(editor)
  return parsed.nodes.find((node) => String(node.id || '') === id) || null
}

const buildTargetAnalysisPayload = ({
  editor,
  prompt,
  action,
  selectionPayload,
  sopMetadata,
}) => {
  const doc = editor.state.doc
  const parsedSop = parseSopDocument(editor)
  const activeScope = buildActiveScopeForTargetAnalysis(parsedSop, selectionPayload)
  const sectionIndex = parsedSop.sections?.length ? parsedSop.sections.map((section) => ({
    id: section.id,
    type: 'section',
    target_type: 'section',
    sectionName: section.title,
    title: section.title,
    sectionType: section.sectionType || 'Heading',
    from: section.position?.from,
    to: section.position?.to,
    text: section.content,
  })) : collectSectionRanges(doc)
  const tableIndex = parsedSop.tables?.length ? parsedSop.tables.map((table) => ({
    id: table.id,
    type: 'table',
    target_type: 'table',
    caption: table.caption || table.title,
    label: table.caption || table.title,
    owningSection: table.owningSection || '',
    from: table.position?.from,
    to: table.position?.to,
    rowCount: table.rowCount,
    columnCount: table.columnCount,
    text: table.content,
  })) : buildEditorTableIndex(editor)
  const fullText = doc.textBetween(0, doc.content.size, '\n')
  const fullTarget = wantsFullDocumentTarget(prompt)
  return {
    user_query: String(prompt || ''),
    action: String(action || ''),
    sop_metadata: sopMetadata || {},
    full_text: fullTarget ? fullText.slice(0, 60000) : '',
    document_excerpt: fullText.slice(0, 6000),
    document_schema: parsedSop.schema || '',
    section_index: compactSectionIndexForTargetAnalysis(sectionIndex),
    table_index: compactTableIndexForTargetAnalysis(tableIndex),
    paragraph_index: compactParagraphIndexForTargetAnalysis(parsedSop.paragraphs || []),
    document_tree: compactDocumentTreeForTargetAnalysis(parsedSop.nodes || []),
    selection: selectionPayload || { empty: true },
    active_scope: activeScope,
    cursor_block_id: activeScope.cursor_block_id || null,
  }
}

const assertUsableTargetAnalysis = (analysis) => {
  if (!analysis || typeof analysis !== 'object') {
    const err = new Error('The backend did not return a usable live-editor target decision.')
    err.code = 'target_analysis_missing'
    throw err
  }
  const candidates = Array.isArray(analysis.candidate_targets) ? analysis.candidate_targets : []
  if (analysis.requires_clarification || !String(analysis.target_id || '').trim()) {
    const labels = candidates.map((candidate) => candidate?.label).filter(Boolean).slice(0, 5)
    const err = new Error(
      labels.length
        ? `The target is ambiguous. Use one exact live heading: ${labels.join(', ')}.`
        : 'The target could not be tied to an exact live editor ID. Use the exact heading or select the text.',
    )
    err.code = 'target_analysis_requires_clarification'
    err.candidates = candidates
    throw err
  }
}

const normalizeTargetAnalysisForResolver = (analysis) => {
  if (!analysis || typeof analysis !== 'object') return analysis
  return analysis
}

const namesStructuralTarget = (value = '') =>
  /\b(?:section|sections|heading|table|tabelle|abschnitt|kapitel)\b/i.test(String(value || ''))
  || /["'][^"']{2,120}["']/.test(String(value || ''))
  || /\b\d+(?:\.\d+)+\b/.test(String(value || ''))

const namesTableTarget = (value = '') =>
  /\b(?:table|tabelle|tabular|matrix)\b/i.test(String(value || ''))

const targetAnalysisMatchesPromptShape = (analysis, promptText = '') => {
  if (!analysis || typeof analysis !== 'object') return false
  if (analysis.requires_clarification) return false
  const targetType = String(analysis.target_type || '').toLowerCase()
  if (namesTableTarget(promptText)) return ['table', 'table_section'].includes(targetType)
  if (namesStructuralTarget(promptText)) return ['section', 'table', 'table_section'].includes(targetType)
  return true
}

const explicitlyUsesSelection = (value = '') =>
  /\b(?:selected\s+(?:text|word|paragraph|section|line|content)|selection|highlighted|marked\s+text|this\s+selection)\b/i
    .test(String(value || ''))

const validateResolvedEditorTarget = ({
  target,
  targetAnalysis,
  prompt,
  userPrompt,
  editor,
}) => {
  if (!target?.text || target.from == null || target.to == null) {
    const err = new Error('Could not find that heading, table, or paragraph in the open SOP.')
    err.code = 'target_not_found'
    throw err
  }

  const combinedPrompt = `${userPrompt || ''}\n${prompt || ''}`
  const targetType = String(targetAnalysis?.target_type || target.resolved_target_type || '').toLowerCase()
  const label = String(targetAnalysis?.target_label || target.sectionName || '').trim()
  const text = String(target.text || '').trim()
  const plainLabel = stripHtml(label).trim()
  const isStructural =
    ['section', 'table', 'table_section'].includes(targetType)
    || namesStructuralTarget(combinedPrompt)
  const isExplicitSelection = explicitlyUsesSelection(combinedPrompt)

  if (isStructural && !isExplicitSelection && target.sectionType === 'Word') {
    const err = new Error('The request names a section or table, but the editor resolved only one word. Use the exact heading/table name or clear the stale selection.')
    err.code = 'unsafe_word_target'
    throw err
  }

  if (
    isStructural
    && !isExplicitSelection
    && plainLabel
    && text.length <= plainLabel.length + 8
    && targetType !== 'table'
  ) {
    const err = new Error(`The target "${plainLabel}" resolved only to its heading. I will not edit until the full section body is resolved.`)
    err.code = 'heading_only_target'
    throw err
  }

  if (targetType === 'table' || targetType === 'table_section' || /\btable|tabelle\b/i.test(combinedPrompt)) {
    const tableContext = extractTableContext(editor, target.from, target.to)
    if (!tableContext.length && targetType === 'table') {
      const err = new Error(`The target "${plainLabel || 'table'}" did not resolve to a real editor table. I will not edit the wrong region.`)
      err.code = 'table_target_missing_table_node'
      throw err
    }
  }

  return {
    ...target,
    audit: {
      target_type: targetType || target.resolved_target_type || target.sectionType || 'unknown',
      target_id: targetAnalysis?.target_id || target.audit?.target_id || null,
      target_label: targetAnalysis?.target_label || target.sectionName || null,
      owning_section: targetAnalysis?.owning_section || null,
      confidence: Number(target.audit?.confidence ?? targetAnalysis?.confidence ?? target.confidence ?? 0),
      source: target.audit?.source || targetAnalysis?.source || target.match_reason || 'TargetResolverAgent',
      match_reason: target.audit?.match_reason || target.match_reason || null,
      selection_used: Boolean(target.audit?.selection_used)
        || target.resolved_target_type === 'selection'
        || target.sectionName === 'Selected text',
    },
  }
}

const resolveValidatedEditorTarget = async ({
  editor,
  action,
  prompt,
  userPrompt = '',
  selectionPayload,
  sopMetadata,
  sectionHint = '',
  targetScope = '',
  targetId = '',
  targetType = '',
  targetLabel = '',
  owningSection = '',
  lineNumber = null,
  recordId = '',
  preferFullSection = false,
}) => {
  const suppliedTargetId = String(targetId || '').trim()
  const contractTargetId = suppliedTargetId && !['doc_root', 'selection'].includes(suppliedTargetId)
    ? suppliedTargetId
    : ''
  const suppliedLiveNode = contractTargetId ? findParsedTargetNodeById(editor, contractTargetId) : null
  const rawTargetAnalysis = contractTargetId
    ? {
        target_id: contractTargetId,
        target_type: String(targetType || suppliedLiveNode?.type || '').trim().toLowerCase(),
        target_label: String(targetLabel || suppliedLiveNode?.title || suppliedLiveNode?.caption || '').trim() || null,
        owning_section: String(owningSection || suppliedLiveNode?.parent_title || suppliedLiveNode?.owningSection || '').trim() || null,
        confidence: suppliedLiveNode ? 1 : 0,
        requires_clarification: !suppliedLiveNode,
        candidate_targets: [],
        reasoning_summary: suppliedLiveNode
          ? 'Exact target ID supplied by the refreshed live editor contract.'
          : 'The supplied target ID is not present in the current editor snapshot.',
        source: 'live_editor_contract',
      }
    : await analyzeSopTarget(buildTargetAnalysisPayload({
        editor,
        prompt: String(prompt || userPrompt || ''),
        action: String(action || ''),
        selectionPayload,
        sopMetadata: sopMetadata || {},
      }))
  const targetAnalysis = normalizeTargetAnalysisForResolver(rawTargetAnalysis)
  assertUsableTargetAnalysis(targetAnalysis)
  const combinedPrompt = `${userPrompt || ''}\n${prompt || ''}`
  const promptNamesStructuralTarget = namesStructuralTarget(combinedPrompt) && !explicitlyUsesSelection(combinedPrompt)
  const useSemanticTargetAnalysis = targetAnalysisMatchesPromptShape(targetAnalysis, combinedPrompt)
  const effectiveAnalysisType =
    promptNamesStructuralTarget
      ? (namesTableTarget(combinedPrompt) ? 'table' : 'section')
      : String(useSemanticTargetAnalysis ? targetAnalysis?.target_type || '' : '')
  const effectiveTargetLabel = useSemanticTargetAnalysis
    ? String(targetAnalysis?.target_label || '')
    : ''
  const effectiveTargetId = useSemanticTargetAnalysis
    ? String(targetAnalysis?.target_id || '')
    : ''
  const effectiveOwningSection = useSemanticTargetAnalysis
    ? String(targetAnalysis?.owning_section || '')
    : ''

  const agentResult = TargetResolverAgent.resolve({
    editor,
    action,
    userQuery: String(prompt || userPrompt || ''),
    prompt: String(prompt || ''),
    userPrompt: String(userPrompt || ''),
    selection: selectionPayload,
    sectionHint: String(effectiveTargetLabel || sectionHint || ''),
    targetScope: String(effectiveAnalysisType || targetScope || ''),
    targetType: String(effectiveAnalysisType || ''),
    targetId: effectiveTargetId,
    targetLabel: effectiveTargetLabel,
    owningSection: effectiveOwningSection,
    lineNumber,
    recordId: String(recordId || ''),
    preferFullSection: Boolean(preferFullSection),
    targetAnalysis: {
      ...targetAnalysis,
      target_type: effectiveAnalysisType || targetAnalysis?.target_type,
      target_id: effectiveTargetId || targetAnalysis?.target_id || null,
      target_label: effectiveTargetLabel || null,
      owning_section: effectiveOwningSection || null,
    },
  })

  if (agentResult?.resolution?.needs_clarification) {
    const choices = (agentResult.resolution.candidate_targets || [])
      .map((candidate) => candidate?.label || '')
      .filter(Boolean)
      .slice(0, 5)
      .join(', ')
    const err = new Error(
      choices
        ? `I found multiple possible targets: ${choices}. Please use the exact heading/table name.`
        : agentResult.resolution.reason || 'I could not confidently identify the target in the open SOP. Select it or use the exact heading/table name.',
    )
    err.code = agentResult.resolution.code || 'target_resolution_clarification'
    err.candidates = agentResult.resolution.candidate_targets || []
    err.confidence = agentResult.resolution.confidence ?? null
    throw err
  }

  const target = agentResult.target

  return {
    target: validateResolvedEditorTarget({
      target,
      targetAnalysis: {
        ...targetAnalysis,
        target_type: effectiveAnalysisType || targetAnalysis?.target_type,
        target_id: effectiveTargetId || targetAnalysis?.target_id || null,
        target_label: effectiveTargetLabel || null,
        owning_section: effectiveOwningSection || null,
      },
      prompt,
      userPrompt,
      editor,
    }),
    targetAnalysis,
    targetResolution: agentResult.resolution,
    targetIntent: agentResult.intent,
  }
}

const buildFindingsHtml = (structuredData) => {
  if (!structuredData || typeof structuredData !== 'object') return ''
  const gaps = Array.isArray(structuredData.gaps) ? structuredData.gaps : []
  const items = gaps.length > 0
    ? gaps
    : [{
      issue: structuredData.issue,
      explanation: structuredData.explanation,
      recommendation: structuredData.recommendation,
    }].filter((entry) => entry && (entry.issue || entry.explanation || entry.recommendation))

  if (items.length === 0) return ''

  const itemsHtml = items
    .map((gap) => {
      const issue = gap?.issue ? `<p><strong>Issue:</strong> ${gap.issue}</p>` : ''
      const explanation = gap?.explanation ? `<p><strong>Explanation:</strong> ${gap.explanation}</p>` : ''
      const recommendation = gap?.recommendation
        ? `<p><strong>Recommendation:</strong> ${gap.recommendation}</p>`
        : ''
      return `<li>${issue}${explanation}${recommendation}</li>`
    })
    .join('')

  return `<h3>AI Gap Check Findings</h3><ul>${itemsHtml}</ul>`
}

const ALLOWED_ACTIONS = new Set([
  EDITOR_AI_ACTIONS.REWRITE,
  EDITOR_AI_ACTIONS.IMPROVE,
  EDITOR_AI_ACTIONS.GAP_CHECK,
  EDITOR_AI_ACTIONS.SUMMARIZE,
  EDITOR_AI_ACTIONS.ANALYZE,
])

/**
 * Bridges KL/KI Assistant action requests into the live SOP editor.
 *
 * Subscribes to {@link EDITOR_AI_ACTION_REQUEST_EVENT} dispatched by chat
 * surfaces (AIWidget, ChatPage). For rewrite / improve / gap_check it runs
 * `/api/ai/action` on the current selection (or whole document when no
 * selection exists), shows the standard {@link AIComparisonModal}, then
 * applies the result into the editor on accept. A
 * {@link EDITOR_AI_ACTION_RESULT_EVENT} is emitted so the chat surface can
 * report status back to the user.
 *
 * The component renders nothing besides the modal portal.
 */
const EditorAIBridge = ({
  editor,
  documentId,
  sopMetadata,
  isEditable = true,
  onPreviewSessionChange,
  onAfterApply,
  onVersionCompareRequest,
}) => {
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [aiResult, setAIResult] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  /** Snapshot of the request that opened the current modal. */
  const activeRequestRef = useRef(null)
  /** Range in the editor that should receive the accepted content. */
  const targetRangeRef = useRef(null)
  /** Tracks whether we are currently using the full document as the source. */
  const isFullDocRef = useRef(false)
  /** True when the resolved rewrite/improve target contains real table nodes. */
  const tableTargetRef = useRef(false)
  const tableContextRef = useRef([])
  const inFlightRef = useRef(false)
  const editorRef = useRef(editor)
  const sopMetadataRef = useRef(sopMetadata)
  const documentIdRef = useRef(documentId)
  const isEditableRef = useRef(isEditable)
  /** Pending inline suggestion from the sidebar Actions tab. */
  const inlinePendingRef = useRef(null)

  useEffect(() => { editorRef.current = editor }, [editor])
  useEffect(() => { sopMetadataRef.current = sopMetadata }, [sopMetadata])
  useEffect(() => { documentIdRef.current = documentId }, [documentId])
  useEffect(() => { isEditableRef.current = isEditable }, [isEditable])

  const notifyPreviewSession = useCallback((active) => {
    if (typeof onPreviewSessionChange === 'function') {
      onPreviewSessionChange(active)
    }
  }, [onPreviewSessionChange])

  const emitResult = useCallback((detail) => {
    dispatchEditorAiActionResult(detail)
  }, [])

  const closeModal = useCallback(() => {
    setIsModalOpen(false)
    setAIResult(null)
    activeRequestRef.current = null
    targetRangeRef.current = null
    isFullDocRef.current = false
    tableTargetRef.current = false
    tableContextRef.current = []
    notifyPreviewSession(false)
  }, [notifyPreviewSession])

  const sopTitle = useMemo(() => {
    const metadata = sopMetadata || {}
    return (metadata.title || metadata.documentId || 'Untitled SOP').toString().trim() || 'Untitled SOP'
  }, [sopMetadata])

  const runActionRequest = useCallback(async (request) => {
    const { action, requestId } = request || {}
    if (action === EDITOR_AI_ACTIONS.COMPARE) {
      const liveEditor = editorRef.current
      if (!liveEditor || liveEditor.isDestroyed) {
        emitResult({
          ...request,
          status: EDITOR_AI_ACTION_STATUS.NOT_AVAILABLE,
          reason: 'editor_unavailable',
          message: 'Editor nicht bereit.',
        })
        console.warn('[kl-editor-action-failed]', { action, requestId, reason: 'editor_unavailable' })
        return
      }
      if (typeof onVersionCompareRequest !== 'function') {
        emitResult({
          ...request,
          status: EDITOR_AI_ACTION_STATUS.ERROR,
          message: 'Versionsvergleich ist hier nicht verfügbar.',
        })
        console.warn('[kl-editor-action-failed]', { action, requestId, reason: 'no_compare_handler' })
        return
      }
      try {
        console.info('[kl-editor-bridge-received]', { action, requestId, phase: 'compare' })
        await Promise.resolve(onVersionCompareRequest())
        emitResult({
          ...request,
          status: EDITOR_AI_ACTION_STATUS.DISPLAYED,
          action: EDITOR_AI_ACTIONS.COMPARE,
        })
      } catch (err) {
        console.error('[kl-editor-action-failed]', err)
        emitResult({
          ...request,
          status: EDITOR_AI_ACTION_STATUS.ERROR,
          message: err?.message || 'Versionsvergleich fehlgeschlagen.',
        })
      }
      return
    }

    if (!ALLOWED_ACTIONS.has(action)) {
      emitResult({
        ...request,
        status: EDITOR_AI_ACTION_STATUS.NOT_AVAILABLE,
        reason: 'unsupported_action',
      })
      return
    }
    const liveEditor = editorRef.current
    if (!liveEditor || liveEditor.isDestroyed || !isEditableRef.current) {
      emitResult({
        ...request,
        status: EDITOR_AI_ACTION_STATUS.NOT_AVAILABLE,
        reason: 'editor_unavailable',
      })
      return
    }

    if (inFlightRef.current) {
      emitResult({
        ...request,
        status: EDITOR_AI_ACTION_STATUS.ERROR,
        message: 'Es läuft bereits eine Editor-Aktion.',
      })
      return
    }

    const { state } = liveEditor
    const { selection } = state
    const hasSelection = Boolean(selection && !selection.empty)
    const selectionPayload = hasSelection
      ? {
          from: selection.from,
          to: selection.to,
          text: state.doc.textBetween(selection.from, selection.to, '\n'),
          empty: false,
        }
      : { empty: true }

    let from = 0
    let to = state.doc.content.size
    let selectedText = state.doc.textBetween(from, to, '\n').trim()
    let isFullDoc = true
    let sectionName = 'Full SOP'
    let sectionType = 'Full Document'

    const actionPrompt = String(request?.prompt || '').trim()
    let targetAnalysis = null
    let targetResolution = null
    let targetIntent = null
    let targetContext = null
    if (actionPrompt) {
      try {
        const resolvedTarget = await resolveValidatedEditorTarget({
          editor: liveEditor,
          action,
          prompt: actionPrompt,
          userPrompt: String(request?.userPrompt || ''),
          selectionPayload,
          sopMetadata: sopMetadataRef.current || {},
          sectionHint: String(request?.sectionHint || ''),
          targetScope: String(request?.targetScope || ''),
          targetId: String(request?.targetId || ''),
          targetType: String(request?.targetType || ''),
          targetLabel: String(request?.targetLabel || ''),
          owningSection: String(request?.owningSection || ''),
          lineNumber: request?.lineNumber ?? null,
          recordId: String(request?.recordId || ''),
          preferFullSection: Boolean(request?.preferFullSection),
        })
        targetAnalysis = resolvedTarget.targetAnalysis
        targetResolution = resolvedTarget.targetResolution
        targetIntent = resolvedTarget.targetIntent
        const resolved = resolvedTarget.target
        if (resolved?.text && resolved.from != null && resolved.to != null) {
          from = resolved.from
          to = resolved.to
          selectedText = resolved.text
          isFullDoc = Boolean(resolved.isFullDoc)
          sectionName = resolved.sectionName || sectionName
          sectionType = resolved.sectionType || sectionType
          targetContext = buildDeepAgentTargetContext({
            editor: liveEditor,
            target: resolved,
            resolution: targetResolution,
            intent: targetIntent,
            userQuery: actionPrompt,
          })
        }
      } catch (err) {
        emitResult({
          ...request,
          status: EDITOR_AI_ACTION_STATUS.ERROR,
          message: err?.message || 'Could not resolve target in the open SOP.',
        })
        return
      }
    } else if (hasSelection) {
      from = selection.from
      to = selection.to
      const fragment = state.doc.textBetween(from, to, '\n').trim()
      if (fragment.length > 0) {
        selectedText = fragment
        isFullDoc = false
        sectionName = 'Selected text'
        sectionType = 'Paragraph'
      }
    }

    if (!selectedText) {
      emitResult({
        ...request,
        status: EDITOR_AI_ACTION_STATUS.NOT_AVAILABLE,
        reason: 'empty_document',
      })
      return
    }

    if (selectedText.length > ACTION_TEXT_WARNING_CHARS) {
      const proceed = window.confirm(
        'Dieser SOP-Inhalt ist möglicherweise zu lang für das lokale LLM und kann mit Kontextlimit-Fehlern abbrechen.\n\nMit der Aktion fortfahren?',
      )
      if (!proceed) {
        emitResult({
          ...request,
          status: EDITOR_AI_ACTION_STATUS.CANCELLED,
          reason: 'user_declined_long_text',
        })
        return
      }
    }

    const tableContext = extractTableContext(liveEditor, from, to)
    const hasTableTarget = tableContext.length > 0
    const wantsTableOutput = /\b(?:table|tabular|tabelle|tabellenformat|table\s+format)\b/i.test(actionPrompt)
    const needsTableFormattedOutput = hasTableTarget || wantsTableOutput
    const resolvedTargetType = String(
      targetResolution?.target_type
      || targetAnalysis?.target_type
      || (hasTableTarget && sectionType === 'Table Section' ? 'table_section' : '')
      || (hasTableTarget && sectionType === 'Table' ? 'table' : ''),
    ).toLowerCase()
    const selectedTextForAction = needsTableFormattedOutput
      ? [
        selectedText,
        '',
        '[Resolved table/structured section - return table-style output when appropriate]',
        ...tableContext.map((table, index) => `Table ${index + 1}:\n${table}`),
      ].join('\n')
      : selectedText
    const tableOutputInstruction = resolvedTargetType === 'table_section'
      ? 'The resolved target is one complete SOP section containing a heading, optional explanatory paragraphs, and a real table. Return the complete section in the same order and preserve all three kinds of content. Use valid HTML such as <h2>/<h3>, <p>, and <table>; keep the table column count unchanged. Never return only the prose or only the table.'
      : 'The resolved target is a table. Return one valid HTML <table> with the same column count and rewritten cell values. Do not return table cells as separate bold lines. If HTML is not possible, return a Markdown table with a separator row. Do not convert requested table content into prose.'
    const actionInstruction = needsTableFormattedOutput
      ? [
        actionPrompt,
        tableOutputInstruction,
      ].filter(Boolean).join('\n\n')
      : actionPrompt

    inFlightRef.current = true
    activeRequestRef.current = request
    targetRangeRef.current = {
      from,
      to,
      target_id: targetResolution?.target_id || targetAnalysis?.target_id || null,
      target_type: resolvedTargetType || null,
      target_label: targetResolution?.target_label || targetAnalysis?.target_label || sectionName,
    }
    isFullDocRef.current = isFullDoc
    tableTargetRef.current = needsTableFormattedOutput
    tableContextRef.current = tableContext
    notifyPreviewSession(true)
    setIsLoading(true)

    try {
      console.info('[kl-editor-bridge-received]', {
        action,
        requestId,
        documentId: documentIdRef.current,
        textLen: selectedTextForAction.length,
        isFullDoc,
        hasTableTarget,
        wantsTableOutput,
        targetAnalysis,
        targetResolution,
        source: request?.source || 'unknown',
      })
      const result = await performAIAction({
        action,
        text: selectedTextForAction,
        document_id: documentIdRef.current || sopMetadataRef.current?.documentId || null,
        section_id: `kl-assistant-${requestId || Date.now()}`,
        sop_title: sopTitle,
        section_name: sectionName,
        section_type: needsTableFormattedOutput ? 'Table Section' : sectionType,
        edit_scope: isFullDoc ? 'full_document' : 'section_only',
        sop_entity_id: documentIdRef.current || null,
        triggered_by: AI_ACTION_TRIGGERED_BY.KL_ASSISTANT,
        instruction: actionInstruction || null,
        learn_to_profile: Boolean(request?.learn_to_profile),
        target_analysis: targetAnalysis,
        target_resolution: targetResolution,
        resolved_target: targetResolution,
        target_context: targetContext,
      })

      const tableAwareSuggestedText = needsTableFormattedOutput
        ? normalizeTableSuggestionHtml(result?.suggested_text, tableContext)
        : result?.suggested_text
      const tableAwareStructuredData = needsTableFormattedOutput && result?.structured_data
        ? {
          ...result.structured_data,
          rewritten_text: normalizeTableSuggestionHtml(result.structured_data.rewritten_text || '', tableContext),
          improved_text: normalizeTableSuggestionHtml(result.structured_data.improved_text || '', tableContext),
          improved_version: normalizeTableSuggestionHtml(result.structured_data.improved_version || '', tableContext),
        }
        : result?.structured_data

      const safeSuggestedHtml = formatAiSuggestionForUi({
        action: result?.action || action,
        suggestedText: tableAwareSuggestedText,
        structuredData: tableAwareStructuredData,
      })

      setAIResult({
        ...result,
        action: result?.action || action,
        suggested_text: safeSuggestedHtml,
        structured_data: tableAwareStructuredData,
        section_name: sectionName,
        contains_table_target: needsTableFormattedOutput,
      })
      setIsModalOpen(true)
      console.info('[kl-editor-action-modal-open]', { action: result?.action || action, requestId, isFullDoc })
    } catch (err) {
      console.error('[kl-editor-action-failed]', err)
      const message = err?.message || 'Editor-Aktion fehlgeschlagen.'
      emitResult({
        ...request,
        status: EDITOR_AI_ACTION_STATUS.ERROR,
        message,
      })
      notifyPreviewSession(false)
      activeRequestRef.current = null
      targetRangeRef.current = null
      isFullDocRef.current = false
      tableTargetRef.current = false
      tableContextRef.current = []
      window.alert(message)
    } finally {
      inFlightRef.current = false
      setIsLoading(false)
    }
  }, [emitResult, notifyPreviewSession, sopTitle, onVersionCompareRequest])

  const handleReadRequest = useCallback((request) => {
    const liveEditor = editorRef.current
    if (!liveEditor || liveEditor.isDestroyed) {
      emitResult({
        ...request,
        status: EDITOR_AI_ACTION_STATUS.NOT_AVAILABLE,
        reason: 'editor_unavailable',
      })
      return
    }
    const metadata = sopMetadataRef.current || {}
    const preview = (liveEditor.getText() || '').slice(0, 400)
    emitResult({
      ...request,
      status: EDITOR_AI_ACTION_STATUS.DISPLAYED,
      sop_id: documentIdRef.current || null,
      sop_title: metadata.title || '',
      sop_number: metadata.documentId || '',
      preview,
    })
  }, [emitResult])

  const resolveDocRange = useCallback((detail) => {
    const liveEditor = editorRef.current
    if (!liveEditor || liveEditor.isDestroyed) return null
    const size = liveEditor.state.doc.content.size
    const from = Number.isFinite(detail.from) ? detail.from : 0
    const to = Number.isFinite(detail.to) ? detail.to : size
    return {
      from: Math.max(0, Math.min(from, size)),
      to: Math.max(from, Math.min(to, size)),
    }
  }, [])

  const emitInlineShown = useCallback((requestId, range) => {
    const liveEditor = editorRef.current
    let toolbarCoords = null
    if (liveEditor?.view && range) {
      try {
        const coords = liveEditor.view.coordsAtPos(range.to)
        toolbarCoords = { top: coords.top + window.scrollY, left: coords.left + window.scrollX }
      } catch {
        toolbarCoords = null
      }
    }
    window.dispatchEvent(
      new CustomEvent(INLINE_SHOWN_EVENT, {
        detail: {
          requestId,
          toolbarCoords,
          from: range?.from,
          to: range?.to,
        },
      }),
    )
  }, [])

  const emitInlineApplied = useCallback((requestId, ok, message = '') => {
    window.dispatchEvent(
      new CustomEvent(INLINE_APPLIED_EVENT, {
        detail: { requestId, ok, message },
      }),
    )
  }, [])

  useEffect(() => {
    const onSelectionQuery = (event) => {
      const requestId = event.detail?.requestId
      const liveEditor = editorRef.current
      if (!requestId) return
      let hasSelection = false
      if (liveEditor && !liveEditor.isDestroyed && isEditableRef.current) {
        const sel = liveEditor.state.selection
        hasSelection = Boolean(sel && !sel.empty)
      }
      window.dispatchEvent(
        new CustomEvent(EDITOR_SELECTION_RESPONSE_EVENT, {
          detail: { requestId, hasSelection },
        }),
      )
    }
    window.addEventListener(EDITOR_SELECTION_QUERY_EVENT, onSelectionQuery)
    return () => window.removeEventListener(EDITOR_SELECTION_QUERY_EVENT, onSelectionQuery)
  }, [])

  useEffect(() => {
    const unsubSnapshot = subscribeEditorSnapshotRequest(async ({
      requestId,
      action,
      prompt,
      userPrompt,
      sectionHint,
      targetScope,
      targetId,
      targetType,
      targetLabel,
      owningSection,
      lineNumber,
      recordId,
      preferFullSection,
    }) => {
      const liveEditor = editorRef.current
      if (!liveEditor || liveEditor.isDestroyed || !isEditableRef.current) {
        dispatchEditorSnapshotResponse({
          requestId,
          ok: false,
          message: 'Editor is not available or is read-only.',
        })
        return
      }
      const { state } = liveEditor
      const { selection } = state
      const hasSelection = Boolean(selection && !selection.empty)
      const selectionPayload = hasSelection
        ? {
            from: selection.from,
            to: selection.to,
            text: state.doc.textBetween(selection.from, selection.to, '\n'),
            empty: false,
          }
        : { empty: true }

      try {
        const fullText = state.doc.textBetween(0, state.doc.content.size, '\n')
        const sectionIndex = collectSectionRanges(state.doc)
        const tableIndex = buildEditorTableIndex(liveEditor)
        const resolvedTarget = await resolveValidatedEditorTarget({
          editor: liveEditor,
          action: String(action || ''),
          prompt: String(prompt || userPrompt || ''),
          userPrompt: String(userPrompt || ''),
          selectionPayload,
          sopMetadata: sopMetadataRef.current || {},
          sectionHint: String(sectionHint || ''),
          targetScope: String(targetScope || ''),
          targetId: String(targetId || ''),
          targetType: String(targetType || ''),
          targetLabel: String(targetLabel || ''),
          owningSection: String(owningSection || ''),
          lineNumber: lineNumber ?? null,
          recordId: String(recordId || ''),
          preferFullSection: Boolean(preferFullSection),
        })
        const targetAnalysis = resolvedTarget.targetAnalysis
        const targetResolution = resolvedTarget.targetResolution
        const targetIntent = resolvedTarget.targetIntent
        const target = resolvedTarget.target
        if (!target?.text || target.from == null || target.to == null) {
          dispatchEditorSnapshotResponse({
            requestId,
            ok: false,
            error: 'Could not find that heading or paragraph in the open SOP. Check the text or select it in the editor.',
          })
          return
        }
        dispatchEditorSnapshotResponse({
          requestId,
          ok: true,
          target,
          sectionIndex,
          tableIndex,
          targetAnalysis,
          targetResolution,
          targetIntent,
          targetAudit: target.audit || null,
          targetContext: buildDeepAgentTargetContext({
            editor: liveEditor,
            target,
            resolution: targetResolution,
            intent: targetIntent,
            userQuery: String(prompt || userPrompt || ''),
          }),
          sopMetadata: sopMetadataRef.current || {},
          fullText,
          docSize: state.doc.content.size,
          selection: selectionPayload,
          sopTitle: (sopMetadataRef.current?.title || 'Untitled SOP').toString(),
          sopNumber: (sopMetadataRef.current?.documentId || '').toString(),
        })
      } catch (err) {
        dispatchEditorSnapshotResponse({
          requestId,
          ok: false,
          error: err?.message || 'Could not resolve target in editor.',
          code: err?.code || 'target_resolution_error',
          candidates: Array.isArray(err?.candidates)
            ? err.candidates
            : buildEditorSectionIndex(liveEditor).slice(0, 8),
          targetPhrase: err?.targetPhrase || '',
          confidence: err?.confidence ?? null,
        })
      }
    })

    const unsubShow = subscribeEditorInlineSuggestionShow((detail) => {
      const liveEditor = editorRef.current
      const requestId = detail?.requestId
      if (!requestId || !liveEditor || liveEditor.isDestroyed || !isEditableRef.current) {
        emitInlineShown(requestId, null)
        return
      }

      if (inlinePendingRef.current?.requestId && inlinePendingRef.current.requestId !== requestId) {
        clearInlineAiSuggestion(liveEditor)
      }

      let range = resolveDocRange(detail)
      const docSize = liveEditor.state.doc.content.size
      if (detail.isFullDoc && docSize > 0) {
        range = { from: 0, to: Math.max(docSize, 1) }
      }
      if (!range || range.to <= range.from) {
        emitInlineShown(requestId, null)
        return
      }

      const previewSource = detail.acceptedContent || detail.suggestedHtml || detail.suggestedPlain
      const detailTargetType = String(detail.targetType || detail.target_type || '').toLowerCase()
      const rangeForPreview = resolveTableReplacementRange(liveEditor, range, previewSource, detailTargetType)
      range = rangeForPreview || range
      const tableContext = extractTableContext(liveEditor, range.from, range.to)
      const hasTableTarget = tableContext.length > 0 || htmlContainsTable(previewSource)
      const normalizedTableHtml = hasTableTarget
        ? normalizeTableSuggestionHtml(
          previewSource,
          tableContext,
        )
        : ''
      const suggestedPlain = String(detail.suggestedPlain || '').trim()
      const suggestedHtml = normalizedTableHtml || detail.suggestedHtml || null
      const acceptedContent = normalizedTableHtml || detail.acceptedContent || null
      if (!suggestedPlain && !suggestedHtml) {
        emitInlineShown(requestId, null)
        return
      }

      inlinePendingRef.current = {
        requestId,
        ...range,
        suggestedPlain,
        suggestedHtml,
        acceptedContent,
        selectedFraction: Number(detail.selectedFraction) || 0,
        structuredData: detail.structuredData || null,
        action: detail.action,
        targetType: detailTargetType,
        isFullDoc: Boolean(detail.isFullDoc),
        originalText: detail.originalText || liveEditor.state.doc.textBetween(range.from, range.to, '\n'),
        tableContext,
      }

      notifyPreviewSession(true)
      setInlineAiSuggestion(liveEditor, {
        from: range.from,
        to: range.to,
        suggestedPlain,
        suggestedHtml,
      })
      try {
        liveEditor.commands.focus()
        liveEditor.commands.setTextSelection({ from: range.from, to: range.to })
        liveEditor.commands.scrollIntoView()
      } catch {
        // non-fatal
      }
      emitInlineShown(requestId, range)
    })

    const unsubClear = subscribeEditorInlineSuggestionClear(({ requestId }) => {
      const liveEditor = editorRef.current
      const pending = inlinePendingRef.current
      if (requestId && pending?.requestId && pending.requestId !== requestId) return
      if (liveEditor && !liveEditor.isDestroyed) {
        clearInlineAiSuggestion(liveEditor)
      }
      inlinePendingRef.current = null
      notifyPreviewSession(false)
    })

    const unsubApply = subscribeEditorInlineSuggestionApply(({ requestId }) => {
      const liveEditor = editorRef.current
      const pending = inlinePendingRef.current
      if (!pending || pending.requestId !== requestId) {
        emitInlineApplied(requestId, false, 'No pending suggestion to apply.')
        return
      }
      if (!liveEditor || liveEditor.isDestroyed) {
        emitInlineApplied(requestId, false, 'Editor is not available.')
        return
      }

      try {
        const {
          from,
          to,
          suggestedPlain,
          suggestedHtml,
          acceptedContent,
          isFullDoc,
          action,
          targetType,
          structuredData,
        } = pending
        const insertPayload =
          acceptedContent
          || (isFullDoc ? suggestedHtml : suggestedPlain)
          || suggestedHtml
          || suggestedPlain

        if (isFullDoc) {
          liveEditor.commands.setContent(insertPayload || '<p></p>', false)
        } else if (typeof insertPayload === 'string' && /<\/?[a-z]/i.test(insertPayload)) {
          const applyRange = resolveTableReplacementRange(liveEditor, { from, to }, insertPayload, targetType)
          liveEditor.chain().focus().insertContentAt(applyRange || { from, to }, insertPayload).run()
        } else {
          liveEditor.chain().focus().insertContentAt({ from, to }, insertPayload || '').run()
        }
        clearInlineAiSuggestion(liveEditor)
        inlinePendingRef.current = null
        notifyPreviewSession(false)
        emitInlineApplied(requestId, true)
        if (typeof onAfterApply === 'function') {
          onAfterApply({
            action,
            applied_scope: isFullDoc ? 'full_document' : 'selection',
            source: 'actions_tab',
            suggestion_id: structuredData?.suggestion_id || null,
          })
        }
      } catch (err) {
        console.error('[editor-actions-bridge] apply failed', err)
        emitInlineApplied(requestId, false, err?.message || 'Could not apply suggestion.')
      }
    })

    const onScrollToRange = (event) => {
      const liveEditor = editorRef.current
      const { from, to } = event.detail || {}
      if (!liveEditor || liveEditor.isDestroyed || from == null || to == null) return
      try {
        liveEditor.chain().focus().setTextSelection({ from, to }).scrollIntoView().run()
      } catch (err) {
        console.warn('[editor-actions-bridge] scrollIntoView failed', err)
      }
    }
    window.addEventListener(EDITOR_SCROLL_TO_RANGE_EVENT, onScrollToRange)

    const onGapAppend = (event) => {
      const liveEditor = editorRef.current
      const html = event.detail?.html
      if (!liveEditor || liveEditor.isDestroyed || !html) return
      try {
        const docEnd = liveEditor.state.doc.content.size
        const appendix = /<h3/i.test(String(html))
          ? html
          : `<h3>AI Gap Check Findings</h3>${html}`
        liveEditor.chain().focus().insertContentAt(docEnd, appendix, { updateSelection: false }).run()
      } catch (err) {
        console.warn('[editor-actions-bridge] gap append failed', err)
      }
    }
    window.addEventListener(EDITOR_GAP_APPEND_EVENT, onGapAppend)

    return () => {
      unsubSnapshot()
      unsubShow()
      unsubClear()
      unsubApply()
      window.removeEventListener(EDITOR_SCROLL_TO_RANGE_EVENT, onScrollToRange)
      window.removeEventListener(EDITOR_GAP_APPEND_EVENT, onGapAppend)
    }
  }, [editor, emitInlineApplied, emitInlineShown, notifyPreviewSession, onAfterApply, resolveDocRange])

  useEffect(() => {
    const unsubscribe = subscribeEditorAiActionRequest((request) => {
      if (!request || !request.action) return
      console.info('[kl-editor-bridge-received]', {
        action: request.action,
        requestId: request.requestId,
        source: request.source,
      })
      if (request.action === EDITOR_AI_ACTIONS.READ) {
        handleReadRequest(request)
        return
      }
      runActionRequest(request)
    })
    return unsubscribe
  }, [handleReadRequest, runActionRequest])

  const handleAccept = useCallback(() => {
    const liveEditor = editorRef.current
    const request = activeRequestRef.current
    const target = targetRangeRef.current
    if (!liveEditor || liveEditor.isDestroyed || !aiResult || !request) {
      closeModal()
      return
    }

    const action = String(aiResult.action || request.action || '').toLowerCase()
    const suggestedHtml = aiResult.suggested_text || ''
    const structuredData = aiResult.structured_data || {}

    try {
      if (action === EDITOR_AI_ACTIONS.GAP_CHECK) {
        const appendix = buildFindingsHtml(structuredData) || `<h3>AI Gap Check Findings</h3>${suggestedHtml}`
        const docEnd = liveEditor.state.doc.content.size
        liveEditor
          .chain()
          .focus()
          .insertContentAt(docEnd, appendix, { updateSelection: false })
          .run()
        console.info('[kl-editor-action-inserted]', { action, scope: 'append', requestId: request?.requestId })
      } else if (isFullDocRef.current) {
        let payloadHtml = suggestedHtml
        if (action === EDITOR_AI_ACTIONS.REWRITE) {
          payloadHtml = formatAiSuggestionForUi({
            action,
            suggestedText: structuredData?.rewritten_text || aiResult?.suggested_text,
            structuredData,
          })
        } else if (
          action === EDITOR_AI_ACTIONS.IMPROVE
          || action === EDITOR_AI_ACTIONS.SUMMARIZE
          || action === EDITOR_AI_ACTIONS.ANALYZE
        ) {
          const improvedSource =
            structuredData?.improved_text || structuredData?.improved_version || aiResult?.suggested_text
          payloadHtml = formatAiSuggestionForUi({
            action: EDITOR_AI_ACTIONS.IMPROVE,
            suggestedText: improvedSource,
            structuredData,
          })
        }
        liveEditor.commands.setContent(payloadHtml || suggestedHtml || '<p></p>', false)
        console.info('[kl-editor-action-inserted]', { action, scope: 'full_document', requestId: request?.requestId })
      } else {
        const from = target?.from ?? 0
        const to = target?.to ?? liveEditor.state.doc.content.size
        const targetId = String(target?.target_id || '').trim()
        if (targetId && targetId !== 'selection') {
          const liveTarget = findParsedTargetNodeById(liveEditor, targetId)
          if (!liveTarget) {
            throw new Error('The resolved target is no longer present in the open SOP. Re-run the action so I do not apply it to the wrong section.')
          }
        }
        const hasTableTarget = Boolean(tableTargetRef.current || aiResult?.contains_table_target)
        let plainContent = ''
        let htmlContent = ''
        if (action === EDITOR_AI_ACTIONS.REWRITE) {
          htmlContent = structuredData?.rewritten_text || aiResult?.suggested_text
          plainContent = stripHtml(htmlContent)
        } else if (
          action === EDITOR_AI_ACTIONS.IMPROVE
          || action === EDITOR_AI_ACTIONS.SUMMARIZE
          || action === EDITOR_AI_ACTIONS.ANALYZE
        ) {
          htmlContent = structuredData?.improved_text || structuredData?.improved_version || aiResult?.suggested_text
          plainContent = stripHtml(htmlContent)
        } else {
          htmlContent = aiResult?.suggested_text
          plainContent = stripHtml(htmlContent)
        }
        const richCandidate = hasTableTarget
          ? normalizeTableSuggestionHtml(htmlContent, tableContextRef.current)
          : htmlContent
        const insertPayload = /<\/?(?:table|thead|tbody|tr|td|th)\b/i.test(String(richCandidate || ''))
          ? richCandidate
          : plainContent
        const applyRange = resolveTableReplacementRange(
          liveEditor,
          { from, to },
          insertPayload,
          targetRangeRef.current?.target_type,
        )
        liveEditor
          .chain()
          .focus()
          .insertContentAt(applyRange || { from, to }, insertPayload || '')
          .run()
        console.info('[kl-editor-action-inserted]', { action, scope: 'selection', requestId: request?.requestId })
      }

      emitResult({
        ...request,
        status: EDITOR_AI_ACTION_STATUS.APPLIED,
        action,
        applied_scope: isFullDocRef.current ? 'full_document' : 'selection',
        sop_id: documentIdRef.current || null,
      })
      console.info('[kl-editor-action-accepted]', { action, requestId: request?.requestId })
      tableContextRef.current = []

      if (typeof onAfterApply === 'function') {
        try {
          onAfterApply({
            action,
            applied_scope: isFullDocRef.current ? 'full_document' : 'selection',
            suggestion_id: structuredData?.suggestion_id || null,
          })
        } catch (err) {
          console.error('[editor-ai-bridge] onAfterApply failed', err)
        }
      }
    } catch (err) {
      console.error('[editor-ai-bridge] failed to apply suggestion', err)
      emitResult({
        ...request,
        status: EDITOR_AI_ACTION_STATUS.ERROR,
        message: err?.message || 'Konnte Vorschlag nicht im Editor anwenden.',
      })
    } finally {
      closeModal()
    }
  }, [aiResult, closeModal, emitResult, onAfterApply])

  const handleReject = useCallback(() => {
    const request = activeRequestRef.current
    const action = String(aiResult?.action || request?.action || '').toLowerCase()
    if (request) {
      emitResult({
        ...request,
        status: EDITOR_AI_ACTION_STATUS.CANCELLED,
        action,
      })
    }
    closeModal()
  }, [aiResult, closeModal, emitResult])

  return (
    <>
      <AIComparisonModal
        isOpen={isModalOpen}
        onClose={handleReject}
        action={aiResult?.action}
        originalText={aiResult?.original_text}
        suggestedText={aiResult?.suggested_text}
        explanation={aiResult?.explanation}
        structuredData={aiResult?.structured_data}
        onAccept={handleAccept}
        sectionName={aiResult?.section_name}
        sopTitle={sopTitle}
      />
      {isLoading ? (
        <div className="editor-ai-bridge-loading" role="status" aria-live="polite">
          <div className="editor-ai-bridge-loading__inner">
            <span className="editor-ai-bridge-loading__spinner" />
            <span>KI-Assistent bearbeitet die SOP…</span>
          </div>
        </div>
      ) : null}
    </>
  )
}

export default EditorAIBridge
