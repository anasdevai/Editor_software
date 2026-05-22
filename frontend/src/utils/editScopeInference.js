/**
 * Infer edit_scope for /api/ai/action (section_only vs full_document).
 */

import { resolveTargetInEditor } from './editorTargetResolver.js'
import { getTraceabilitySectionKind, wantsFullSopIntent } from './sopActionIntent.js'

const RECORD_ID_RE = /\b(?:DEV|CAPA|AUD|DEC)-[A-Z0-9]+-\d+\b/gi
const BACKBONE_RE =
  /(?:^|\n)\s*(?:#{1,6}\s*)?(?:1\.|2\.|3\.|4\.|5\.)\s*(?:ZWECK|PURPOSE|GELTUNGSBEREICH|SCOPE|VERANTWORTLICH|VERFAHREN|PROCEDURE)/im
const NUMBERED_SECTION_HEADING_RE =
  /^\s*(?:#{1,6}\s*)?\d+(?:\.\d+)*[.)]?\s+[\p{L}\p{N}][\p{L}\p{N}\s/&()|:.,-]{1,120}$/iu
const NAMED_SECTION_HEADING_RE =
  /^\s*(?:#{1,6}\s*)?(?:zweck|purpose|ziel|scope|geltungsbereich|anwendungsbereich|responsibilit(?:y|ies)|verantwortlich(?:keiten)?|procedure|verfahren|definitions?|definitionen|deviations?|abweichungen|capas?|caps|capa|audit(?:\s+findings?)?|entscheidungen|decisions?)\b/iu
const REGISTER_ID_HEADING_RE = /^\s*(?:DEV|CAPA|AUD|DEC)-[A-Z0-9]+-\d+\b/i

export function isTraceabilityRegisterSelection(text = '') {
  const t = String(text || '').trim()
  if (t.length < 80) return false
  const ids = t.match(RECORD_ID_RE) || []
  if (ids.length < 2) return false
  if (BACKBONE_RE.test(t)) return false
  return true
}

export function inferEditScope({ text = '', from = 0, to = 0, docSize = 0, instruction = '' } = {}) {
  if (wantsFullSopIntent(instruction)) {
    return 'full_document'
  }
  const t = String(text || '')
  if (isTraceabilityRegisterSelection(t)) {
    return 'section_only'
  }
  const fraction = docSize > 0 ? Math.abs(to - from) / docSize : 0
  if (fraction >= 0.92) {
    return 'full_document'
  }
  return 'section_only'
}

function expansionHintForSelection(text = '') {
  const t = String(text || '').trim()
  const traceabilityKind = getTraceabilitySectionKind(t)
  if (traceabilityKind) return traceabilityKind
  const recordPrefix = (t.match(REGISTER_ID_HEADING_RE)?.[0] || '').slice(0, 4).toUpperCase()
  if (recordPrefix === 'CAPA') return 'capas'
  if (recordPrefix === 'DEV-') return 'deviations'
  if (recordPrefix === 'AUD-') return 'audit'
  if (recordPrefix === 'DEC-') return 'decisions'
  return t.replace(/\s+/g, ' ').trim()
}

function isExpandableSectionCue(text = '') {
  const t = String(text || '').trim()
  if (!t || t.length > 180) return false
  const lines = t.split(/\r?\n/).map((line) => line.trim()).filter(Boolean)
  if (lines.length > 2) return false
  if (getTraceabilitySectionKind(t)) return true
  if (REGISTER_ID_HEADING_RE.test(t)) return true
  if (NUMBERED_SECTION_HEADING_RE.test(t)) return true
  if (NAMED_SECTION_HEADING_RE.test(t)) return true
  return false
}

function expandHeadingSelectionToSection(editor, { from, to, text } = {}) {
  if (!isExpandableSectionCue(text)) return null
  const sectionHint = expansionHintForSelection(text)
  if (!sectionHint) return null
  try {
    const target = resolveTargetInEditor(editor, {
      prompt: `rewrite ${sectionHint} section`,
      userPrompt: `rewrite ${sectionHint} section`,
      sectionHint,
      targetScope: 'section',
    })
    if (!target?.text) return null
    const expandedFrom = Number(target.from)
    const expandedTo = Number(target.to)
    if (!Number.isFinite(expandedFrom) || !Number.isFinite(expandedTo)) return null
    if (expandedFrom > from || expandedTo < to) return null
    if (target.text.trim().length <= String(text || '').trim().length) return null
    return target
  } catch {
    return null
  }
}

/**
 * Snapshot the editor selection at action time (bubble menu / inline actions).
 * Expands near-full spans to the whole document so rewrite/improve/gap use the real range.
 */
export function captureEditorSelectionForAction(editor) {
  if (!editor || editor.isDestroyed) return null
  const { state } = editor
  const { selection } = state
  if (!selection || selection.empty) return null

  const docSize = state.doc.content.size
  let from = selection.from
  let to = selection.to
  let text = state.doc.textBetween(from, to, '\n').trim()
  if (!text) return null

  let selectedFraction = Math.abs(to - from) / Math.max(1, docSize)
  const startsNearDocStart = from <= 2
  const nearFullFromTop = startsNearDocStart && selectedFraction >= 0.75
  const nearFullSpan = selectedFraction >= 0.92 || nearFullFromTop

  if (nearFullSpan) {
    from = 0
    to = docSize
    text = state.doc.textBetween(from, to, '\n').trim()
    selectedFraction = 1
  } else {
    const expanded = expandHeadingSelectionToSection(editor, { from, to, text })
    if (expanded) {
      from = expanded.from
      to = expanded.to
      text = expanded.text.trim()
      selectedFraction = Math.abs(to - from) / Math.max(1, docSize)
    }
  }

  const editScope = nearFullSpan
    ? 'full_document'
    : inferEditScope({ text, from, to, docSize })

  const isFullDocument = editScope === 'full_document'
  if (isFullDocument) {
    from = 0
    to = docSize
    text = state.doc.textBetween(from, to, '\n').trim()
    selectedFraction = 1
  }

  return {
    from,
    to,
    selectedText: text,
    structuredText: text,
    selectedFraction,
    editScope,
    isFullDocument,
  }
}

export function inferSectionMetaForSelection(editor, snapshot) {
  if (!snapshot) {
    return { sectionName: 'Selected text', sectionType: 'Paragraph' }
  }
  if (snapshot.isFullDocument) {
    return { sectionName: 'Full Document', sectionType: 'Full Document' }
  }
  const selectedText = snapshot.structuredText || snapshot.selectedText || ''
  if (isTraceabilityRegisterSelection(selectedText)) {
    let sectionName = 'DEVIATIONS'
    try {
      const resolvedPos = editor.state.doc.resolve(snapshot.from)
      for (let depth = resolvedPos.depth; depth >= 0; depth -= 1) {
        const node = resolvedPos.node(depth)
        if (node.type.name === 'heading' && node.textContent) {
          sectionName = node.textContent
          break
        }
      }
    } catch {
      // best-effort
    }
    return { sectionName, sectionType: 'Section' }
  }

  let sectionName = 'Selected text'
  let sectionType = 'Paragraph'
  try {
    const resolvedPos = editor.state.doc.resolve(snapshot.from)
    for (let depth = resolvedPos.depth; depth >= 0; depth -= 1) {
      const node = resolvedPos.node(depth)
      if (node.type.name === 'heading') {
        sectionName = node.textContent
        sectionType = 'Heading'
        break
      }
      if (node.type.name === 'table') {
        sectionType = 'Table'
      } else if (
        node.type.name === 'bulletList' ||
        node.type.name === 'orderedList' ||
        node.type.name === 'listItem'
      ) {
        sectionType = 'List'
      } else if (node.type.name === 'paragraph') {
        sectionType = 'Paragraph'
      }
    }
  } catch {
    // best-effort
  }
  return { sectionName, sectionType }
}
