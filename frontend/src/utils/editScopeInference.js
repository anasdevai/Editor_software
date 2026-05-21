/**
 * Infer edit_scope for /api/ai/action (section_only vs full_document).
 */

import { wantsFullSopIntent } from './sopActionIntent'

const RECORD_ID_RE = /\b(?:DEV|CAPA|AUD|DEC)-[A-Z0-9]+-\d+\b/gi
const BACKBONE_RE =
  /(?:^|\n)\s*(?:#{1,6}\s*)?(?:1\.|2\.|3\.|4\.|5\.)\s*(?:ZWECK|PURPOSE|GELTUNGSBEREICH|SCOPE|VERANTWORTLICH|VERFAHREN|PROCEDURE)/im

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
