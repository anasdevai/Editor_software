/**
 * Shared rewrite / improve handling for bubble menu and Actions tab.
 * Keeps API payloads, extracted text, and editor insert content identical.
 */

import { formatAiSuggestionForUi } from './aiOutputFormatter'
import { sanitizeRenderedHtml } from './aiOutputFormatter'

export const stripHtmlToPlain = (value) =>
  String(value || '')
    .replace(/<br\s*\/?>/gi, '\n')
    .replace(/<\/p>/gi, '\n')
    .replace(/<\/div>/gi, '\n')
    .replace(/<[^>]+>/g, '')
    .replace(/\n{3,}/g, '\n\n')
    .trim()

/**
 * Normalize /api/ai/action response the same way the selection bubble menu does.
 */
export function normalizeAiActionResult(action, apiResult) {
  const actionKey = String(action || apiResult?.action || '').toLowerCase()
  const structured = apiResult?.structured_data || {}
  const suggestedHtml = formatAiSuggestionForUi({
    action: actionKey,
    suggestedText: apiResult?.suggested_text,
    structuredData: structured,
  })

  let suggestedPlain = ''
  if (actionKey === 'rewrite') {
    suggestedPlain = stripHtmlToPlain(structured.rewritten_text || apiResult?.suggested_text)
  } else if (actionKey === 'improve') {
    suggestedPlain = stripHtmlToPlain(
      structured.improved_text || structured.improved_version || apiResult?.suggested_text,
    )
  } else if (actionKey === 'gap_check') {
    suggestedPlain = stripHtmlToPlain(structured.analysis || apiResult?.suggested_text)
  } else {
    suggestedPlain = stripHtmlToPlain(apiResult?.suggested_text)
  }

  if (!suggestedPlain && suggestedHtml) {
    suggestedPlain = stripHtmlToPlain(suggestedHtml)
  }

  return {
    action: actionKey,
    structured,
    suggestedHtml,
    suggestedPlain,
    explanation: apiResult?.explanation || '',
    originalText: apiResult?.original_text || '',
    raw: apiResult,
  }
}

/**
 * Content to insert on accept — mirrors AIAssistantBubbleMenu.buildAcceptedContent.
 */
export function buildAcceptedInsertContent(aiResult, { selectedFraction = 0, isFullDoc = false, preferRichBlock = false } = {}) {
  const action = String(aiResult?.action || '').toLowerCase()
  const structured = aiResult?.structured_data || {}
  const isPartialSelection = !isFullDoc && selectedFraction > 0 && selectedFraction < 0.85
  const richHtml = formatAiSuggestionForUi({
    action,
    suggestedText: aiResult?.suggested_text,
    structuredData: structured,
  })
  const hasRichTable = /<\/?(?:table|thead|tbody|tr|td|th)\b/i.test(richHtml)

  if (preferRichBlock && (action === 'rewrite' || action === 'improve')) {
    return richHtml || stripHtmlToPlain(structured.rewritten_text || structured.improved_text || aiResult?.suggested_text)
  }

  if (isPartialSelection && (action === 'rewrite' || action === 'improve')) {
    if (hasRichTable) return richHtml
    if (action === 'rewrite') {
      return stripHtmlToPlain(structured.rewritten_text || aiResult?.suggested_text)
    }
    if (action === 'improve') {
      return stripHtmlToPlain(structured.improved_text || structured.improved_version || aiResult?.suggested_text)
    }
  }

  return richHtml || stripHtmlToPlain(structured.rewritten_text || structured.improved_text || aiResult?.suggested_text)
}

/** Safe HTML for inline suggestion widget (no raw markdown). */
export function buildInlineSuggestionHtml(normalized) {
  const html = normalized?.suggestedHtml || ''
  if (html && /<\/?[a-z]/i.test(html)) {
    return sanitizeRenderedHtml(html)
  }
  const plain = normalized?.suggestedPlain || ''
  if (!plain) return '<p></p>'
  return plain
    .split(/\n{2,}/)
    .map((block) => `<p>${block.replace(/\n/g, '<br>')}</p>`)
    .join('')
}

export function inferSectionMetaFromEditor(editor, from, to, docSize) {
  let sectionName = 'Selected text'
  let sectionType = 'Paragraph'
  const fraction = docSize > 0 ? Math.abs(to - from) / docSize : 0

  if (!editor || editor.isDestroyed) {
    return { sectionName, sectionType, selectedFraction: fraction }
  }

  if (fraction >= 0.85) {
    return { sectionName: 'Full Document', sectionType: 'Full Document', selectedFraction: fraction }
  }

  try {
    const resolvedPos = editor.state.doc.resolve(from)
    for (let depth = resolvedPos.depth; depth >= 0; depth -= 1) {
      const node = resolvedPos.node(depth)
      if (node.type.name === 'heading') {
        sectionName = node.textContent
        sectionType = 'Heading'
        break
      }
      if (node.type.name === 'table') sectionType = 'Table'
      else if (node.type.name === 'bulletList' || node.type.name === 'orderedList' || node.type.name === 'listItem') {
        sectionType = 'List'
      } else if (node.type.name === 'paragraph') {
        sectionType = 'Paragraph'
      }
    }
  } catch {
    // best effort
  }

  return { sectionName, sectionType, selectedFraction: fraction }
}
