/**
 * Resolve which slice of the open SOP an Actions-tab rewrite/improve should target.
 */

const FULL_SOP_PATTERNS = [
  /\brewrite\s+(?:the\s+)?full\s+sop\b/i,
  /\bfull\s+sop\s+rewrite\b/i,
  /\bgesamte\s+sop\b/i,
  /\bkomplette\s+sop\b/i,
  /\bumschreib(?:en|e)\s+(?:die\s+)?(?:gesamte|komplette)\s+sop\b/i,
]

const SECTION_INTENT_PATTERNS = [
  /\b(?:rewrite|improve|enhance|polish|optimi[sz]e|refine|verbessere?n?)\s+(?:the\s+)?(.+?)\s+section\b/i,
  /\b(?:rewrite|improve)\s+(?:the\s+)?(.+?)\s+(?:teil|abschnitt)\b/i,
  /\b(?:section|abschnitt)\s+["']?(.+?)["']?\s+(?:rewrite|improve)\b/i,
]

const REWRITE_IMPROVE_VERB = /\b(?:rewrite|re-?write|improve|enhance|rephrase|umschreiben|überarbeiten|verbessern?)\b/i

const DOC_ID_PATTERN = /\b([A-Z]{2,}(?:-[A-Z0-9]+)+)\b/g

/** "DEVIATIONS (…) rewrite this" — label before the verb */
const LABEL_BEFORE_VERB_PATTERNS = [
  /^(.+?)\s+rewrite\s+this\b/i,
  /^(.+?)\s+improve\s+this\b/i,
  /^(.+?)\s+rewrite\b/i,
  /^(.+?)\s+improve\b/i,
  /^(.+?)\s+umschreiben\b/i,
  /^(.+?)\s+verbessern\b/i,
]

function escapeRegExp(value) {
  return String(value || '').replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

export function extractDocumentRefsFromPrompt(prompt) {
  const text = String(prompt || '')
  const refs = []
  let match = DOC_ID_PATTERN.exec(text)
  while (match) {
    refs.push(match[1])
    match = DOC_ID_PATTERN.exec(text)
  }
  DOC_ID_PATTERN.lastIndex = 0
  return [...new Set(refs)]
}

function extractLabelsFromPrompt(promptText) {
  const text = String(promptText || '').trim()
  const labels = []

  for (const pattern of LABEL_BEFORE_VERB_PATTERNS) {
    const match = text.match(pattern)
    if (match?.[1]) {
      labels.push(
        match[1]
          .trim()
          .replace(/\s*section\s*$/i, '')
          .replace(/[.!?]+$/, '')
          .trim(),
      )
    }
  }

  const afterVerb = text.match(/\b(?:rewrite|improve|rephrase|umschreiben|verbessern?)\s+(?:the\s+)?(.+)$/i)
  if (afterVerb?.[1]) {
    labels.push(
      afterVerb[1]
        .trim()
        .replace(/\s*section\s*$/i, '')
        .replace(/[.!?]+$/, '')
        .trim(),
    )
  }

  labels.push(...extractDocumentRefsFromPrompt(text))

  const unique = [...new Set(labels.map((l) => l.trim()).filter(Boolean))]
  return unique
}

function lineMatchesLabel(trimmed, labelRe) {
  const lineNorm = trimmed.replace(/^[^\p{L}\p{N}]+/u, '').trim()
  return labelRe.test(trimmed) || (lineNorm && labelRe.test(lineNorm))
}

/**
 * Locate a section by heading label in plain document text.
 */
export function findSectionRangeInPlainText(fullText, sectionLabel) {
  const text = String(fullText || '')
  const label = String(sectionLabel || '').trim()
  if (!text || !label) return null

  const candidates = [
    label,
    label.replace(/\s*\([^)]*\)\s*/g, ' ').replace(/\s+/g, ' ').trim(),
    label.split('(')[0].trim(),
    label.split(/\s+/)[0].trim(),
  ].filter((c, i, arr) => c && c.length >= 3 && arr.indexOf(c) === i)

  for (const candidate of candidates) {
    const found = findSectionRangeForLabel(text, candidate)
    if (found) return found
  }
  return null
}

function findSectionRangeForLabel(text, label) {
  const labelRe = new RegExp(escapeRegExp(label), 'i')
  const lines = text.split('\n')
  let startLine = -1
  let offset = 0

  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i]
    const trimmed = line.trim()
    const isHeading =
      /^#{1,6}\s+/.test(trimmed)
      || (/^[A-Z0-9\u00C0-\u024F][A-Za-z0-9\u00C0-\u024F\s/&–—-]{2,}$/u.test(trimmed) && trimmed.length < 140 && !trimmed.endsWith('.'))
      || /^\d+(\.\d+)*\s+/.test(trimmed)

    if ((isHeading || lineMatchesLabel(trimmed, labelRe)) && lineMatchesLabel(trimmed, labelRe)) {
      startLine = i
      break
    }
    offset += line.length + 1
  }

  if (startLine < 0) {
    const idx = text.search(labelRe)
    if (idx < 0) return null
    startLine = text.slice(0, idx).split('\n').length - 1
    offset = text.slice(0, idx).lastIndexOf('\n') + 1
    if (offset < 0) offset = 0
  } else {
    offset = lines.slice(0, startLine).join('\n').length
    if (startLine > 0) offset += 1
  }

  let endOffset = text.length
  let lineOffset = offset
  for (let i = startLine + 1; i < lines.length; i += 1) {
    const line = lines[i]
    const trimmed = line.trim()
    const isNextHeading =
      /^#{1,6}\s+/.test(trimmed)
      || (/^[A-Z0-9\u00C0-\u024F][A-Za-z0-9\u00C0-\u024F\s/&–—-]{2,}$/u.test(trimmed) && trimmed.length < 140 && !trimmed.endsWith('.'))
      || /^\d+(\.\d+)*\s+/.test(trimmed)
    if (isNextHeading && trimmed && !lineMatchesLabel(trimmed, labelRe)) {
      endOffset = lineOffset
      break
    }
    lineOffset += line.length + 1
  }

  const slice = text.slice(offset, endOffset).trim()
  if (!slice || slice.length < 10) return null

  return {
    start: offset,
    end: endOffset,
    label: lines[startLine]?.trim() || label,
    text: slice,
  }
}

export function mapPlainOffsetsToDocRange(doc, plainStart, plainEnd) {
  if (!doc) return null
  const size = doc.content.size
  const targetStart = Math.max(0, Number(plainStart) || 0)
  const targetEnd = Math.max(targetStart, Number(plainEnd) || 0)
  const plainLenAt = (pos) => doc.textBetween(0, Math.min(pos, size), '\n').length

  let from = 0
  let to = size
  for (let pos = 0; pos <= size; pos += 1) {
    if (plainLenAt(pos) >= targetStart) {
      from = pos
      break
    }
  }
  for (let pos = from; pos <= size; pos += 1) {
    if (plainLenAt(pos) >= targetEnd) {
      to = pos
      break
    }
  }

  from = Math.max(0, Math.min(from, size))
  to = Math.max(from, Math.min(to, size))
  return { from, to }
}

function tryResolveLabelInDocument(full, docSize, label) {
  if (!label) return null
  const sectionRange = findSectionRangeInPlainText(full, label)
  if (sectionRange?.text) {
    return {
      plainStart: sectionRange.start,
      plainEnd: sectionRange.end,
      text: sectionRange.text,
      isFullDoc: false,
      sectionName: sectionRange.label,
      sectionType: 'Section',
    }
  }
  return null
}

function resolveFromPromptLabels(full, docSize, promptText) {
  for (const pattern of SECTION_INTENT_PATTERNS) {
    const match = promptText.match(pattern)
    if (match?.[1]) {
      const resolved = tryResolveLabelInDocument(full, docSize, match[1].trim())
      if (resolved) return resolved
    }
  }

  const labels = extractLabelsFromPrompt(promptText)
  for (const label of labels) {
    const resolved = tryResolveLabelInDocument(full, docSize, label)
    if (resolved) return resolved
  }
  return null
}

/**
 * @param {object} opts
 * @param {string} opts.fullText
 * @param {number} opts.docSize
 * @param {object|null} opts.selection
 * @param {string} [opts.prompt]
 */
export function resolveActionTarget({ fullText, docSize, selection, prompt = '' }) {
  const promptText = String(prompt || '').trim()
  const full = String(fullText || '')
  const hasPromptIntent = REWRITE_IMPROVE_VERB.test(promptText)

  for (const pattern of FULL_SOP_PATTERNS) {
    if (pattern.test(promptText)) {
      return {
        from: 0,
        to: docSize,
        text: full.trim(),
        isFullDoc: true,
        sectionName: 'Full SOP',
        sectionType: 'Full Document',
      }
    }
  }

  if (hasPromptIntent) {
    const fromPrompt = resolveFromPromptLabels(full, docSize, promptText)
    if (fromPrompt) return fromPrompt
  }

  if (selection && !selection.empty) {
    const fragment = String(selection.text || '').trim()
    if (fragment) {
      return {
        from: selection.from,
        to: selection.to,
        text: fragment,
        isFullDoc: false,
        sectionName: 'Selected text',
        sectionType: 'Paragraph',
      }
    }
  }

  if (hasPromptIntent) {
    const labels = extractLabelsFromPrompt(promptText)
    throw new Error(
      `Could not find "${labels[0] || 'that section'}" in the SOP. Select the text in the editor, or check the heading spelling.`,
    )
  }

  return null
}
