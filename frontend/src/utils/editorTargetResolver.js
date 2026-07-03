/**
 * Find rewrite/improve/gap targets in the TipTap document (accurate from/to).
 */

import {
  extractSopRefs,
  getTraceabilitySectionKind,
  isGenericLabelToken,
  isTraceabilitySectionHeading,
  wantsFullSopIntent,
} from './sopActionIntent.js'

const FULL_SOP_PATTERNS = [
  /\brewrite\s+(?:the\s+)?full\s+sop\b/i,
  /\brewrite\s+this\s+full\s+sop\b/i,
  /\bimprove\s+this\s+full\s+sop\b/i,
  /\bfull\s+sop\s+rewrite\b/i,
  /\b(?:entire|whole|complete)\s+sop\b/i,
  /\bgesamte\s+sop\b/i,
  /\bkomplette\s+sop\b/i,
]

const REWRITE_IMPROVE_VERB = /\b(?:rewrite|re-?write|improve|enhance|rephrase|umschreiben|überarbeiten|verbessern?)\b/i

const GAP_CHECK_VERB =
  /\b(?:gap\s*check|gap\s*analysis|what\s+(?:is|are)\s+the\s+gaps?|gaps?\s+in|compliance\s+(?:gap|check|review)|l(?:ü|ue|u)cken[\s-]?(?:analyse|pr(?:ü|ue)fung|check)?|(?:finde|zeige)\s+(?:die\s+)?l(?:ü|ue|u)cken|welche\s+l(?:ü|ue|u)cken)\b/i

const SUMMARIZE_VERB =
  /\b(?:summarize|summary|zusammenfass|kurzfass|fasse\s+zusammen|verkürz|verkuerz)\b/i

const FULL_SOP_GAP_PATTERNS = [
  /\bgap\s*check\s+(?:this\s+)?sop\b/i,
  /\bwhat\s+(?:is|are)\s+the\s+gaps?\s+in\s+(?:this\s+)?sop\b/i,
  /\bgaps?\s+in\s+(?:this\s+)?sop\b/i,
  /\b(?:finde|zeige)\s+(?:die\s+)?l(?:ü|ue|u)cken\s+(?:in\s+)?(?:dieser|diese|der|die)?\s*sop\b/i,
  /\bl(?:ü|ue|u)cken\s+in\s+(?:dieser|diese|der|die)?\s*sop\b/i,
  /\bcompliance\s+(?:check|review)\s+(?:of\s+)?(?:this\s+)?sop\b/i,
]

const DOC_ID_PATTERN = /\b([A-Z]{2,}(?:-[A-Z0-9]+)+)\b/g

const RECORD_ENTRY_RE = /^\s*(?:DEV|CAPA|AUD|DEC)-[A-Z0-9]+-\d+/i
const TABLE_OF_CONTENTS_HEADING_RE = /^(?:table\s+of\s+contents|contents|inhaltsverzeichnis)$/i

const GAP_LABEL_PATTERNS = [
  /what\s+(?:is|are)\s+the\s+gaps?\s+(?:in|for)\s+(.+?)(?:\s+section)?\s*$/i,
  /gap\s+check\s+(?:on|for|in)?\s*(?:the\s+)?(.+?)(?:\s+section)?\s*$/i,
  /gaps?\s+in\s+(?:the\s+)?(.+?)(?:\s+section)?\s*$/i,
]

const SECTION_LABEL_PATTERNS = [
  /^rewrite\s+this\s+(.+?)\s+sections?(?:\s+only)?\s*$/i,
  /^improve\s+this\s+(.+?)\s+sections?(?:\s+only)?\s*$/i,
  /^summarize\s+this\s+(.+?)\s+sections?(?:\s+only)?\s*$/i,
  /^rewrite\s+(?:the\s+)?(.+?)\s+sections?(?:\s+only)?\s*$/i,
  /^improve\s+(?:the\s+)?(.+?)\s+sections?(?:\s+only)?\s*$/i,
  /^summarize\s+(?:the\s+)?(.+?)\s+sections?(?:\s+only)?\s*$/i,
  /^(.+?)\s+rewrite\s+this\s*$/i,
  /^(.+?)\s+improve\s+this\s*$/i,
  /^(.+?)\s+summarize\s+this\s*$/i,
  /^(.+?)\s+rewrite\b/i,
  /^(.+?)\s+improve\b/i,
  /^(.+?)\s+summarize\b/i,
  /^fasse\s+(?:den|die|das)?\s*(.+?)\s+zusammen\s*$/i,
  /^im\s+abschnitt\s+(.+?)\s*$/i,
]

function normalizeForMatch(value) {
  return String(value || '')
    .replace(/[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}]/gu, ' ')
    .replace(/^[^\p{L}\p{N}]+/gu, '')
    .replace(/^\s*\d+(?:\.\d+)*[.)\]:-]?\s*/, '')
    .replace(/\s+/g, ' ')
    .trim()
    .toLowerCase()
}

const SECTION_ALIAS_MAP = {
  abbreviations: ['abbreviations', 'abbreviation', 'abbrev', 'acronyms', 'abkuerzungen', 'abkuerzung'],
  purpose: ['purpose', 'zweck', 'ziel', 'aim', 'objective', 'sweck', 'zwect', 'zwek'],
  scope: ['scope', 'geltungsbereich', 'anwendungsbereich', 'bereich'],
  responsibilities: ['responsibilities', 'responsibility', 'verantwortlichkeiten', 'verantwortung', 'roles'],
  definitions: ['definitions', 'definitionen', 'begriffe', 'abkuerzungen', 'abkürzungen'],
  definitions: ['definitions', 'definition', 'definitionen', 'begriffe', 'terms', 'term'],
  procedure: ['procedure', 'verfahren', 'prozess', 'process', 'steps', 'vorgehen'],
  deviations: ['deviations', 'deviation', 'abweichungen', 'abweichung', 'dev'],
  capas: ['capas', 'capa', 'cap', 'caps', 'corrective actions', 'corrective and preventive actions', 'korrekturmaßnahmen', 'korrekturmassnahmen'],
  caps: ['capas', 'capa', 'caps'],
  audits: ['audits', 'audit', 'audit findings', 'auditbericht', 'auditberichte'],
}

function aliasCandidates(label) {
  const norm = normalizeForMatch(label)
  const out = new Set([norm])
  if (norm === 'caps') out.add('capas')
  Object.entries(SECTION_ALIAS_MAP).forEach(([canonical, aliases]) => {
    const normalizedAliases = aliases.map(normalizeForMatch)
    if (norm === canonical || normalizedAliases.includes(norm)) {
      out.add(canonical)
      normalizedAliases.forEach((a) => out.add(a))
    }
  })
  return [...out].filter(Boolean)
}

function cleanLabel(raw) {
  const original = String(raw || '').trim()
  const isTableSectionRequest = /^\s*(?:table|tabelle)\b/i.test(original) && /\bsections?\s*$/i.test(original)
  const quoted = original.match(/["']([^"']{2,160})["']/)
  const base = quoted?.[1] || original
  return base
    .trim()
    .replace(/^["']+|["']+$/g, '')
    .replace(/^(?:ok(?:ay)?|now|then|please)\s+/i, '')
    .replace(/\s+(?:in|using|with|for)\s+(?:the\s+)?(?:current\s+)?(?:sop|document|style)[\s\S]*$/i, '')
    .replace(/\s+sections?\s+only\s*$/i, '')
    .replace(/\s+only\s*$/i, '')
    .replace(/\s+(?:sections?\s+)?tables?\s*$/i, '')
    .replace(isTableSectionRequest ? /\s*$/ : /\s*sections?\s*$/i, '')
    .replace(/\bfull\s+sop\b/gi, '')
    .replace(/[.!?]+$/, '')
    .trim()
}

function isContextualSectionReference(label) {
  const normalized = normalizeForMatch(label)
  return /^(?:summarize|explain|rewrite|improve|shorten|expand)?\s*(?:this|that|it|them)(?:\s+section)?$/.test(normalized)
}

function buildLabelCandidates(label) {
  const cleaned = cleanLabel(label)
  if (!cleaned || wantsFullSopIntent(cleaned) || isContextualSectionReference(cleaned)) return []

  const candidates = new Set()
  const norm = normalizeForMatch(cleaned)
  if (norm && !isGenericLabelToken(norm)) candidates.add(norm)
  aliasCandidates(cleaned).forEach((alias) => {
    if (alias && !isGenericLabelToken(alias)) candidates.add(alias)
  })

  const noParen = norm.replace(/\s*\([^)]*\)\s*/g, ' ').replace(/\s+/g, ' ').trim()
  if (noParen && !isGenericLabelToken(noParen)) candidates.add(noParen)

  const withoutTrailingSection = norm.replace(/\s+sections?$/i, '').trim()
  if (withoutTrailingSection && !isGenericLabelToken(withoutTrailingSection)) candidates.add(withoutTrailingSection)

  const kind = getTraceabilitySectionKind(cleaned)
  if (kind) candidates.add(kind)

  for (const ref of extractSopRefs(cleaned)) {
    candidates.add(ref)
  }

  return [...candidates].filter((c) => c && c.length >= 3 && !isGenericLabelToken(c))
}

function stripLeadingNumbering(value) {
  return normalizeForMatch(value).replace(/^\d+(?:\.\d+)*[.)\]:-]?\s+/, '').trim()
}

function leadingSectionNumber(value) {
  const match = String(value || '').trim().match(/^(\d+(?:\.\d+)*)/)
  return match ? match[1] : ''
}

function semanticSectionRootsMatch(label, blockText) {
  const a = stripLeadingNumbering(label)
  const b = stripLeadingNumbering(blockText)
  if (!a || !b) return false
  if (a === b) return true
  const rootsA = new Set(aliasCandidates(a).map((x) => stripLeadingNumbering(x) || x))
  const rootsB = new Set(aliasCandidates(b).map((x) => stripLeadingNumbering(x) || x))
  for (const ra of rootsA) {
    if (rootsB.has(ra)) return true
  }
  return false
}

function numberedSectionPrefixesMatch(label, blockText) {
  const want = leadingSectionNumber(label)
  const have = leadingSectionNumber(blockText)
  if (!want) return true
  if (!have) return false
  return want === have
}

function editDistance(a, b) {
  const left = String(a || '')
  const right = String(b || '')
  if (!left) return right.length
  if (!right) return left.length

  const previous = Array.from({ length: right.length + 1 }, (_, index) => index)
  const current = Array(right.length + 1).fill(0)

  for (let i = 1; i <= left.length; i += 1) {
    current[0] = i
    for (let j = 1; j <= right.length; j += 1) {
      const cost = left[i - 1] === right[j - 1] ? 0 : 1
      current[j] = Math.min(
        previous[j] + 1,
        current[j - 1] + 1,
        previous[j - 1] + cost,
      )
    }
    for (let j = 0; j <= right.length; j += 1) previous[j] = current[j]
  }

  return previous[right.length]
}

const ASSISTANT_CONSTRAINTS_MARKER = /\n\s*\[Assistant constraints\]/i

const TARGET_SECTION_CONSTRAINT_RE = /^\s*Target section:\s*(.+)$/im

/** User message only — label patterns must not run on appended classifier hints. */
export function stripAssistantConstraints(promptText = '') {
  const raw = String(promptText || '')
  const idx = raw.search(ASSISTANT_CONSTRAINTS_MARKER)
  return (idx >= 0 ? raw.slice(0, idx) : raw).trim()
}

/** True when the user asked to rewrite/improve/check a named or contextual section. */
export function wantsSectionScopeIntent(promptText = '') {
  const text = stripAssistantConstraints(promptText)
  if (!text || wantsFullSopIntent(text)) return false
  if (/\b(?:this|that|the)\s+section\b/i.test(text)) return true
  if (/\b(?:rewrite|improve|gap\s*check|umschreiben|verbessern?)\b[\s\S]*\bsections?\b/i.test(text)) return true
  if (/\bsections?\s+(?:rewrite|improve|only)\b/i.test(text)) return true
  if (TARGET_SECTION_CONSTRAINT_RE.test(String(promptText || ''))) return true
  return extractLabelsFromPrompt(text).length > 0
}

function extractLabelsFromPrompt(promptText) {
  if (wantsFullSopIntent(promptText)) return []

  const text = stripAssistantConstraints(promptText)
  const labels = []

  for (const pattern of GAP_LABEL_PATTERNS) {
    const match = text.match(pattern)
    if (match?.[1]) labels.push(cleanLabel(match[1]))
  }

  const firstLine = text.split(/\r?\n/)[0]?.trim() || text

  const tableSection = firstLine.match(
    /\b(?:rewrite|improve|rephrase|umschreiben|verbessern?|summarize|formalize|shorten|expand)\s+(?:this\s+)?(?:the\s+)?((?:table|tabelle)\s+\d+[\s\S]+?\s+sections?)\s*$/i,
  )
  if (tableSection?.[1]) labels.push(cleanLabel(tableSection[1]))

  for (const pattern of SECTION_LABEL_PATTERNS) {
    const match = firstLine.match(pattern)
    if (match?.[1]) labels.push(cleanLabel(match[1]))
  }

  const afterVerb = firstLine.match(
    /\b(?:rewrite|improve|rephrase|umschreiben|verbessern?|summarize|formalize|shorten|expand)\s+(?:this\s+)?(?:the\s+)?(.+?)(?:\s+sections?(?:\s+only)?)?\s*$/i,
  )
  if (afterVerb?.[1]) {
    const captured = cleanLabel(afterVerb[1])
    if (captured && !wantsFullSopIntent(captured) && !isContextualSectionReference(captured)) labels.push(captured)
  }

  const namedSection = text.match(
    /\b(?:the\s+)?([A-Za-zÀ-ÿ][\w\s\-&]{1,50}?)\s+sections?\b/i,
  )
  if (namedSection?.[1]) {
    const captured = cleanLabel(namedSection[1])
    if (captured && !wantsFullSopIntent(captured) && !isContextualSectionReference(captured)) labels.push(captured)
  }

  const constraintSection = String(promptText || '').match(TARGET_SECTION_CONSTRAINT_RE)
  if (constraintSection?.[1]) {
    labels.push(cleanLabel(constraintSection[1]))
  }

  let m = DOC_ID_PATTERN.exec(text)
  while (m) {
    labels.push(m[1])
    m = DOC_ID_PATTERN.exec(text)
  }
  DOC_ID_PATTERN.lastIndex = 0

  return [
    ...new Set(
      labels
        .map((l) => l.trim())
        .filter((l) => l.length >= 2 && !wantsFullSopIntent(l) && !isGenericLabelToken(normalizeForMatch(l))),
    ),
  ]
}

function isRecordEntryLine(text) {
  return RECORD_ENTRY_RE.test(String(text || '').trim())
}

const REGISTER_FIELD_LINE_RE =
  /^(?:Linked|Finding|Datum|Beschreibung|Ursache|Aktion|Verantwortlich|Verknüpfungen|Entscheidung|Risiko|Begründung|Status|Fällig|Schweregrad|Ergebnis)\s*:/i

const EMBEDDED_TRACEABILITY_MARKER_RE =
  /(?:[\u{1F534}\u{1F7E0}\u{1F7E1}\u{1F7E2}\u{1F535}\u{1F7E3}]\s*)?(?:DEVIATIONS?|ABWEICHUNGEN?|CAPAS?|AUDIT(?:\s+FINDINGS?)?|AUDITS?|DECISIONS?|ENTSCHEIDUNGEN?)\b(?:\s*\([^)]*SOP-[^)]+\))?/giu

function isTraceabilitySectionTitle(text) {
  const t = String(text || '').trim()
  return Boolean(t && t.length < 160 && !/[.!?]$/u.test(t) && isTraceabilitySectionHeading(t))
}

function traceabilityKindForBlock(block) {
  if (!block) return null
  if (block.sectionKind) return block.sectionKind
  const head = String(block.text || '').split(/\r?\n/).map((line) => line.trim()).find(Boolean) || ''
  return getTraceabilitySectionKind(head)
}

function isRealTraceabilitySectionStart(block, targetKind) {
  if (!block?.isSectionHeader) return false
  const head = String(block.text || '').split(/\r?\n/).map((line) => line.trim()).find(Boolean) || ''
  if (!head) return false
  if (isRecordEntryLine(head) || REGISTER_FIELD_LINE_RE.test(head)) return false
  if (!isTraceabilitySectionTitle(head)) return false
  const kind = traceabilityKindForBlock(block)
  return kind === targetKind
}

function numberedHeadingLevel(text) {
  const t = String(text || '').trim()
  const match = t.match(/^(\d+(?:\.\d+)*)(?:[.)\]:-]|\s)\s+\S/u)
  if (!match) return null
  const firstNumber = match[1].split('.')[0]
  const afterNumber = t.slice(match[0].indexOf(match[1]) + match[1].length).replace(/^[.)\]:\-\s]+/, '')
  if (/^0\d+$/u.test(firstNumber)) return null
  if (/^(?:DEV|CAPA|AUD|DEC)-[A-Z0-9]+-\d+/iu.test(afterNumber)) return null
  return match[1].split('.').filter(Boolean).length
}

function isTableOfContentsHeading(text) {
  return TABLE_OF_CONTENTS_HEADING_RE.test(normalizeForMatch(text))
}

function isLikelyTocEntry(text) {
  const t = String(text || '').trim()
  if (!t || t.length > 180) return false
  if (!/^\d+(?:\.\d+)*[.)]?\s+\S/u.test(t)) return false
  if (!/\s\d{1,4}\s*$/u.test(t)) return false
  if (/[.!?]\s*$/u.test(t)) return false
  const withoutPage = t.replace(/\s+\d{1,4}\s*$/u, '').trim()
  if (withoutPage === t || withoutPage.length < 4) return false
  return /^\d+(?:\.\d+)*[.)]?\s+\S/u.test(withoutPage)
}

function isSemanticSectionHeading(text) {
  const t = String(text || '').trim()
  if (!t || isRecordEntryLine(t) || REGISTER_FIELD_LINE_RE.test(t)) return false
  if (isTableOfContentsHeading(t) || isLikelyTocEntry(t)) return false
  if (isTraceabilitySectionTitle(t)) return true
  if (numberedHeadingLevel(t) != null && t.length < 160) return true
  const norm = normalizeForMatch(t)
  const wordCount = norm.split(/\s+/).filter(Boolean).length
  const knownSection = /\b(purpose|zweck|scope|geltungsbereich|responsibilities|verantwortlichkeiten|procedure|verfahren|process|definitions|definitionen|abbreviations|approval|records|training|deviation|deviations|capa|capas|review|history|references|appendix|annex)\b/i.test(norm)
  if (knownSection && wordCount <= 6 && t.length < 120 && !/[.!?]$/u.test(t)) return true
  const isAllCaps = /^[A-ZÄÖÜ0-9\s/&().:-]{4,120}$/u.test(t)
  return isAllCaps && !/[.!?]$/.test(t)
}

function isMajorSectionHeader(text, nodeType) {
  const t = String(text || '').trim()
  if (!t || isRecordEntryLine(t)) return false
  if (nodeType === 'heading' && t.length < 180 && !/[.!?]$/u.test(t)) return true
  return isSemanticSectionHeading(t)
}

function blockTraceabilityKind(block) {
  if (!block) return null
  if (block.sectionKind) return block.sectionKind
  if (block.embedded) return null
  return getTraceabilitySectionKind(block.text)
}

function isEmbeddedMarkerAtBoundary(text, index) {
  if (index === 0) return true
  const prev = text[index - 1]
  // Hyphen must not count as a boundary (avoids "CAPA" inside "CAPA-IT-001").
  return /[\s\n\r.:;!?]/u.test(prev)
}

function blockMatchesLabel(blockText, label, block = null) {
  const labelKind = getTraceabilitySectionKind(label)
  const blockKind = blockTraceabilityKind(block) || getTraceabilitySectionKind(blockText)

  if (labelKind && blockKind && labelKind !== blockKind) {
    return false
  }

  const sopRefs = extractSopRefs(label)
  const blockNorm = normalizeForMatch(blockText)
  if (sopRefs.length) {
    return sopRefs.some((ref) => blockNorm.includes(ref))
  }

  if (labelKind) {
    if (blockKind === labelKind) return true
    if (blockKind) return false
  }

  if (
    semanticSectionRootsMatch(label, blockText)
    && numberedSectionPrefixesMatch(label, blockText)
  ) {
    return true
  }

  const candidates = buildLabelCandidates(label)
  return candidates.some((c) => {
    if (c.startsWith('sop-')) return blockNorm.includes(c)
    const root = stripLeadingNumbering(c) || c
    if (root.length < 4) return false
    if (blockNorm === root || blockNorm.includes(root)) return true
    return semanticSectionRootsMatch(c, blockText) && numberedSectionPrefixesMatch(c, blockText)
  })
}

function scoreSectionStartBlock(block, label) {
  const exactMatch = blockMatchesLabel(block.text, label)
  const labelNorm = stripLeadingNumbering(label)
  const blockNorm = stripLeadingNumbering(block.text)
  const aliasSemanticMatch =
    !exactMatch
    && block.isSectionHeader
    && semanticSectionRootsMatch(label, block.text)
    && numberedSectionPrefixesMatch(label, block.text)

  const fuzzyHeadingMatch =
    !exactMatch
    && !aliasSemanticMatch
    && block.isSectionHeader
    && labelNorm.length >= 4
    && blockNorm.length >= 4
    && Math.abs(labelNorm.length - blockNorm.length) <= 2
    && editDistance(labelNorm, blockNorm) <= 2

  if (!exactMatch && !fuzzyHeadingMatch && !aliasSemanticMatch) return 0

  let score = 20
  if (aliasSemanticMatch) score += 70
  if (fuzzyHeadingMatch) score += 55
  const labelKind = getTraceabilitySectionKind(label)
  const blockKind = getTraceabilitySectionKind(block.text)

  if (labelKind && blockKind === labelKind) score += 80
  if (isMajorSectionHeader(block.text, block.node.type.name)) score += 25
  if (isRecordEntryLine(block.text)) score -= 40
  if (/\bzugehörig zu\s+SOP-/i.test(block.text)) score += 20

  const sopRefs = extractSopRefs(label)
  if (sopRefs.some((ref) => normalizeForMatch(block.text).includes(ref))) score += 60

  return score
}

function isExactSemanticSectionLabel(label, blockText) {
  const blockRoot = stripLeadingNumbering(blockText)
  if (!blockRoot) return false
  const candidates = new Set(aliasCandidates(label).map((item) => stripLeadingNumbering(item) || item))
  const singularBlockRoot = blockRoot.replace(/s$/i, '')
  for (const candidate of candidates) {
    const candidateRoot = stripLeadingNumbering(candidate) || candidate
    if (!candidateRoot) continue
    if (blockRoot === candidateRoot) return true
    if (candidateRoot.replace(/s$/i, '') === singularBlockRoot) return true
    if (blockRoot.startsWith(`${candidateRoot} `) || blockRoot.startsWith(`${candidateRoot} &`)) {
      const suffix = blockRoot.slice(candidateRoot.length).trim()
      if (/^(?:&|\/|:|-|\(|terms?\b)/i.test(suffix)) return true
    }
  }
  return false
}

function isTableCaptionText(text) {
  return /^\s*(?:table|tabelle)\s*\d+[A-Za-z0-9.\-:)]*\b/i.test(String(text || '').trim())
}

function wantsParentSectionForTableLabel(label) {
  const raw = String(label || '')
  return isTableCaptionText(raw) && /\bsections?\b/i.test(raw)
}

function findOwningHeadingIndex(blocks, startIdx) {
  for (let i = startIdx - 1; i >= 0; i -= 1) {
    const block = blocks[i]
    if (block.isTocBlock || !block.isSectionHeader) continue
    if (isTableCaptionText(block.text)) continue
    return i
  }
  return startIdx
}

function nextTableBlockIndex(blocks, startIdx) {
  for (let i = startIdx + 1; i < blocks.length; i += 1) {
    const block = blocks[i]
    if (block.isTocBlock) continue
    if (block.isTableBlock) return i
    if (block.isSectionHeader && !isTableCaptionText(block.text)) return -1
  }
  return -1
}

function buildTableCaptionTargetFromBlocks(doc, blocks, startIdx, label = '') {
  const tableIdx = nextTableBlockIndex(blocks, startIdx)
  if (tableIdx < 0) return buildSectionTargetFromBlocks(doc, blocks, startIdx, { label })
  const from = blocks[startIdx].start
  const to = sectionRangeEndForBlock(blocks, tableIdx)
  const text = doc.textBetween(from, to, '\n').trim()
  if (!text) return null
  return {
    from,
    to,
    text,
    isFullDoc: false,
    sectionName: blocks[startIdx].text.slice(0, 160),
    sectionType: 'Table',
    confidence: 0.9,
  }
}

function getHeadingLevel(node, text = '') {
  if (!node) return null
  const level = Number(node.attrs?.level)
  if (node.type?.name === 'heading') {
    return Number.isFinite(level) && level >= 1 && level <= 6 ? level : numberedHeadingLevel(text) || 2
  }
  return numberedHeadingLevel(text)
}

function splitEmbeddedSectionBlocks({ node, pos, text, headingLevel, isSectionHeader, isRecordEntry, isTocBlock = false, isTableBlock = false }) {
  const baseStart = pos + 1
  const matches = []
  EMBEDDED_TRACEABILITY_MARKER_RE.lastIndex = 0
  let match = EMBEDDED_TRACEABILITY_MARKER_RE.exec(text)
  while (match) {
    const index = match.index
    const marker = match[0] || ''
    const kind = getTraceabilitySectionKind(marker)
    const after = text.slice(index + marker.length, index + marker.length + 24)
    const hasVisualMarker = /[\u{1F534}\u{1F7E0}\u{1F7E1}\u{1F7E2}\u{1F535}\u{1F7E3}]/u.test(marker)
    const isStandaloneLineMarker = index === 0 || marker === marker.toUpperCase()
    const isRecordIdPrefix = /^[\s-]*[A-Z]{2,}-[A-Z0-9]+-\d+/i.test(after)
    const atBoundary = isEmbeddedMarkerAtBoundary(text, index)
    if (kind && atBoundary && !isRecordIdPrefix && (hasVisualMarker || isStandaloneLineMarker)) matches.push({ index, kind })
    match = EMBEDDED_TRACEABILITY_MARKER_RE.exec(text)
  }
  EMBEDDED_TRACEABILITY_MARKER_RE.lastIndex = 0

  if (!matches.length) {
    return [{
      node,
      pos,
      start: baseStart,
      end: pos + node.nodeSize - 1,
      text,
      headingLevel,
      isSectionHeader,
      isRecordEntry,
      isTocBlock,
      isTableBlock,
      embedded: false,
    }]
  }

  const segments = []
  if (matches[0].index > 0) {
    const prefixText = text.slice(0, matches[0].index).trim()
    if (prefixText) {
      segments.push({
        node,
        pos,
        start: baseStart,
        end: baseStart + matches[0].index,
        text: prefixText,
        headingLevel,
        isSectionHeader,
        isRecordEntry: isRecordEntryLine(prefixText),
        isTocBlock,
        isTableBlock,
        embedded: false,
      })
    }
  }

  matches.forEach((item, idx) => {
    const next = matches[idx + 1]
    const rawEnd = next ? next.index : text.length
    const segmentText = text.slice(item.index, rawEnd).trim()
    if (!segmentText) return
    segments.push({
      node,
      pos: baseStart + item.index,
      nodePos: pos,
      start: baseStart + item.index,
      end: baseStart + rawEnd,
      text: segmentText,
      headingLevel: null,
      isSectionHeader: true,
      isRecordEntry: false,
      isTocBlock,
      isTableBlock,
      embedded: true,
      markerIndex: item.index,
      sectionKind: item.kind,
    })
  })

  return segments
}

function isHeadingOnlyTargetText(text = '') {
  const trimmed = String(text || '').trim()
  if (!trimmed) return false
  const lines = trimmed.split(/\r?\n/).map((line) => line.trim()).filter(Boolean)
  if (lines.length !== 1) return false
  const line = lines[0]
  return isTraceabilitySectionTitle(line) || (isSemanticSectionHeading(line) && line.length < 200)
}

function extendSectionEndIdx(blocks, startIdx, { label = '', targetKind = null } = {}) {
  const startBlock = blocks[startIdx]
  const startHeadingLevel = startBlock.headingLevel
  let endIdx = startIdx

  for (let j = startIdx + 1; j < blocks.length; j += 1) {
    const block = blocks[j]
    const blockKind = blockTraceabilityKind(block)
    if (targetKind) {
      if (
        block.isSectionHeader
        && blockKind
        && blockKind !== targetKind
        && !blockMatchesLabel(block.text, label, block)
      ) {
        break
      }
      if (!blockKind && numberedHeadingLevel(block.text) != null && isSemanticSectionHeading(block.text)) break
      endIdx = j
      continue
    }
    if (block.embedded && block.isSectionHeader && block.sectionKind && block.sectionKind !== blockTraceabilityKind(startBlock)) {
      break
    }
    if (block.embedded && block.isSectionHeader && !blockMatchesLabel(block.text, label, block)) {
      break
    }
    if (isTableCaptionText(block.text)) {
      endIdx = j
      continue
    }
    if (
      startHeadingLevel != null
      && block.isSectionHeader
      && block.headingLevel == null
      && !blockMatchesLabel(block.text, label, block)
    ) {
      break
    }
    if (startHeadingLevel != null && block.headingLevel != null) {
      if (endIdx > startIdx || (block.headingLevel <= startHeadingLevel && !blockMatchesLabel(block.text, label, block))) {
        break
      }
    } else if (block.isSectionHeader && !blockMatchesLabel(block.text, label, block)) {
      break
    }
    endIdx = j
  }

  if (endIdx === startIdx && startBlock.headingLevel != null) {
    for (let j = startIdx + 1; j < blocks.length; j += 1) {
      const block = blocks[j]
      const blockKind = blockTraceabilityKind(block)
      if (
        block.isSectionHeader
        && !blockMatchesLabel(block.text, label, block)
        && blockKind
        && blockKind !== targetKind
      ) {
        break
      }
      endIdx = j
      if (!block.isSectionHeader || block.isRecordEntry) break
    }
  }

  return endIdx
}

function sectionRangeEndForBlock(blocks, endIdx) {
  const block = blocks[endIdx]
  if (!block) return 0
  if (!block.embedded) return block.pos + block.node.nodeSize
  if (Number.isFinite(block.nodePos)) {
    let lastSameNode = endIdx
    for (let k = endIdx + 1; k < blocks.length; k += 1) {
      if (blocks[k].nodePos === block.nodePos) lastSameNode = k
      else break
    }
    const tail = blocks[lastSameNode]
    return tail.embedded ? tail.end : tail.pos + tail.node.nodeSize
  }
  return block.end
}

function buildSectionTargetFromBlocks(doc, blocks, startIdx, { label = '', targetKind = null } = {}) {
  const startBlock = blocks[startIdx]
  let endIdx = extendSectionEndIdx(blocks, startIdx, { label, targetKind })

  const startEmbeddedAtBlockStart =
    startBlock.embedded
    && Number(startBlock.markerIndex || 0) === 0
    && Number.isFinite(startBlock.nodePos)
  const rangeStart = startEmbeddedAtBlockStart ? blocks[startIdx].nodePos : blocks[startIdx].start
  let rangeEnd = sectionRangeEndForBlock(blocks, endIdx)
  let text = doc.textBetween(rangeStart, rangeEnd, '\n').trim()
  if (!text || text.length < 3) return null

  const startTextNorm = normalizeForMatch(blocks[startIdx].text)
  let targetTextNorm = normalizeForMatch(text)
  if (startBlock.headingLevel != null && targetTextNorm === startTextNorm) {
    const forcedEnd = extendSectionEndIdx(blocks, startIdx, { label, targetKind })
    if (forcedEnd <= startIdx) return null
    endIdx = forcedEnd
    rangeEnd = sectionRangeEndForBlock(blocks, endIdx)
    text = doc.textBetween(rangeStart, rangeEnd, '\n').trim()
    targetTextNorm = normalizeForMatch(text)
    if (!text || targetTextNorm === startTextNorm) return null
  }

  return {
    from: rangeStart,
    to: rangeEnd,
    text,
    isFullDoc: false,
    sectionName: blocks[startIdx].text.slice(0, 160),
    sectionType: getTraceabilitySectionKind(blocks[startIdx].text) ? 'Section' : 'Heading',
    confidence: targetKind ? 0.92 : 0.82,
  }
}

function findSectionByTraceabilityKind(doc, kindOrLabel) {
  const normalized = normalizeForMatch(kindOrLabel)
  const canonicalKinds = new Set(['capas', 'deviations', 'audit', 'decisions'])
  const targetKind =
    getTraceabilitySectionKind(kindOrLabel)
    || (canonicalKinds.has(normalized) ? normalized : null)
  if (!targetKind) return null

  const blocks = collectBlocks(doc)
  if (!blocks.length) return null

  let startIdx = -1
  let bestScore = 0
  for (let i = 0; i < blocks.length; i += 1) {
    const block = blocks[i]
    if (block.isTocBlock) continue
    if (!isRealTraceabilitySectionStart(block, targetKind)) continue
    let score = scoreSectionStartBlock(block, kindOrLabel) + 90
    if (/\bzugehörig zu\s+SOP-/i.test(block.text)) score += 30
    if (block.headingLevel != null) score += 15
    if (score > bestScore) {
      bestScore = score
      startIdx = i
    }
  }
  if (startIdx < 0 || bestScore < 50) return null
  return buildSectionTargetFromBlocks(doc, blocks, startIdx, { label: kindOrLabel, targetKind })
}

function collectBlocks(doc) {
  const rawBlocks = []
  doc.descendants((node, pos) => {
    const nodeName = node.type?.name
    if (nodeName === 'table') {
      const text = node.textContent.trim()
      if (!text) return false
      rawBlocks.push({
        node,
        pos,
        start: pos + 1,
        end: pos + node.nodeSize - 1,
        text,
        headingLevel: null,
        isSectionHeader: false,
        isRecordEntry: false,
        isTocBlock: false,
        isTableBlock: true,
      })
      return false
    }
    if (nodeName === 'tableRow' || nodeName === 'tableCell' || nodeName === 'tableHeader') return false
    if (!node.isBlock) return true
    const text = node.textContent.trim()
    if (!text) return true
    const headingLevel = getHeadingLevel(node, text)
    const block = {
      node,
      pos,
      start: pos + 1,
      end: pos + node.nodeSize - 1,
      text,
      headingLevel,
      isSectionHeader: isMajorSectionHeader(text, node.type.name),
      isRecordEntry: isRecordEntryLine(text),
      isTocBlock: false,
    }
    rawBlocks.push(block)
    return true
  })

  const blocks = []
  let insideToc = false
  for (const block of rawBlocks) {
    const isTocHeading = isTableOfContentsHeading(block.text)
    const isTocEntry = isLikelyTocEntry(block.text)

    if (isTocHeading) {
      insideToc = true
      blocks.push(...splitEmbeddedSectionBlocks({
        ...block,
        headingLevel: null,
        isSectionHeader: false,
        isTocBlock: true,
      }))
      continue
    }

    if (insideToc && isTocEntry) {
      blocks.push(...splitEmbeddedSectionBlocks({
        ...block,
        headingLevel: null,
        isSectionHeader: false,
        isTocBlock: true,
      }))
      continue
    }

    if (insideToc && !isTocEntry) insideToc = false
    blocks.push(...splitEmbeddedSectionBlocks(block))
  }

  return blocks
}

function findSectionByLabelInDoc(doc, label) {
  const blocks = collectBlocks(doc)
  if (!blocks.length) return null

  let startIdx = -1
  let bestScore = 0
  const scored = []
  for (let i = 0; i < blocks.length; i += 1) {
    const block = blocks[i]
    if (block.isTocBlock) continue
    if (!block.isSectionHeader) continue
    let score = scoreSectionStartBlock(block, label)
    if (block.headingLevel != null) score += 20
    if (block.isSectionHeader && block.headingLevel == null) score += 10
    if (score >= 35) scored.push({ index: i, score, text: block.text })
    if (score > bestScore) {
      bestScore = score
      startIdx = i
    }
  }
  const closeMatches = scored
    .filter((item) => item.score >= bestScore - 8)
    .filter((item, index, items) => items.findIndex((other) => normalizeForMatch(other.text) === normalizeForMatch(item.text)) === index)
  if (closeMatches.length > 1) {
    const exactSemanticMatches = closeMatches.filter((item) => isExactSemanticSectionLabel(label, item.text))
    if (exactSemanticMatches.length === 1) {
      startIdx = exactSemanticMatches[0].index
    } else {
      const choices = closeMatches.slice(0, 5).map((item) => `"${item.text.slice(0, 80)}"`).join(', ')
      throw new Error(`Multiple matching sections found for "${label}": ${choices}. Use the exact heading.`)
    }
  }
  if (startIdx >= 0 && bestScore >= 35) {
    const startBlock = blocks[startIdx]
    if (isTableCaptionText(startBlock.text)) {
      if (wantsParentSectionForTableLabel(label)) {
        const ownerIdx = findOwningHeadingIndex(blocks, startIdx)
        return buildSectionTargetFromBlocks(doc, blocks, ownerIdx, { label: blocks[ownerIdx]?.text || label })
      }
      return buildTableCaptionTargetFromBlocks(doc, blocks, startIdx, label)
    }
    const targetKind = getTraceabilitySectionKind(label) || getTraceabilitySectionKind(startBlock.text)
    return buildSectionTargetFromBlocks(doc, blocks, startIdx, { label, targetKind })
  }

  const numberedPrefix = leadingSectionNumber(label)
  if (numberedPrefix) {
    for (let i = 0; i < blocks.length; i += 1) {
      const block = blocks[i]
      if (block.isTocBlock) continue
      if (!block.isSectionHeader) continue
      if (leadingSectionNumber(block.text) !== numberedPrefix) continue
      if (!semanticSectionRootsMatch(label, block.text)) continue
      const targetKind = getTraceabilitySectionKind(label) || getTraceabilitySectionKind(block.text)
      return buildSectionTargetFromBlocks(doc, blocks, i, { label, targetKind })
    }
  }

  return null
}

function expandSelectionToSection(doc, { from, to, text }) {
  const hints = []
  const traceabilityKind = getTraceabilitySectionKind(text)
  if (traceabilityKind) hints.push(traceabilityKind)
  if (text) hints.push(text)
  const recordMatch = String(text || '').match(RECORD_ENTRY_RE)
  if (recordMatch) {
    const prefix = recordMatch[0].trim().slice(0, 4).toUpperCase()
    if (prefix === 'CAPA') hints.push('capas')
    if (prefix === 'DEV-') hints.push('deviations')
    if (prefix === 'AUD-') hints.push('audit')
    if (prefix === 'DEC-') hints.push('decisions')
  }
  const uniqueHints = [...new Set(hints.filter(Boolean))]
  for (const hint of uniqueHints) {
    const expanded =
      findSectionByLabelInDoc(doc, hint)
      || findSectionByTraceabilityKind(doc, hint)
    if (expanded?.text && expanded.text.length > String(text || '').trim().length) {
      return expanded
    }
  }
  return null
}

function resolveSelectionTarget(editor, selection, { preferFullSection = false } = {}) {
  if (!selection || selection.empty) return null
  const doc = editor.state.doc
  const from = selection.from
  const to = selection.to
  const text = doc.textBetween(from, to, '\n').trim()
  if (!text) return null

  const shouldExpand = preferFullSection

  const withTableAwareness = (target) => {
    if (!target) return target
    let containsTable = false
    doc.descendants?.((node, pos) => {
      if (node.type?.name !== 'table') return !containsTable
      const tableFrom = pos
      const tableTo = pos + (node.nodeSize || 0)
      if (target.from < tableTo && tableFrom < target.to) containsTable = true
      return !containsTable
    })
    return {
      ...target,
      contains_table: containsTable,
      sectionType: containsTable ? 'Table Section' : target.sectionType,
      resolved_target_type: containsTable ? 'table_section' : (target.resolved_target_type || 'selection'),
    }
  }

  if (shouldExpand) {
    const expanded = expandSelectionToSection(doc, { from, to, text })
    if (expanded) return withTableAwareness(expanded)
  }

  return withTableAwareness({
    from,
    to,
    text,
    isFullDoc: false,
    sectionName: inferSectionName(editor, from),
    sectionType: /\s/.test(text) ? 'Paragraph' : 'Word',
  })
}

function resolveFullDocument(doc) {
  const size = doc.content.size
  return {
    from: 0,
    to: size,
    text: doc.textBetween(0, size, '\n').trim(),
    isFullDoc: true,
    sectionName: 'Full SOP',
    sectionType: 'Full Document',
  }
}

function inferSectionName(editor, from) {
  try {
    const $pos = editor.state.doc.resolve(from)
    for (let d = $pos.depth; d >= 0; d -= 1) {
      const node = $pos.node(d)
      if (node.type.name === 'heading') return node.textContent
    }
  } catch {
    // ignore
  }
  return 'Selected text'
}

function findSectionByHints(doc, hints = []) {
  const labels = [...new Set(hints.map((h) => String(h || '').trim()).filter(Boolean))]
  for (const label of labels) {
    const section = findSectionByLabelInDoc(doc, label) || findSectionByTraceabilityKind(doc, label)
    if (section) return section
  }
  return null
}

function ensureFullSectionTarget(doc, target, hints = []) {
  if (!target?.text || !isHeadingOnlyTargetText(target.text)) return target
  const expanded = findSectionByHints(doc, hints)
  if (expanded?.text && expanded.text.length > target.text.length) return expanded
  const kind = getTraceabilitySectionKind(target.text)
  if (kind) {
    const byKind = findSectionByTraceabilityKind(doc, kind)
    if (byKind?.text && byKind.text.length > target.text.length) return byKind
  }
  return target
}

function ordinalSectionIndex(promptText) {
  const text = normalizeForMatch(promptText)
  const ordinalMap = new Map([
    ['first', 0],
    ['1st', 0],
    ['one', 0],
    ['second', 1],
    ['2nd', 1],
    ['two', 1],
    ['third', 2],
    ['3rd', 2],
    ['three', 2],
    ['fourth', 3],
    ['4th', 3],
    ['fifth', 4],
    ['5th', 4],
  ])

  for (const [token, index] of ordinalMap.entries()) {
    if (new RegExp(`\\b(?:this\\s+|the\\s+)?${token}\\s+section\\b`, 'i').test(text)) {
      return index
    }
  }
  if (/\blast\s+section\b/i.test(text)) return -1
  return null
}

function resolveNumberedSemanticSection(doc, label) {
  const wantNum = leadingSectionNumber(label)
  if (!wantNum) return null
  const ranges = collectSectionRanges(doc)
  for (const section of ranges) {
    if (leadingSectionNumber(section.sectionName) !== wantNum) continue
    if (!semanticSectionRootsMatch(label, section.sectionName)) continue
    return section
  }
  const blocks = collectBlocks(doc)
  for (let i = 0; i < blocks.length; i += 1) {
    const block = blocks[i]
    if (block.isTocBlock) continue
    if (!block.isSectionHeader) continue
    if (leadingSectionNumber(block.text) !== wantNum) continue
    if (!semanticSectionRootsMatch(label, block.text)) continue
    const targetKind = getTraceabilitySectionKind(label) || getTraceabilitySectionKind(block.text)
    return buildSectionTargetFromBlocks(doc, blocks, i, { label, targetKind })
  }
  return null
}

export function collectSectionRanges(doc) {
  const blocks = collectBlocks(doc)
  const starts = blocks
    .map((block, index) => ({ ...block, index }))
    .filter((block) => block.isSectionHeader && !block.isTocBlock)

  return starts.map((start, position) => {
    const next = starts[position + 1]
    const endBlock = next ? blocks[Math.max(start.index, next.index - 1)] : blocks[blocks.length - 1]
    const from = start.start
    const to = endBlock.embedded ? endBlock.end : endBlock.pos + endBlock.node.nodeSize
    const text = doc.textBetween(from, to, '\n').trim()
    const nodeAttrs = start.node?.attrs || {}
    const nodeId = String(nodeAttrs.id || nodeAttrs.blockId || nodeAttrs.block_id || nodeAttrs['data-id'] || '').trim()
    const embeddedSuffix = start.embedded
      ? `::section:${start.sectionKind || 'embedded'}:${Number(start.markerIndex) || position + 1}`
      : ''
    return {
      id: nodeId ? `${nodeId}${embeddedSuffix}` : null,
      from,
      to,
      text,
      isFullDoc: false,
      sectionName: start.text.slice(0, 160),
      sectionType: getTraceabilitySectionKind(start.text) ? 'Section' : 'Heading',
    }
  }).filter((section) => section.text)
}

export function buildEditorSectionIndex(editor) {
  const doc = editor?.state?.doc || editor
  if (!doc) return []
  const headingIdsByPos = new Map()
  doc.descendants?.((node, pos) => {
    if (node.type?.name === 'heading') {
      const attrs = node.attrs || {}
      const id = String(attrs.id || attrs.blockId || attrs.block_id || attrs['data-id'] || '').trim()
      if (id) headingIdsByPos.set(pos, id)
    }
    return true
  })

  const headingIdNear = (from) => {
    if (headingIdsByPos.has(from)) return headingIdsByPos.get(from)
    for (const [pos, id] of headingIdsByPos.entries()) {
      if (Math.abs(Number(pos) - Number(from)) <= 1) return id
    }
    return null
  }

  return collectSectionRanges(doc).map((section, index) => ({
    id: section.id || headingIdNear(section.from) || `section-${index + 1}`,
    index,
    title: section.sectionName,
    heading: section.sectionName,
    sectionName: section.sectionName,
    sectionType: section.sectionType,
    from: section.from,
    to: section.to,
    text: section.text,
    confidence: 1,
  }))
}

export function buildEditorTableIndex(editor) {
  const doc = editor?.state?.doc || editor
  if (!doc) return []
  const sections = collectSectionRanges(doc)
  const tables = []

  doc.descendants((node, pos) => {
    if (node.type?.name !== 'table') return true
    const from = pos
    const to = pos + node.nodeSize
    const text = node.textContent?.trim?.() || ''
    const owningSection = [...sections]
      .reverse()
      .find((section) => section.from <= from && section.to >= to)
      || [...sections].reverse().find((section) => section.from <= from)
      || null
    let rowCount = 0
    let columnCount = 0
    node.forEach?.((row) => {
      rowCount += 1
      let cells = 0
      row.forEach?.(() => { cells += 1 })
      columnCount = Math.max(columnCount, cells)
    })
    const previousSection = [...sections].reverse().find((section) => section.from < from)
    const caption = previousSection && previousSection.to <= from + 2 && isTableCaptionText(previousSection.sectionName)
      ? previousSection.sectionName
      : ''
    tables.push({
      id: String(node.attrs?.id || node.attrs?.blockId || node.attrs?.block_id || node.attrs?.['data-id'] || '').trim() || `table-${tables.length + 1}`,
      index: tables.length,
      caption: caption || `Table ${tables.length + 1}`,
      owningSection: owningSection?.sectionName || '',
      from,
      to,
      text,
      rowCount,
      columnCount,
    })
    return false
  })

  return tables
}

function resolveOrdinalSectionTarget(doc, promptText) {
  const index = ordinalSectionIndex(promptText)
  if (index == null) return null

  const sections = collectSectionRanges(doc)
  if (!sections.length) return null
  return index === -1 ? sections[sections.length - 1] : sections[index] || null
}

function resolveRecordTarget(doc, recordId) {
  const id = String(recordId || '').trim().toUpperCase()
  if (!id) return null
  const blocks = collectBlocks(doc)
  const startIdx = blocks.findIndex((block) => {
    const t = String(block.text || '').trim()
    return t.startsWith(id) || t.includes(id)
  })
  if (startIdx < 0) return null

  let endIdx = startIdx
  for (let j = startIdx + 1; j < blocks.length; j += 1) {
    const block = blocks[j]
    if (isRecordEntryLine(block.text) && !String(block.text || '').toUpperCase().includes(id)) {
      break
    }
    if (isMajorSectionHeader(block.text) && !String(block.text || '').toUpperCase().includes(id)) {
      break
    }
    endIdx = j
    if (isRecordEntryLine(block.text)) break
  }

  const from = blocks[startIdx].start
  const to = blocks[endIdx].embedded
    ? blocks[endIdx].end
    : blocks[endIdx].pos + blocks[endIdx].node.nodeSize
  const text = doc.textBetween(from, to, '\n').trim()
  if (!text) return null
  return {
    from,
    to,
    text,
    isFullDoc: false,
    sectionName: blocks[startIdx].text.slice(0, 160),
    sectionType: 'Record',
  }
}

function resolveSubheadingTarget(doc, subLabel) {
  const label = String(subLabel || '').trim()
  if (!label) return null
  const blocks = collectBlocks(doc)
  const labelRe = new RegExp(`^\\s*${label.replace(/\./g, '\\.')}(?:[.)\]:-]|\\s+)`, 'i')
  const normalizedLabel = normalizeForMatch(label)
  let startIdx = -1
  for (let i = 0; i < blocks.length; i += 1) {
    if (blocks[i].isTocBlock) continue
    if (labelRe.test(blocks[i].text) || (normalizedLabel && normalizeForMatch(blocks[i].text).startsWith(normalizedLabel))) {
      startIdx = i
      break
    }
  }
  if (startIdx < 0) return null

  const startLevel = numberedHeadingLevel(blocks[startIdx].text) || 99
  let endIdx = startIdx
  for (let j = startIdx + 1; j < blocks.length; j += 1) {
    const block = blocks[j]
    if (block.isTocBlock) continue
    const level = numberedHeadingLevel(block.text)
    if (level != null && level <= startLevel) break
    if (block.isSectionHeader && isSemanticSectionHeading(block.text)) break
    endIdx = j
  }

  const from = blocks[startIdx].start
  const to = blocks[endIdx].embedded
    ? blocks[endIdx].end
    : blocks[endIdx].pos + blocks[endIdx].node.nodeSize
  const text = doc.textBetween(from, to, '\n').trim()
  if (!text) return null
  return {
    from,
    to,
    text,
    isFullDoc: false,
    sectionName: blocks[startIdx].text.slice(0, 160),
    sectionType: 'Sub-section',
  }
}

function resolveLineTarget(doc, promptText) {
  const match = String(promptText || '').match(
    /\b(?:line|zeile)\s+(\d{1,4})\b|\bin\s+zeile\s+(\d{1,4})\b/i,
  )
  if (!match) return null

  const lineNumber = Number(match[1] || match[2])
  if (!Number.isInteger(lineNumber) || lineNumber <= 0) return null

  const blocks = collectBlocks(doc)
  const block = blocks[lineNumber - 1]
  if (!block?.text) return null

  return {
    from: block.start,
    to: block.end,
    text: block.text,
    isFullDoc: false,
    sectionName: `Line ${lineNumber}`,
    sectionType: 'Line',
  }
}

/**
 * Resolve target range inside the live editor document.
 *
 * @param {object} [options]
 * @param {string} [options.prompt] — enriched user instruction
 * @param {object} [options.selection] — TipTap selection snapshot
 * @param {string} [options.sectionHint] — classifier section name (Purpose, Scope, …)
 * @param {string} [options.targetScope] — selection | section | full_document | linked_context
 */
export function resolveTargetInEditor(editor, {
  prompt = '',
  userPrompt = '',
  selection,
  sectionHint = '',
  targetScope = '',
  targetType = '',
  targetLabel = '',
  owningSection = '',
  lineNumber = null,
  recordId = '',
  preferFullSection: preferFullSectionOpt = false,
} = {}) {
  if (!editor || editor.isDestroyed) return null

  const doc = editor.state.doc
  const promptText = String(prompt || '').trim()
  const userText = String(userPrompt || '').trim() || stripAssistantConstraints(promptText)
  const scope = String(targetScope || '').trim().toLowerCase()
  const semanticTargetType = String(targetType || '').trim().toLowerCase()
  const semanticTargetLabel = String(targetLabel || '').trim()
  const semanticOwningSection = String(owningSection || '').trim()
  const hint = String(sectionHint || '').trim()
  const backendScoped = Boolean(scope) || Boolean(hint) || Boolean(recordId) || Number.isFinite(lineNumber)
  const isGapRequest = !backendScoped && (GAP_CHECK_VERB.test(promptText) || GAP_CHECK_VERB.test(userText))
  const isRewriteImprove = !backendScoped && (REWRITE_IMPROVE_VERB.test(promptText) || REWRITE_IMPROVE_VERB.test(userText))
  const isSummarizeRequest = !backendScoped && (SUMMARIZE_VERB.test(promptText) || SUMMARIZE_VERB.test(userText))
  const sectionIntent = !backendScoped && (wantsSectionScopeIntent(promptText) || wantsSectionScopeIntent(userText))
  const userLabels = extractLabelsFromPrompt(userText)
  const userTraceabilityKind = getTraceabilitySectionKind(userText)
  let effectiveHint = String(semanticTargetLabel || hint || '').trim()
  const hintKind = getTraceabilitySectionKind(effectiveHint)
  if (userTraceabilityKind && hintKind && userTraceabilityKind !== hintKind) {
    effectiveHint = ''
  }
  const preferFullSection =
    Boolean(preferFullSectionOpt)
    || scope === 'section'
    || semanticTargetType === 'section'
    || semanticTargetType === 'table_section'
    || Boolean(effectiveHint || userTraceabilityKind || userLabels.length)
    || sectionIntent

  if (scope === 'full_document') {
    return resolveFullDocument(doc)
  }

  if (semanticTargetType === 'full_document') {
    return resolveFullDocument(doc)
  }

  if (!backendScoped && (wantsFullSopIntent(promptText) || wantsFullSopIntent(userText))) {
    return resolveFullDocument(doc)
  }

  if (!backendScoped) {
    for (const pattern of FULL_SOP_PATTERNS) {
      if (pattern.test(promptText) || pattern.test(userText)) {
        return resolveFullDocument(doc)
      }
    }
  }

  const structuredHints = [
    semanticTargetType === 'table_section' ? semanticOwningSection : '',
    effectiveHint,
    recordId,
    ...userLabels,
    ...(effectiveHint ? [] : extractLabelsFromPrompt(stripAssistantConstraints(promptText))),
  ].filter(Boolean)

  const explicitNamedTarget =
    ['section', 'table', 'table_section'].includes(semanticTargetType)
    || Boolean(effectiveHint || userTraceabilityKind || userLabels.length || hint || recordId)
  const explicitSelectionReference = /\b(?:selected\s+(?:text|word|paragraph|section|heading|line|content)|selection|highlighted|mark(?:ed)?\s+(?:text|section|heading))\b/i.test(`${userText}\n${promptText}`)

  if (
    scope === 'selection'
    && selection
    && !selection.empty
    && explicitSelectionReference
    && !['section', 'table', 'table_section'].includes(semanticTargetType)
  ) {
    return resolveSelectionTarget(editor, selection, { preferFullSection: Boolean(preferFullSectionOpt) })
  }

  if (scope === 'selection' && selection && !selection.empty && !explicitNamedTarget) {
    return resolveSelectionTarget(editor, selection, { preferFullSection: Boolean(preferFullSectionOpt) })
  }

  if (semanticTargetType === 'selection' && selection && !selection.empty) {
    return resolveSelectionTarget(editor, selection, { preferFullSection: false })
  }

  if (semanticTargetType === 'section' && semanticTargetLabel) {
    const sectionTarget = findSectionByHints(doc, [semanticTargetLabel, semanticOwningSection].filter(Boolean))
    if (sectionTarget) {
      const fullSectionTarget = ensureFullSectionTarget(
        doc,
        sectionTarget,
        [semanticTargetLabel, semanticOwningSection].filter(Boolean),
      ) || sectionTarget
      return {
        ...fullSectionTarget,
        resolved_target_type: 'section',
        match_reason: 'llm_section_label',
      }
    }
  }

  if (semanticTargetType === 'table' && semanticTargetLabel) {
    const tableTarget = findSectionByLabelInDoc(doc, semanticTargetLabel)
    if (tableTarget) return {
      ...tableTarget,
      resolved_target_type: 'table',
      match_reason: 'llm_table_label',
    }
  }

  if (semanticTargetType === 'table_section' && (semanticOwningSection || semanticTargetLabel)) {
    const section = findSectionByHints(doc, [semanticOwningSection, semanticTargetLabel].filter(Boolean))
    if (section) return {
      ...section,
      resolved_target_type: 'table_section',
      match_reason: 'llm_table_section',
    }
  }

  const trySectionByHints = () => {
    if (userTraceabilityKind) {
      const byUserKind = findSectionByTraceabilityKind(doc, userTraceabilityKind)
      if (byUserKind) return byUserKind
    }
    const byHint = structuredHints.length ? findSectionByHints(doc, structuredHints) : null
    if (byHint) return byHint
    const numberedSemantic = resolveNumberedSemanticSection(doc, effectiveHint || userText)
    if (numberedSemantic) return numberedSemantic
    return null
  }

  const finalizeSectionTarget = (target) => {
    if (!target) return target
    if (!preferFullSection) return target
    return ensureFullSectionTarget(doc, target, structuredHints)
  }

  const recordHint = recordId || structuredHints.find((h) => /^(?:DEV|CAPA|AUD|DEC)-/i.test(String(h)))
  const recordTarget = resolveRecordTarget(doc, recordHint)
  if (recordTarget) return finalizeSectionTarget(recordTarget)

  const subMatch = userText.match(/\b(?:section\s+)?(\d+(?:\.\d+)+)(?=$|[.)\]:-]|\s)/i)
    || promptText.match(/\b(?:section\s+)?(\d+(?:\.\d+)+)(?=$|[.)\]:-]|\s)/i)
  if (subMatch) {
    const subTarget = resolveSubheadingTarget(doc, subMatch[1])
    if (subTarget) return finalizeSectionTarget(subTarget)
  }

  const lineTarget =
    (Number.isFinite(lineNumber) && lineNumber > 0
      ? resolveLineTarget(doc, `line ${lineNumber}`)
      : null)
    || resolveLineTarget(doc, promptText)
    || resolveLineTarget(doc, userText)
  if (lineTarget) return lineTarget

  const ordinalSectionTarget = resolveOrdinalSectionTarget(doc, promptText)
    || resolveOrdinalSectionTarget(doc, userText)
  if (ordinalSectionTarget) return finalizeSectionTarget(ordinalSectionTarget)

  const resolvedByHints = trySectionByHints()

  if (scope === 'section' || hint || structuredHints.length) {
    const section = finalizeSectionTarget(resolvedByHints)
    if (section) return section
  }

  if (isGapRequest || isRewriteImprove || isSummarizeRequest) {
    const section = finalizeSectionTarget(resolvedByHints)
    if (section) return section
  }

  if (userTraceabilityKind && !resolvedByHints) {
    throw new Error(
      `Could not find the ${userTraceabilityKind.toUpperCase()} section heading in the open SOP. `
      + 'Use the exact block title (e.g. CAPAs, DEVIATIONS) or select that section in the editor.',
    )
  }

  const skipSelectionFallback =
    Boolean(userTraceabilityKind)
    || (
      scope === 'section'
      && Boolean(effectiveHint || userLabels.length)
      && (backendScoped || sectionIntent || isRewriteImprove || isGapRequest)
    )

  if (selection && !selection.empty && !skipSelectionFallback && (!explicitNamedTarget || explicitSelectionReference)) {
    const selected = finalizeSectionTarget(
      resolveSelectionTarget(editor, selection, { preferFullSection }),
    )
    if (selected) return selected
  }

  if (isGapRequest) {
    if (wantsFullSopIntent(promptText) || wantsFullSopIntent(userText)) return resolveFullDocument(doc)
    for (const pattern of FULL_SOP_GAP_PATTERNS) {
      if (pattern.test(promptText) || pattern.test(userText)) return resolveFullDocument(doc)
    }
    const labels = extractLabelsFromPrompt(promptText)
    if (labels.length === 0 && /\b(?:this\s+)?sop\b/i.test(userText || promptText) && !/\bsection\b/i.test(userText || promptText)) {
      return resolveFullDocument(doc)
    }
    const section = finalizeSectionTarget(findSectionByHints(doc, labels))
    if (section) return section
  }

  if (isRewriteImprove) {
    const section = finalizeSectionTarget(findSectionByHints(doc, [
      ...extractLabelsFromPrompt(promptText),
      ...extractLabelsFromPrompt(userText),
    ]))
    if (section) return section
  }

  if (scope === 'section' || hint || isGapRequest || isRewriteImprove || isSummarizeRequest || sectionIntent) {
    const labels = structuredHints.length
      ? structuredHints
      : [...extractLabelsFromPrompt(promptText), ...extractLabelsFromPrompt(userText)]
    const missing = hint || labels[0] || 'that section'
    throw new Error(
      `Could not find that section ("${missing}") in the open SOP. Use the exact section heading (e.g. Purpose, Scope, Procedure) or select the section in the editor.`,
    )
  }

  return null
}
