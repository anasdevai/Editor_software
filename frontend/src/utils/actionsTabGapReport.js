/**
 * Parse gap_check analysis text into sidebar-friendly sections.
 */

const stripHtml = (value) =>
  String(value || '')
    .replace(/<[^>]+>/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()

const TITLE_ALIASES = {
  summary: 'Summary',
  zusammenfassung: 'Summary',
  base: 'RAG/NLP Basis',
  basis: 'RAG/NLP Basis',
  'rag/nlp basics': 'RAG/NLP Basis',
  'rag/nlp basis': 'RAG/NLP Basis',
  'rag/nlp-grundlage': 'RAG/NLP Basis',
  details: 'Details',
  status: 'Status',
  sources: 'Sources',
  references: 'References',
  'cross-refs': 'Cross-Refs',
  'cross refs': 'Cross-Refs',
  'identified gaps': 'Identified Gaps',
  'identified gap': 'Identified Gaps',
  'festgestellte lucken': 'Identified Gaps',
  'festgestellte lücken': 'Identified Gaps',
  'recommended corrections': 'Recommended Fixes',
  'recommended fixes': 'Recommended Fixes',
  'empfohlene korrekturen': 'Recommended Fixes',
  'suggested sop amendment text': 'Suggested SOP Text',
  'suggested sop text': 'Suggested SOP Text',
  'vorgeschlagener sop-erganzungstext': 'Suggested SOP Text',
  'vorgeschlagener sop-ergänzungstext': 'Suggested SOP Text',
  'remaining assumptions': 'Residual Assumptions',
  'residual assumptions': 'Residual Assumptions',
  'verbleibende annahmen': 'Residual Assumptions',
}

const normalizeHeadingKey = (value) =>
  String(value || '')
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/\s+/g, ' ')
    .trim()
    .toLowerCase()

const HEADING_RE = new RegExp(
  `^/?\\s*(${Object.keys(TITLE_ALIASES)
    .map((key) => key.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
    .join('|')})\\s*:?\\s*(.*)$`,
  'i',
)

function splitAnalysisBlocks(analysis) {
  const raw = String(analysis || '').replace(/\r\n/g, '\n').trim()
  if (!raw) return []

  const lines = raw.split('\n')
  const blocks = []
  let currentTitle = 'Summary'
  let currentLines = []

  const flush = () => {
    const body = currentLines.join('\n').trim()
    if (body) blocks.push({ title: currentTitle, body })
    currentLines = []
  }

  for (const line of lines) {
    const trimmed = line.trim()
    const headingMatch = trimmed.match(HEADING_RE)
    if (headingMatch) {
      flush()
      currentTitle = TITLE_ALIASES[normalizeHeadingKey(headingMatch[1])] || headingMatch[1]
      if (headingMatch[2]) currentLines.push(headingMatch[2])
    } else {
      currentLines.push(line)
    }
  }
  flush()
  return blocks.length ? blocks : [{ title: 'Gap analysis', body: raw }]
}

function extractGapItems(body) {
  const items = []
  const lines = String(body || '').split('\n')
  let current = null

  const pushCurrent = () => {
    if (current?.issue) items.push(current)
    current = null
  }

  for (const line of lines) {
    const trimmed = line.trim()
    if (!trimmed) continue
    const numbered = trimmed.match(/^\d+[.)]\s*(?:Gap:)?\s*(.+)$/i)
    if (numbered) {
      pushCurrent()
      current = { issue: numbered[1], explanation: '', recommendation: '' }
      continue
    }
    if (/^gap\s*:/i.test(trimmed)) {
      pushCurrent()
      current = { issue: trimmed.replace(/^gap\s*:\s*/i, ''), explanation: '', recommendation: '' }
      continue
    }
    if (/^evidence\s*:/i.test(trimmed) && current) {
      current.explanation = `${current.explanation}\n${trimmed}`.trim()
      continue
    }
    if (/^(risk|impact|recommended fix|empfehlung)\s*:/i.test(trimmed) && current) {
      current.recommendation = `${current.recommendation}\n${trimmed}`.trim()
      continue
    }
    if (current) {
      current.explanation = `${current.explanation}\n${trimmed}`.trim()
    } else {
      items.push({ issue: trimmed, explanation: '', recommendation: '' })
    }
  }
  pushCurrent()
  return items
}

export function buildGapCheckSidebarReport(result) {
  const structured = result?.structured_data || {}
  const analysis =
    structured.analysis ||
    stripHtml(result?.suggested_text) ||
    stripHtml(result?.explanation) ||
    ''

  const sections = []
  const explanation = stripHtml(result?.explanation || '')
  if (explanation) {
    sections.push({ id: 'intro', title: 'Gap check', body: explanation })
  }

  const gaps = Array.isArray(structured.gaps) ? structured.gaps : []
  if (gaps.length > 0) {
    sections.push({
      id: 'gaps-structured',
      title: 'Identified gaps',
      gapItems: gaps,
    })
  } else {
    for (const block of splitAnalysisBlocks(analysis)) {
      const titleLower = block.title.toLowerCase()
      if (titleLower.includes('gap') || titleLower.includes('lücke') || titleLower.includes('lucke')) {
        const gapItems = extractGapItems(block.body)
        if (gapItems.length) {
          sections.push({ id: `gaps-${sections.length}`, title: block.title, gapItems })
        } else {
          sections.push({ id: `block-${sections.length}`, title: block.title, body: block.body })
        }
      } else {
        sections.push({ id: `block-${sections.length}`, title: block.title, body: block.body })
      }
    }
  }

  if (!sections.length && analysis) {
    sections.push({ id: 'full', title: 'Full report', body: analysis })
  }

  return { sections, analysisHtml: result?.suggested_text || '', analysisPlain: analysis }
}
