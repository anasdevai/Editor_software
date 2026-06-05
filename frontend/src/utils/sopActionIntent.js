/**
 * Shared intent detection for Actions tab / editor target resolution.
 */

const FULL_SOP_INTENT_RE =
  /\b(?:rewrite|re-?write|improve|revise|umschreiben|verbessern?)\b[\s\S]*\b(?:(?:this\s+)?full\s+sop|(?:entire|whole|complete|gesamte?|komplette?)\s+sop|sop\s+(?:komplett|vollständig|gesamt)|standard\s+operating\s+procedures?|(?:the\s+)?procedure)\b|\b(?:(?:this\s+)?full\s+sop|(?:entire|whole|complete)\s+sop|standard\s+operating\s+procedures?)\b[\s\S]*\b(?:rewrite|improve|revise|umschreiben|verbessern?)\b/i

const LABEL_STOPWORDS = new Set([
  'full',
  'sop',
  'section',
  'only',
  'now',
  'it',
  'them',
  'this',
  'that',
  'these',
  'those',
  'same',
  'again',
  'shorter',
  'smaller',
  'summary',
  'summarize',
  'the',
  'rewrite',
  'improve',
  'entire',
  'whole',
  'complete',
  'gesamte',
  'komplette',
  'document',
])

export function wantsFullSopIntent(promptText = '') {
  const text = String(promptText || '').trim()
  if (/\b(?:the\s+)?procedure\s+section\b|\bsection\s+procedure\b/i.test(text)) return false
  return FULL_SOP_INTENT_RE.test(text)
}

const REGISTER_FIELD_LINE_RE =
  /^(?:Linked|Finding|Datum|Beschreibung|Ursache|Aktion|Verantwortlich|Verknüpfungen|Entscheidung|Risiko|Begründung|Status|Fällig|Schweregrad|Ergebnis)\s*:/i

const RECORD_ENTRY_HEAD_RE = /^(?:DEV|CAPA|AUD|DEC)-[A-Z0-9]+-\d+/i

/** First non-empty line — section kind must not be inferred from record IDs in the body. */
function sectionHeadline(text = '') {
  const lines = String(text || '').split(/\r?\n/).map((line) => line.trim()).filter(Boolean)
  return lines[0] || String(text || '').trim()
}

export function isTraceabilitySectionHeading(text = '') {
  const head = sectionHeadline(text)
  if (!head || REGISTER_FIELD_LINE_RE.test(head) || RECORD_ENTRY_HEAD_RE.test(head)) return false
  if (/\b(?:zugeh[oö]rig|zugehoerig)\s+zu\s+sop-/i.test(head)) return true
  if (/[\u{1F534}\u{1F7E0}\u{1F7E1}\u{1F7E2}\u{1F535}\u{1F7E3}]/u.test(head)) return true
  return /^(?:DEVIATIONS?|ABWEICHUNGEN?|CAPAS?|AUDIT(?:\s+FINDINGS?)?|AUDITS?|DECISIONS?|ENTSCHEIDUNGEN?)\b/i.test(head)
}

export function getTraceabilitySectionKind(text = '') {
  const raw = String(text || '').trim()
  if (!raw) return null
  const head = sectionHeadline(raw)
  if (REGISTER_FIELD_LINE_RE.test(head) || RECORD_ENTRY_HEAD_RE.test(head)) return null
  if (raw.length > 220 && !isTraceabilitySectionHeading(head)) return null

  const t = head.replace(/[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}]/gu, ' ').toLowerCase()
  const hasCapa = /\b(?:capas?|caps)\b/.test(t)
  if (/\bcaps\b/.test(t) && !/\bscope\b/.test(t)) return 'capas'
  const hasDev = /\bdeviations?\b|\babweichungen?\b/.test(t)
  const hasDec = /\bdecisions?\b|\bentscheidungen?\b/.test(t)
  const hasAud = /\baudit\s+findings?\b/.test(t) || (/\baudit\b/.test(t) && !hasCapa && !hasDev)
  if (hasCapa && !hasDev) return 'capas'
  if (hasDev && !hasCapa) return 'deviations'
  if (hasDec) return 'decisions'
  if (hasAud) return 'audit'
  return null
}

export function extractSopRefs(text = '') {
  return [...new Set((String(text || '').match(/\b(SOP-[A-Z0-9-]+)\b/gi) || []).map((r) => r.toLowerCase()))]
}

export function isGenericLabelToken(token = '') {
  const t = String(token || '').trim().toLowerCase()
  if (!t || t.length < 3) return true
  if (LABEL_STOPWORDS.has(t)) return true
  return false
}
