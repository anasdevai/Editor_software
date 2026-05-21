/**
 * Shared intent detection for Actions tab / editor target resolution.
 */

const FULL_SOP_INTENT_RE =
  /\b(?:rewrite|re-?write|improve|umschreiben|verbessern?)\b[\s\S]*\b(?:(?:this\s+)?full\s+sop|(?:entire|whole|complete|gesamte?|komplette?)\s+sop|sop\s+(?:komplett|vollständig|gesamt))\b|\b(?:(?:this\s+)?full\s+sop|(?:entire|whole|complete)\s+sop)\b[\s\S]*\b(?:rewrite|improve|umschreiben|verbessern?)\b/i

const LABEL_STOPWORDS = new Set([
  'full',
  'sop',
  'section',
  'only',
  'this',
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
  return FULL_SOP_INTENT_RE.test(String(promptText || '').trim())
}

const REGISTER_FIELD_LINE_RE =
  /^(?:Linked|Finding|Datum|Beschreibung|Ursache|Aktion|Verantwortlich|Verknüpfungen|Entscheidung|Risiko|Begründung|Status|Fällig|Schweregrad|Ergebnis)\s*:/i

export function getTraceabilitySectionKind(text = '') {
  const raw = String(text || '').trim()
  if (REGISTER_FIELD_LINE_RE.test(raw)) return null
  const t = raw.replace(/[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}]/gu, ' ').toLowerCase()
  const hasCapa = /\bcapas?\b/.test(t)
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
