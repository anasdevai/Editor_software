/**
 * Heuristics for /api/ai/action selection: avoid sending formatted gap-check HTML
 * through Improve/Rewrite (which expect raw SOP prose).
 */

export function selectionLooksLikeFormattedAiReport(text) {
  const t = String(text || '').toLowerCase()
  if (t.length < 100) return false
  const markers = [
    'identified gaps:',
    'summary:',
    'risk/impact:',
    'recommended fixes:',
    'suggested sop text:',
    'structured qa findings',
    'compliance-lückenanalyse',
    'gap check',
  ]
  const hits = markers.filter((m) => t.includes(m))
  return hits.length >= 2
}

function normalizeForCompare(s) {
  return String(s || '')
    .replace(/<[^>]+>/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .toLowerCase()
}

/** True when the user re-selected the last AI suggestion body (same action). */
export function selectionMatchesLastAiSuggestion(selectionText, lastSuggestedPlain) {
  const a = normalizeForCompare(selectionText)
  const b = normalizeForCompare(lastSuggestedPlain)
  if (!a || !b || a.length < 80 || b.length < 80) return false
  if (a === b) return true
  if (a.includes(b) || b.includes(a)) return Math.abs(a.length - b.length) < Math.max(a.length, b.length) * 0.15
  return false
}
