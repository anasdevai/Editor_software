const stripHtml = (value = '') =>
  String(value || '')
    .replace(/<br\s*\/?>/gi, '\n')
    .replace(/<\/p>/gi, '\n')
    .replace(/<\/div>/gi, '\n')
    .replace(/<[^>]+>/g, '')
    .replace(/\n{3,}/g, '\n\n')
    .trim()

export const containsHtmlTable = (value = '') =>
  /<\/?(?:table|thead|tbody|tr|td|th)\b/i.test(String(value || ''))

export function stripFullParagraphBold(value = '') {
  return String(value || '')
    .replace(/<p>\s*<(?:strong|b)>([\s\S]*?)<\/(?:strong|b)>\s*<\/p>/gi, '<p>$1</p>')
    .replace(/^<(?:strong|b)>([\s\S]*?)<\/(?:strong|b)>$/i, '$1')
}

export function normalizeProseReplacement(value = '') {
  const raw = stripFullParagraphBold(value)
  if (/<\/?[a-z][\s\S]*>/i.test(raw)) return raw
  return stripHtml(raw)
    .split(/\n{2,}/)
    .map((paragraph) => paragraph.trim())
    .filter(Boolean)
    .map((paragraph) => `<p>${paragraph}</p>`)
    .join('')
}

export function normalizeReplacementForPatch({
  replacement = '',
  containsTable = false,
  tableNormalizer = null,
  tableContext = [],
} = {}) {
  const raw = String(replacement || '').trim()
  if (!raw) return ''
  if (containsTable) {
    const normalized = typeof tableNormalizer === 'function'
      ? tableNormalizer(raw, tableContext)
      : raw
    return containsHtmlTable(normalized) ? normalized : raw
  }
  return normalizeProseReplacement(raw)
}

export function applyResolvedPatch(editor, { from, to, content }) {
  if (!editor || from == null || to == null || !content) return false
  editor.chain().focus().insertContentAt({ from, to }, content).run()
  return true
}

export default {
  applyResolvedPatch,
  containsHtmlTable,
  normalizeProseReplacement,
  normalizeReplacementForPatch,
  stripFullParagraphBold,
}
