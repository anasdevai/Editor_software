const MARKDOWN_TOKENS_RE = /(^|\s)(#{1,6}\s+|\*\*|__|---+|\*{1,3})(?=\S|$)/gm

const ensureString = (value) => (typeof value === 'string' ? value : String(value || ''))

const normalizeLine = (line = '') =>
  ensureString(line)
    .replace(/\*\*(.*?)\*\*/g, '$1')
    .replace(/__(.*?)__/g, '$1')
    .replace(/^\s*#{1,6}\s+/, '')
    .replace(/^\s*---+\s*$/, '')
    .replace(/^\s*[-*]\s+(?=\*)/, '- ')
    .replace(/\s+/g, ' ')
    .trim()

const escapeHtml = (text = '') =>
  ensureString(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')

const cleanPlainText = (text = '') =>
  ensureString(text)
    .replace(/\r\n/g, '\n')
    .replace(/\r/g, '\n')
    .replace(MARKDOWN_TOKENS_RE, '$1')
    .replace(/\n{3,}/g, '\n\n')
    .split('\n')
    .map(normalizeLine)
    .filter((line, index, arr) => line || arr[index - 1])
    .join('\n')
    .trim()

export const renderCleanSuggestionHtml = (text = '') => {
  const cleaned = cleanPlainText(text)
  if (!cleaned) return '<p>No suggestion returned.</p>'

  const lines = cleaned.split('\n')
  const html = []
  let paragraph = []
  let listItems = []
  let listType = null

  const flushParagraph = () => {
    if (!paragraph.length) return
    html.push(`<p>${escapeHtml(paragraph.join(' '))}</p>`)
    paragraph = []
  }

  const flushList = () => {
    if (!listItems.length || !listType) return
    const tag = listType
    html.push(`<${tag}>${listItems.map((item) => `<li>${escapeHtml(item)}</li>`).join('')}</${tag}>`)
    listItems = []
    listType = null
  }

  for (const rawLine of lines) {
    const line = rawLine.trim()
    if (!line) {
      flushParagraph()
      flushList()
      continue
    }

    const headingMatch = line.match(/^(Summary|Identified Gaps|Risk\/Impact|Recommended Fixes|Suggested SOP Text)\s*:?\s*$/i)
    if (headingMatch) {
      flushParagraph()
      flushList()
      html.push(`<h3>${escapeHtml(headingMatch[1])}</h3>`)
      continue
    }

    const orderedMatch = line.match(/^\d+[.)]\s+(.+)$/)
    if (orderedMatch) {
      flushParagraph()
      if (listType && listType !== 'ol') flushList()
      listType = 'ol'
      listItems.push(orderedMatch[1].trim())
      continue
    }

    const bulletMatch = line.match(/^[-*]\s+(.+)$/)
    if (bulletMatch) {
      flushParagraph()
      if (listType && listType !== 'ul') flushList()
      listType = 'ul'
      listItems.push(bulletMatch[1].trim())
      continue
    }

    if (listType) {
      listItems[listItems.length - 1] = `${listItems[listItems.length - 1]} ${line}`.trim()
      continue
    }

    paragraph.push(line)
  }

  flushParagraph()
  flushList()

  return html.join('') || '<p>No suggestion returned.</p>'
}

export const sanitizeRenderedHtml = (html = '') => {
  const raw = ensureString(html)
  if (!raw.trim()) return '<p>No suggestion returned.</p>'

  // Keep output readable and strip common risky tags/attrs.
  return raw
    .replace(/<script[\s\S]*?>[\s\S]*?<\/script>/gi, '')
    .replace(/<style[\s\S]*?>[\s\S]*?<\/style>/gi, '')
    .replace(/\son\w+="[^"]*"/gi, '')
    .replace(/\son\w+='[^']*'/gi, '')
    .replace(/javascript:/gi, '')
}

export const formatAiSuggestionForUi = ({ action, suggestedText, structuredData }) => {
  const normalizedAction = String(action || '').toLowerCase()
  const fallbackRaw = ensureString(suggestedText)
  const gapRaw = normalizedAction === 'gap_check' ? ensureString(structuredData?.analysis || fallbackRaw) : fallbackRaw

  const asHtml =
    /<\/?[a-z][\s\S]*>/i.test(gapRaw)
      ? sanitizeRenderedHtml(gapRaw)
      : sanitizeRenderedHtml(renderCleanSuggestionHtml(gapRaw))

  return asHtml || '<p>No suggestion returned.</p>'
}

