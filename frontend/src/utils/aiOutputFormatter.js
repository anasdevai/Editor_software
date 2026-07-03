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

const isMarkdownTableLine = (line = '') => /^\s*\|.+\|\s*$/.test(line)

const isLooseMarkdownTableLine = (line = '') => {
  const value = ensureString(line).trim()
  if (!value || !value.includes('|')) return false
  if (/^[-*:|\s]+$/.test(value)) return true
  const cells = splitMarkdownTableRow(value)
  return cells.length >= 2 && cells.some(Boolean)
}

const splitMarkdownTableRow = (line = '') =>
  ensureString(line)
    .trim()
    .replace(/^\||\|$/g, '')
    .split('|')
    .map((cell) => cell.replace(/\\\|/g, '|').trim())

const isMarkdownTableSeparator = (line = '') => {
  const cells = splitMarkdownTableRow(line)
  return cells.length > 0 && cells.every((cell) => /^:?-{3,}:?$/.test(cell))
}

const collectMarkdownTable = (lines = [], startIndex = 0) => {
  if (!isLooseMarkdownTableLine(lines[startIndex] || '')) return null

  const rows = []
  let index = startIndex
  while (index < lines.length && isLooseMarkdownTableLine(lines[index])) {
    const line = lines[index]
    if (!isMarkdownTableSeparator(line)) rows.push(splitMarkdownTableRow(line))
    index += 1
  }

  if (rows.length < 2) return null
  const width = Math.max(...rows.map((row) => row.length))
  const mostlyConsistent = rows.filter((row) => Math.abs(row.length - width) <= 1).length >= Math.ceil(rows.length * 0.75)
  if (width < 2 || !mostlyConsistent) return null

  return {
    header: rows[0],
    rows: rows.slice(1),
    endIndex: index,
  }
}

const renderMarkdownTable = (header = [], rows = []) => {
  if (!header.length) return ''
  const width = Math.max(header.length, ...rows.map((row) => row.length))
  const normalize = (row) => Array.from({ length: width }, (_, index) => row[index] || '')
  const headCells = normalize(header)
    .map((cell) => `<th>${escapeHtml(cell)}</th>`)
    .join('')
  const bodyRows = rows
    .map((row) => `<tr>${normalize(row).map((cell) => `<td>${escapeHtml(cell)}</td>`).join('')}</tr>`)
    .join('')
  return `<table><thead><tr>${headCells}</tr></thead><tbody>${bodyRows}</tbody></table>`
}

const parseAbbreviationDefinitionLine = (line = '') => {
  const match = ensureString(line).trim().match(/^([A-Z0-9][A-Z0-9/&().+\-\s]{1,28})\s*:\s+(.{2,})$/)
  if (!match) return null
  const term = match[1].trim()
  const definition = match[2].trim()
  if (!term || !definition || /[.!?]$/.test(term)) return null
  return [term, definition]
}

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

  for (let lineIndex = 0; lineIndex < lines.length; lineIndex += 1) {
    const line = lines[lineIndex].trim()
    if (!line) {
      flushParagraph()
      flushList()
      continue
    }

    const markdownTable = collectMarkdownTable(lines, lineIndex)
    if (markdownTable) {
      flushParagraph()
      flushList()
      html.push(renderMarkdownTable(markdownTable.header, markdownTable.rows))
      lineIndex = markdownTable.endIndex - 1
      continue
    }

    const definitionRow = parseAbbreviationDefinitionLine(line)
    if (definitionRow) {
      const rows = [definitionRow]
      let tableIndex = lineIndex + 1
      while (tableIndex < lines.length) {
        const nextRow = parseAbbreviationDefinitionLine(lines[tableIndex])
        if (!nextRow) break
        rows.push(nextRow)
        tableIndex += 1
      }
      if (rows.length >= 3) {
        flushParagraph()
        flushList()
        html.push(renderMarkdownTable(['Term', 'Definition'], rows))
        lineIndex = tableIndex - 1
        continue
      }
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
    .replace(/<(p|div)(\s[^>]*)?>\s*<(strong|b)>\s*([\s\S]*?)\s*<\/\3>\s*<\/\1>/gi, '<$1$2>$4</$1>')
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

