function decodeEntities(text) {
  return String(text || '')
    .replaceAll('&nbsp;', ' ')
    .replaceAll('&amp;', '&')
    .replaceAll('&lt;', '<')
    .replaceAll('&gt;', '>')
    .replaceAll('&quot;', '"')
    .replaceAll('&#39;', "'")
}

export function htmlToPlainText(html) {
  const normalized = String(html || '')
    .replace(/<br\s*\/?>/gi, '\n')
    .replace(/<\/p>/gi, '\n')
    .replace(/<p[^>]*>/gi, '')
    .replace(/<[^>]+>/g, '')
  return decodeEntities(normalized)
}

export function deriveSopTitleFromText(text) {
  const lines = String(text || '')
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
  if (lines.length === 0) return 'Generated SOP'

  const explicitTitle = lines.find((line) => /^1\.0\s+title\b/i.test(line))
  if (explicitTitle) {
    const idx = lines.indexOf(explicitTitle)
    const next = lines[idx + 1]
    if (next) return next.slice(0, 120)
  }
  return lines[0].slice(0, 120)
}

export function plainTextToTiptapDoc(text) {
  const lines = String(text || '')
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)

  if (lines.length === 0) {
    return { type: 'doc', content: [] }
  }

  const content = lines.map((line) => {
    // Keep structure readable by mapping major numbered lines to headings.
    if (/^\d+\.0\s+/.test(line)) {
      return { type: 'heading', attrs: { level: 2 }, content: [{ type: 'text', text: line }] }
    }
    return { type: 'paragraph', content: [{ type: 'text', text: line }] }
  })

  return {
    type: 'doc',
    content,
  }
}
