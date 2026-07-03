/**
 * editorUtils.js
 * ==============
 * Utilities for inspecting TipTap editor state.
 * Kept separate from App.jsx so they can be unit-tested independently.
 */

/**
 * Recursively extract all plain text from a TipTap node tree.
 * Returns a flat string of every text leaf joined by spaces.
 *
 * @param {object} node - Any TipTap JSON node
 * @returns {string}
 */
export function extractTextFromNode(node) {
  if (!node || typeof node !== 'object') return ''
  if (node.type === 'text') return (node.text || '').trim()

  const children = node.content || []
  return children
    .map(extractTextFromNode)
    .filter(Boolean)
    .join(' ')
    .trim()
}

/**
 * Determine whether a TipTap JSON document is effectively empty.
 *
 * A document is considered empty when ALL of the following are true:
 *   - It has no nodes, OR
 *   - Every node is a blank paragraph (type=paragraph with no content children), OR
 *   - Every text leaf in the entire tree is whitespace-only
 *
 * A document is NOT empty if it contains:
 *   - A heading with any text
 *   - A paragraph with any non-whitespace text
 *   - A list, table, image, or any other non-empty block
 *
 * @param {object|null} tiptapJson - The result of editor.getJSON()
 * @returns {boolean} true if the document has no meaningful content
 */
export function isEditorContentEmpty(tiptapJson) {
  if (!tiptapJson || typeof tiptapJson !== 'object') return true

  const nodes = tiptapJson.content || []
  if (nodes.length === 0) return true

  // Extract all text from the entire document tree
  const allText = extractTextFromNode(tiptapJson)
  if (allText.length > 0) return false

  // Check for non-text meaningful nodes (images, horizontal rules, etc.)
  const hasMeaningfulNode = nodes.some((node) => {
    const t = node.type || ''
    // These node types are meaningful even without text
    return ['image', 'horizontalRule', 'codeBlock', 'table'].includes(t)
  })

  return !hasMeaningfulNode
}

/**
 * Count the approximate number of words in a TipTap document.
 *
 * @param {object|null} tiptapJson
 * @returns {number}
 */
export function countWordsInDocument(tiptapJson) {
  const text = extractTextFromNode(tiptapJson || {})
  if (!text) return 0
  return text.split(/\s+/).filter(Boolean).length
}

const _text = (s) => ({ type: 'text', text: String(s ?? '') })
const _strongText = (s) => ({ type: 'text', text: String(s ?? ''), marks: [{ type: 'bold' }] })

const splitParagraphLines = (text = '') =>
  String(text || '')
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)

const isBulletLine = (line = '') => /^[-*•]\s+/.test(line)
const isNumberedLine = (line = '') => /^\(?[A-Za-z0-9]+\)?[.)]\s+/.test(line)
const isKeyValueLine = (line = '') => /^[A-ZÄÖÜ][A-Za-zÄÖÜäöüß\s/&()\-]{1,40}:\s+\S+/.test(line)

const paragraphNode = (text = '') => {
  const value = String(text ?? '')
  return value ? { type: 'paragraph', content: [_text(value)] } : { type: 'paragraph' }
}
const headingNode = (text = '', level = 2) => ({ type: 'heading', attrs: { level }, content: [_text(text)] })
const listItemNode = (text = '') => ({ type: 'listItem', content: [paragraphNode(text)] })

const blockText = (block) =>
  String(block?.text ?? block?.content ?? block?.value ?? block?.markdown ?? '').trim()

const normalizeTableRows = (rows = []) => {
  if (!Array.isArray(rows)) return []
  const cleaned = rows
    .map((row) => (Array.isArray(row) ? row.map((cell) => String(cell ?? '').trim()) : []))
    .filter((row) => row.some(Boolean))
  const maxCols = cleaned.reduce((max, row) => Math.max(max, row.length), 0)
  if (!maxCols) return []
  return cleaned.map((row) => {
    const next = row.slice(0, maxCols)
    while (next.length < maxCols) next.push('')
    return next
  })
}

/**
 * Map backend PDF/OCR blocks to a TipTap-compatible doc JSON (StarterKit + table).
 * @param {Array<{type: string, text?: string, items?: string[], rows?: string[][]}>} blocks
 * @param {string} fallbackText
 * @returns {{ type: 'doc', content: object[] }}
 */
export function mapBlocksToTipTapDoc(blocks, fallbackText = '') {
  const content = []
  if (!Array.isArray(blocks) || blocks.length === 0) {
    const t = String(fallbackText || '').trim()
    if (t) content.push({ type: 'paragraph', content: [_text(t)] })
    return { type: 'doc', content }
  }

  for (const block of blocks) {
    const typ = String(block.type || '').toLowerCase()
    const text = blockText(block)
    if (typ === 'text' && text) {
      const style = String(block.style || 'paragraph').toLowerCase()
      if (style === 'heading') {
        content.push(headingNode(text, 2))
      } else {
        content.push(paragraphNode(text))
      }
    } else if ((typ === 'section_heading' || typ === 'heading' || typ === 'title') && text) {
      const level = Math.min(3, Math.max(1, Number(block.level) || 2))
      content.push(headingNode(text, level))
    } else if ((typ === 'paragraph' || typ === 'body' || typ === 'caption') && text) {
      const lines = splitParagraphLines(text)
      if (!lines.length) continue
      if (lines.every(isBulletLine)) {
        content.push({
          type: 'bulletList',
          content: lines.map((line) => listItemNode(line.replace(/^[-*•]\s+/, '').trim())),
        })
        continue
      }
      if (lines.every(isNumberedLine)) {
        content.push({
          type: 'orderedList',
          content: lines.map((line) => listItemNode(line.replace(/^\(?[A-Za-z0-9]+\)?[.)]\s+/, '').trim())),
        })
        continue
      }
      for (const line of lines) {
        if (isKeyValueLine(line)) {
          const [key, ...rest] = line.split(':')
          const value = rest.join(':').trim()
          content.push({
            type: 'paragraph',
            content: [_strongText(`${key.trim()}: `), _text(value)],
          })
        } else {
          content.push(paragraphNode(line))
        }
      }
    } else if ((typ === 'two_column_row' || typ === 'key_value') && (block.left || block.right)) {
      const left = String(block.left || '').trim()
      const right = String(block.right || '').trim()
      if (left && right) {
        content.push({
          type: 'paragraph',
          content: [
            _strongText(`${left}: `),
            _text(right),
          ],
        })
      } else {
        content.push(paragraphNode(left || right))
      }
    } else if ((typ === 'bullet_list' || typ === 'numbered_list') && Array.isArray(block.items)) {
      const listType = typ === 'numbered_list' ? 'orderedList' : 'bulletList'
      const items = block.items
        .filter((it) => String(it ?? '').trim())
        .map((it) => listItemNode(it))
      if (items.length) content.push({ type: listType, content: items })
    } else if (typ === 'table' && (Array.isArray(block.rows) || Array.isArray(block.content))) {
      const sourceRows = Array.isArray(block.rows) ? block.rows : block.content
      const headerRows = Math.max(0, Number(block.header_rows ?? block.headerRows ?? 0) || 0)
      const normalizedRows = normalizeTableRows(sourceRows)
      const rows = []
      for (let rowIndex = 0; rowIndex < normalizedRows.length; rowIndex += 1) {
        const row = normalizedRows[rowIndex]
        const cellType = rowIndex < headerRows ? 'tableHeader' : 'tableCell'
        const cells = row.map((cell) => ({
          type: cellType,
          content: [paragraphNode(cell)],
        }))
        if (cells.length) rows.push({ type: 'tableRow', content: cells })
      }
      if (rows.length) {
        content.push({
          type: 'table',
          content: rows,
        })
      }
    } else if (text) {
      content.push(paragraphNode(text))
    }
  }

  if (!content.length) {
    const t = String(fallbackText || '').trim()
    if (t) content.push({ type: 'paragraph', content: [_text(t)] })
  }

  return { type: 'doc', content }
}
