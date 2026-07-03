import {
  buildEditorSectionIndex,
  buildEditorTableIndex,
  collectSectionRanges,
} from '../editorTargetResolver.js'

const compact = (value = '', limit = 1200) =>
  String(value || '').replace(/\s+/g, ' ').trim().slice(0, limit)

const nodeId = (prefix, index, from, to = null) => {
  const start = Math.max(0, Number(from) || 0)
  const end = Number.isFinite(Number(to)) ? Math.max(start, Number(to)) : ''
  return `${prefix}_${index + 1}_${start}${end !== '' ? `_${end}` : ''}`
}

const nodeAttrId = (node) => {
  const attrs = node?.attrs || {}
  return String(attrs.id || attrs.blockId || attrs.block_id || attrs['data-id'] || '').trim() || null
}

const buildBlockMeta = (doc) => {
  const byPos = new Map()
  const headings = []
  const tables = []
  const paragraphs = []
  let blockIndex = 0
  doc?.descendants?.((node, pos) => {
    if (!node?.isBlock) return true
    const type = node.type?.name || 'block'
    const from = pos
    const to = pos + (node.nodeSize || 0)
    const id = nodeAttrId(node)
    const meta = { id, type, from, to, index: blockIndex, text: compact(node.textContent || '', 3000), node }
    byPos.set(pos, meta)
    blockIndex += 1
    if (type === 'heading') headings.push(meta)
    else if (type === 'table') tables.push(meta)
    else if (type !== 'tableRow' && type !== 'tableCell' && type !== 'tableHeader') paragraphs.push(meta)
    return type !== 'table'
  })
  return { byPos, headings, tables, paragraphs }
}

const headingIdForSection = (section, index, blockMeta) => {
  const heading = blockMeta.headings.find((item) => item.from === section.from)
    || blockMeta.headings.find((item) => Math.abs(item.from - section.from) <= 1)
  return heading?.id || nodeId('section', index, section.from, section.to)
}

const parentSectionForRange = (sections, from, to = from) =>
  [...sections]
    .reverse()
    .find((section) => Number(section.position?.from) <= from && Number(section.position?.to) >= to)
  || [...sections].reverse().find((section) => Number(section.position?.from) <= from)
  || null

const tableToRows = (tableNode) => {
  const rows = []
  tableNode?.forEach?.((rowNode) => {
    const cells = []
    rowNode?.forEach?.((cellNode) => {
      cells.push({
        text: compact(cellNode.textContent || '', 800),
        header: cellNode.type?.name === 'tableHeader',
      })
    })
    if (cells.length) rows.push(cells)
  })
  return rows
}

export function parseSopDocument(editor) {
  const doc = editor?.state?.doc
  if (!doc) {
    return {
      root: null,
      nodes: [],
      sections: [],
      tables: [],
      paragraphs: [],
      fullText: '',
      docSize: 0,
    }
  }

  const fullText = doc.textBetween(0, doc.content.size, '\n')
  const rawSections = collectSectionRanges(doc)
  const blockMeta = buildBlockMeta(doc)
  const sections = buildEditorSectionIndex(editor).map((section, index) => ({
    id: section.id || headingIdForSection(section, index, blockMeta),
    type: 'section',
    title: section.sectionName || section.title || `Section ${index + 1}`,
    content: section.text || '',
    parent_id: null,
    children: [],
    position: { from: section.from, to: section.to },
    confidence: section.confidence ?? 0.8,
    sectionType: section.sectionType || 'Heading',
    raw: section,
  }))

  const tables = buildEditorTableIndex(editor).map((table, index) => ({
    id: table.id || blockMeta.byPos.get(table.from)?.id || nodeId('table', index, table.from, table.to),
    type: 'table',
    title: table.caption || `Table ${index + 1}`,
    content: table.text || '',
    parent_id: parentSectionForRange(sections, table.from, table.to)?.id || null,
    children: [],
    position: { from: table.from, to: table.to },
    caption: table.caption || `Table ${index + 1}`,
    owningSection: table.owningSection || '',
    rowCount: table.rowCount || 0,
    columnCount: table.columnCount || 0,
    rows: table.rows || [],
    raw: table,
  }))

  const paragraphs = []
  let paragraphIndex = 0
  doc.descendants?.((node, pos) => {
    if (!node?.isBlock || node.type?.name === 'table' || node.type?.name === 'tableRow' || node.type?.name === 'tableCell') {
      return true
    }
    const text = compact(node.textContent || '', 3000)
    if (!text || node.type?.name === 'heading') return true
    paragraphIndex += 1
    const to = pos + (node.nodeSize || text.length + 2)
    const parent = parentSectionForRange(sections, pos, to)
    paragraphs.push({
      id: nodeAttrId(node) || nodeId('paragraph', paragraphIndex - 1, pos, to),
      type: 'paragraph',
      title: text.slice(0, 80),
      content: text,
      parent_id: parent?.id || null,
      parent_title: parent?.title || null,
      children: [],
      position: { from: pos, to },
      raw: { node, pos },
    })
    return true
  })

  doc.descendants?.((node, pos) => {
    if (node?.type?.name !== 'table') return true
    const existing = tables.find((table) => table.position.from === pos)
    if (existing && (!existing.rows || existing.rows.length === 0)) {
      existing.rows = tableToRows(node)
      existing.rowCount = existing.rows.length || existing.rowCount
      existing.columnCount = Math.max(0, ...existing.rows.map((row) => row.length), existing.columnCount || 0)
    }
    return false
  })

  const allNodes = [
    {
      id: 'doc_root',
      type: 'full_document',
      title: 'Current SOP',
      content: fullText,
      parent_id: null,
      children: sections.map((section) => section.id),
      position: { from: 0, to: doc.content.size },
    },
    ...sections,
    ...tables,
    ...paragraphs,
  ]
  const schema = allNodes
    .filter((node) => node.id !== 'doc_root')
    .map((node) => {
      const parent = node.parent_id ? sections.find((section) => section.id === node.parent_id) : null
      const indent = parent && node.type !== 'section' ? '  ' : ''
      if (node.type === 'table') {
        const firstRow = Array.isArray(node.rows?.[0]) ? node.rows[0] : []
        const columns = firstRow.map((cell) => cell?.text).filter(Boolean).join(' | ')
        return `${indent}[table:${node.id}] columns: ${columns || compact(node.content, 80)} (${node.rowCount || 0} rows)`
      }
      if (node.type === 'paragraph') return `${indent}[para:${node.id}] ${compact(node.content, 80)}`
      return `[section:${node.id}] ${compact(node.title, 120)}`
    })
    .join('\n')

  return {
    root: allNodes[0],
    nodes: allNodes,
    schema,
    sections,
    rawSections,
    tables,
    paragraphs,
    fullText,
    docSize: doc.content.size,
  }
}

export default parseSopDocument
