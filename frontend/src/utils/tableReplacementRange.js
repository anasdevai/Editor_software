export const htmlContainsTable = (value = '') =>
  /<\/?(?:table|thead|tbody|tr|td|th)\b/i.test(String(value || ''))

const findEnclosingTableRange = (editor, from, to) => {
  const doc = editor?.state?.doc
  if (!doc || from == null || to == null) return null
  const docSize = doc.content.size
  const safeFrom = Math.max(0, Math.min(Number(from) || 0, docSize))
  const safeTo = Math.max(safeFrom, Math.min(Number(to) || safeFrom, docSize))
  let best = null

  const inspectResolvedPos = (pos) => {
    try {
      const resolved = doc.resolve(Math.max(0, Math.min(pos, docSize)))
      for (let depth = resolved.depth; depth >= 0; depth -= 1) {
        const node = resolved.node(depth)
        if (node?.type?.name === 'table') {
          const start = depth > 0 ? resolved.before(depth) : 0
          const end = start + node.nodeSize
          if (!best || (end - start) < (best.to - best.from)) best = { from: start, to: end }
          break
        }
      }
    } catch {
      // Best effort: nodesBetween below can still find the table.
    }
  }

  inspectResolvedPos(safeFrom)
  inspectResolvedPos(Math.max(safeFrom, safeTo - 1))

  try {
    doc.nodesBetween(safeFrom, safeTo, (node, pos) => {
      if (node.type?.name === 'table') {
        const candidate = { from: pos, to: pos + node.nodeSize }
        if (!best || (candidate.to - candidate.from) < (best.to - best.from)) best = candidate
        return false
      }
      return true
    })
  } catch {
    // Best effort only.
  }

  return best
}

export const resolveTableReplacementRange = (editor, range, html = '', targetType = '') => {
  if (!htmlContainsTable(html)) return range
  // A table_section includes its heading and prose. Keep that full range rather
  // than silently narrowing the accepted edit to the nested table.
  if (String(targetType || '').toLowerCase() === 'table_section') return range
  return findEnclosingTableRange(editor, range?.from, range?.to) || range
}
