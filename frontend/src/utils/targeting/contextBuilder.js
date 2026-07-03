import { parseSopDocument } from './sopParser.js'

const clamp = (value, min, max) => Math.max(min, Math.min(max, value))

const textBetweenSafe = (editor, from, to) => {
  try {
    return editor?.state?.doc?.textBetween?.(from, to, '\n') || ''
  } catch {
    return ''
  }
}

export function buildDeepAgentTargetContext({
  editor,
  target,
  resolution,
  intent,
  userQuery = '',
  profileContext = null,
  ragContext = null,
} = {}) {
  const tree = parseSopDocument(editor)
  const docSize = tree.docSize || editor?.state?.doc?.content?.size || 0
  const from = Number(target?.from ?? resolution?.range?.from ?? 0)
  const to = Number(target?.to ?? resolution?.range?.to ?? 0)
  const windowSize = 900

  const before = textBetweenSafe(editor, clamp(from - windowSize, 0, docSize), from).trim()
  const after = textBetweenSafe(editor, to, clamp(to + windowSize, 0, docSize)).trim()
  const nearbyTables = tree.tables
    .filter((table) => table.position.from <= to + windowSize && table.position.to >= from - windowSize)
    .map((table) => ({
      id: table.id,
      caption: table.caption,
      owningSection: table.owningSection,
      rowCount: table.rowCount,
      columnCount: table.columnCount,
      text: table.content,
      range: table.position,
    }))

  return {
    user_query: userQuery,
    intent: intent?.intent || resolution?.intent || '',
    resolved_target: resolution,
    target_text: target?.text || '',
    target_range: { from, to },
    surrounding_context: {
      before,
      after,
      parent_heading: target?.sectionName || resolution?.resolved_heading || '',
      sibling_table_count: nearbyTables.length,
    },
    contains_table: Boolean(resolution?.contains_table || nearbyTables.some((table) => table.range.from >= from && table.range.to <= to)),
    table_context: nearbyTables,
    profile_context: profileContext,
    rag_context: ragContext,
  }
}

export default buildDeepAgentTargetContext
