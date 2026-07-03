import assert from 'node:assert/strict'
import { resolveTableReplacementRange } from '../../frontend/src/utils/tableReplacementRange.js'

const tableNode = { type: { name: 'table' }, nodeSize: 30 }
const editor = {
  state: {
    doc: {
      content: { size: 100 },
      resolve() { throw new Error('No resolved-position fixture needed') },
      nodesBetween(from, to, callback) {
        if (from < 50 && to > 20) callback(tableNode, 20)
      },
    },
  },
}

const mixedSection = { from: 5, to: 70 }
const tableHtml = '<h2>Definitions</h2><p>Intro</p><table><tr><td>A</td></tr></table>'

assert.deepEqual(
  resolveTableReplacementRange(editor, mixedSection, tableHtml, 'table_section'),
  mixedSection,
  'mixed heading/prose/table sections must retain their complete resolved range',
)
assert.deepEqual(
  resolveTableReplacementRange(editor, mixedSection, tableHtml, 'table'),
  { from: 20, to: 50 },
  'a table-only target should still expand to the exact table node',
)
assert.deepEqual(
  resolveTableReplacementRange(editor, mixedSection, '<p>Rewritten prose</p>', 'table_section'),
  mixedSection,
  'non-table output must never alter the resolved target range',
)

console.log(JSON.stringify({ ok: true, mixedSection, tableRange: { from: 20, to: 50 } }))
