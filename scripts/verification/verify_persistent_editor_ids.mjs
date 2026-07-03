import assert from 'node:assert/strict'
import { getSchema } from '../../frontend/node_modules/@tiptap/core/dist/index.js'
import StarterKit from '../../frontend/node_modules/@tiptap/starter-kit/dist/index.js'
import { Table } from '../../frontend/node_modules/@tiptap/extension-table/dist/index.js'
import { TableRow } from '../../frontend/node_modules/@tiptap/extension-table-row/dist/index.js'
import { TableCell } from '../../frontend/node_modules/@tiptap/extension-table-cell/dist/index.js'
import { TableHeader } from '../../frontend/node_modules/@tiptap/extension-table-header/dist/index.js'
import UniqueID, { generateUniqueIds } from '../../frontend/node_modules/@tiptap/extension-unique-id/dist/index.js'

import { buildEditorSectionIndex, buildEditorTableIndex } from '../../frontend/src/utils/editorTargetResolver.js'
import { parseSopDocument } from '../../frontend/src/utils/targeting/sopParser.js'

const uniqueId = UniqueID.configure({
  attributeName: 'id',
  types: ['heading', 'paragraph', 'table'],
})
const extensions = [StarterKit, Table, TableRow, TableHeader, TableCell, uniqueId]
const source = {
  type: 'doc',
  content: [
    { type: 'heading', attrs: { level: 1 }, content: [{ type: 'text', text: 'Purpose' }] },
    { type: 'paragraph', content: [{ type: 'text', text: 'Defines why this SOP exists.' }] },
    { type: 'heading', attrs: { level: 2 }, content: [{ type: 'text', text: 'Scope' }] },
    { type: 'paragraph', content: [{ type: 'text', text: 'Applies to the quality team.' }] },
    {
      type: 'table',
      content: [{
        type: 'tableRow',
        content: [{
          type: 'tableHeader',
          content: [{ type: 'paragraph', content: [{ type: 'text', text: 'Role' }] }],
        }, {
          type: 'tableCell',
          content: [{ type: 'paragraph', content: [{ type: 'text', text: 'QA' }] }],
        }],
      }],
    },
  ],
}

const identified = generateUniqueIds(source, extensions)
const targetNodes = []
const walk = (node) => {
  if (['heading', 'paragraph', 'table'].includes(node?.type)) targetNodes.push(node)
  for (const child of node?.content || []) walk(child)
}
walk(identified)

assert.ok(targetNodes.length >= 7)
assert.ok(targetNodes.every((node) => typeof node.attrs?.id === 'string' && node.attrs.id.length > 8))
assert.equal(new Set(targetNodes.map((node) => node.attrs.id)).size, targetNodes.length)

const identifiedAgain = generateUniqueIds(identified, extensions)
const idsAgain = []
const collectIds = (node) => {
  if (['heading', 'paragraph', 'table'].includes(node?.type)) idsAgain.push(node.attrs.id)
  for (const child of node?.content || []) collectIds(child)
}
collectIds(identifiedAgain)
assert.deepEqual(idsAgain, targetNodes.map((node) => node.attrs.id))

const schema = getSchema(extensions)
const doc = schema.nodeFromJSON(identified)
const editor = { state: { doc } }
const sections = buildEditorSectionIndex(editor)
const tables = buildEditorTableIndex(editor)
const parsed = parseSopDocument(editor)

assert.equal(sections.length, 2)
assert.deepEqual(sections.map((section) => section.id), targetNodes.filter((node) => node.type === 'heading').map((node) => node.attrs.id))
assert.equal(tables.length, 1)
assert.equal(tables[0].id, targetNodes.find((node) => node.type === 'table').attrs.id)
assert.equal(new Set(parsed.nodes.map((node) => node.id)).size, parsed.nodes.length)
assert.ok(parsed.paragraphs.every((paragraph) => paragraph.id && paragraph.parent_id))

console.log(JSON.stringify({
  ok: true,
  sections: sections.length,
  tables: tables.length,
  paragraphs: parsed.paragraphs.length,
  unique_targets: parsed.nodes.length,
}, null, 2))
