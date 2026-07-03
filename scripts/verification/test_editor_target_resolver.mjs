import assert from 'node:assert/strict'
import { captureEditorSelectionForAction } from '../../frontend/src/utils/editScopeInference.js'
import { buildEditorSectionIndex, resolveTargetInEditor } from '../../frontend/src/utils/editorTargetResolver.js'
import { TargetResolverAgent } from '../../frontend/src/utils/targeting/targetResolverAgent.js'
import { parseSopDocument } from '../../frontend/src/utils/targeting/sopParser.js'

function makeNode(text, type = 'paragraph', attrs = {}) {
  return {
    isBlock: true,
    textContent: text,
    type: { name: type },
    attrs,
    nodeSize: String(text || '').length + 2,
  }
}

function makeDoc(blockDefs) {
  const blocks = []
  let pos = 0
  for (const def of blockDefs) {
    const node = makeNode(def.text, def.type, def.attrs)
    blocks.push({ node, pos, text: def.text })
    pos += node.nodeSize
  }
  return {
    _blocks: blocks,
    content: { size: pos },
    descendants(callback) {
      for (const block of blocks) callback(block.node, block.pos)
    },
    textBetween(from, to, separator = '\n') {
      const parts = []
      for (const block of blocks) {
        const start = block.pos + 1
        const end = start + block.text.length
        const overlapStart = Math.max(from, start)
        const overlapEnd = Math.min(to, end)
        if (overlapEnd > overlapStart) {
          parts.push(block.text.slice(overlapStart - start, overlapEnd - start))
        }
      }
      return parts.join(separator)
    },
  }
}

const doc = makeDoc([
  { text: 'SOP-IT-002 - Netzwerksicherheit & Firewall (OT/IT-Trennung)' },
  { text: 'Version: 2.2 | Status: Effective | Department: IT/OT' },
  { text: '1. Zweck', type: 'heading', attrs: { level: 2 } },
  { text: 'Schutz des Produktionsnetzwerks vor unbefugten Zugriffen aus dem Büronetzwerk und dem Internet.' },
  { text: '🔴 DEVIATIONS (zugehörig zu SOP-IT-002)' },
  { text: 'DEV-IT-011 – Firewall-Regel zu permissiv' },
  { text: 'Datum: 2024-05-22 | Status: Closed' },
  { text: '🟠 CAPAs (zugehörig zu SOP-IT-002)' },
  { text: 'CAPA-IT-011 – Firewall-Regel-Review' },
  { text: 'Linked DEV: DEV-IT-011 | Status: Effective | Fällig: 2024-06-15' },
  { text: 'Aktion: Vollständiger Review aller Firewall-Regeln, Least-Privilege-Prinzip' },
  { text: 'Verantwortlich: IT-Sicherheit' },
  { text: 'CAPA-IT-012 – Segmentierung nach IEC 62443' },
  { text: 'Linked DEV: DEV-IT-012 | Status: Effective | Fällig: 2024-07-31' },
  { text: '🟣 DECISIONS (zugehörig zu SOP-IT-002)' },
  { text: 'DEC-IT-001 – Zero Trust Netzwerksegmentierung' },
  { text: 'Entscheidung: Umsetzung der Segmentierung' },
])

const editor = { isDestroyed: false, state: { doc } }

const capa = resolveTargetInEditor(editor, {
  prompt: 'rewrite the caps section',
  sectionHint: 'caps',
  targetScope: 'section',
})
assert.match(capa.text, /CAPA-IT-011/)
assert.match(capa.text, /CAPA-IT-012/)
assert.doesNotMatch(capa.text, /DEC-IT-001/)
assert.ok(capa.text.length > '🟠 CAPAs (zugehörig zu SOP-IT-002)'.length)
assert.equal(capa.from, doc._blocks.find((block) => block.text.startsWith('🟠 CAPAs')).pos)

const zweck = resolveTargetInEditor(editor, {
  prompt: 'improve zweck section',
  sectionHint: 'zweck',
  targetScope: 'section',
})
assert.match(zweck.text, /1\. Zweck/)
assert.match(zweck.text, /Schutz des Produktionsnetzwerks/)
assert.ok(zweck.text.length > '1. Zweck'.length)

const decisions = resolveTargetInEditor(editor, {
  prompt: 'rewrite Decisions section',
  sectionHint: 'Decisions',
  targetScope: 'section',
})
assert.match(decisions.text, /DEC-IT-001/)
assert.doesNotMatch(decisions.text, /CAPA-IT-012/)
assert.ok(decisions.text.length > '🟣 DECISIONS (zugehörig zu SOP-IT-002)'.length)

const capaHeadingBlock = doc._blocks.find((block) => block.text.includes('CAPAs'))
const headingSelectionEditor = {
  isDestroyed: false,
  state: {
    doc,
    selection: {
      from: capaHeadingBlock.pos + 1,
      to: capaHeadingBlock.pos + 1 + capaHeadingBlock.text.length,
      empty: false,
    },
  },
}
const capturedCapa = captureEditorSelectionForAction(headingSelectionEditor)
assert.equal(capturedCapa.from, capaHeadingBlock.pos + 1)
assert.equal(capturedCapa.selectedText, capaHeadingBlock.text)
assert.doesNotMatch(capturedCapa.selectedText, /CAPA-IT-011/)
assert.doesNotMatch(capturedCapa.selectedText, /DEC-IT-001/)

const zweckHeadingBlock = doc._blocks.find((block) => block.text === '1. Zweck')
const zweckSelectionEditor = {
  isDestroyed: false,
  state: {
    doc,
    selection: {
      from: zweckHeadingBlock.pos + 1,
      to: zweckHeadingBlock.pos + 1 + zweckHeadingBlock.text.length,
      empty: false,
    },
  },
}
const capturedZweck = captureEditorSelectionForAction(zweckSelectionEditor)
assert.match(capturedZweck.selectedText, /1\. Zweck/)
assert.doesNotMatch(capturedZweck.selectedText, /Schutz des Produktionsnetzwerks/)

const embeddedDoc = makeDoc([
  { text: 'SOP-IT-003 - Notfallzugriff (Break-Glass-Verfahren)' },
  { text: '1. Zweck', type: 'heading', attrs: { level: 2 } },
  { text: 'Ermöglichung von Zugriffen auf OT-Systeme in kritischen Produktionsstillständen.' },
  { text: 'DEV-IT-025 – Zwei-Umschläge-Prinzip umgangen' },
  { text: 'Ursache: Nicht autorisierte Software installiert 🟠 CAPAs (zugehörig zu SOP-IT-003)' },
  { text: 'CAPA-IT-021 – Automatisches Ticket nach Break-Glass' },
  { text: 'Linked DEV: DEV-IT-021 | Status: Effective | Fällig: 2024-07-31' },
  { text: 'Aktion: Nach Notfallzugriff wird automatisch QMS-Ticket erstellt, Frist 24h' },
  { text: 'Verantwortlich: IT' },
  { text: 'CAPA-IT-022 – Physische Sicherheit der Umschläge' },
  { text: 'Linked DEV: DEV-IT-022 | Status: Open | Fällig: 2024-08-31' },
  { text: 'Aktion: Neue versiegelte Umschläge mit fortlaufender Nummer, wöchentliche Sichtkontrolle' },
  { text: 'Verantwortlich: QA' },
  { text: '🟣 DECISIONS (zugehörig zu SOP-IT-003)' },
  { text: 'DEC-IT-001 – Vier-Augen-Prinzip bleibt verpflichtend' },
])
const embeddedEditor = { isDestroyed: false, state: { doc: embeddedDoc } }
const embeddedCapa = resolveTargetInEditor(embeddedEditor, {
  prompt: 'rewrite the capas sections',
  sectionHint: '',
  targetScope: '',
})
assert.match(embeddedCapa.text, /CAPAs/)
assert.match(embeddedCapa.text, /CAPA-IT-021/)
assert.match(embeddedCapa.text, /CAPA-IT-022/)
assert.doesNotMatch(embeddedCapa.text, /DEV-IT-025/)
assert.doesNotMatch(embeddedCapa.text, /DEC-IT-001/)
assert.ok(embeddedCapa.text.length > '🟠 CAPAs (zugehörig zu SOP-IT-003)'.length + 180)

const embeddedCapaMarker = embeddedDoc._blocks.find((block) => block.text.includes('CAPAs'))
const selectedEmbeddedHeadingEditor = {
  isDestroyed: false,
  state: {
    doc: embeddedDoc,
    selection: {
      from: embeddedCapaMarker.pos + 1 + embeddedCapaMarker.text.indexOf('🟠 CAPAs'),
      to: embeddedCapaMarker.pos + 1 + embeddedCapaMarker.text.length,
      empty: false,
    },
  },
}
const selectedEmbeddedCapa = resolveTargetInEditor(selectedEmbeddedHeadingEditor, {
  prompt: 'rewrite this section',
  selection: selectedEmbeddedHeadingEditor.state.selection,
  targetScope: 'selection',
})
assert.match(selectedEmbeddedCapa.text, /CAPAs/)
assert.doesNotMatch(selectedEmbeddedCapa.text, /CAPA-IT-021/)
assert.doesNotMatch(selectedEmbeddedCapa.text, /CAPA-IT-022/)
assert.doesNotMatch(selectedEmbeddedCapa.text, /DEC-IT-001/)

const enrichedSidebar = `rewrite the capas sections

[Assistant constraints]
Target section: CAPAs
Apply to the complete section body under that heading (all paragraphs until the next section), not the heading line alone.`
const classifierSelectionScope = resolveTargetInEditor(selectedEmbeddedHeadingEditor, {
  prompt: enrichedSidebar,
  userPrompt: 'rewrite the capas sections',
  selection: selectedEmbeddedHeadingEditor.state.selection,
  targetScope: 'selection',
})
assert.match(classifierSelectionScope.text, /CAPAs/)
assert.match(classifierSelectionScope.text, /CAPA-IT-021/)
assert.match(classifierSelectionScope.text, /CAPA-IT-022/)
assert.doesNotMatch(classifierSelectionScope.text, /DEC-IT-001/)
assert.ok(classifierSelectionScope.text.length > embeddedCapaMarker.text.length)

const sop002CapaBlock = doc._blocks.find((block) => block.text.includes('CAPAs'))
const sop002Editor = {
  isDestroyed: false,
  state: {
    doc,
    selection: {
      from: sop002CapaBlock.pos + 1,
      to: sop002CapaBlock.pos + 1 + sop002CapaBlock.text.length,
      empty: false,
    },
  },
}
const capaSectionPrompt = resolveTargetInEditor(sop002Editor, {
  prompt: 'rewrite the capa section',
  userPrompt: 'rewrite the capa section',
  targetScope: 'section',
})
assert.match(capaSectionPrompt.text, /CAPA-IT-011/)
assert.match(capaSectionPrompt.text, /CAPA-IT-012/)
assert.ok(capaSectionPrompt.text.length > 200)
assert.ok(capaSectionPrompt.from < capaSectionPrompt.to)

const englishDoc = makeDoc([
  { text: 'SOP-QA-100 - Batch Release' },
  { text: 'Purpose', type: 'heading', attrs: { level: 2 } },
  { text: 'Define the controlled process for batch release and QA approval.' },
  { text: 'Scope', type: 'heading', attrs: { level: 2 } },
  { text: 'Applies to commercial batches released by Quality Assurance.' },
  { text: '2.1 Responsibilities', type: 'heading', attrs: { level: 3 } },
  { text: 'QA reviews the batch record and Manufacturing resolves open deviations.' },
  { text: 'Procedure', type: 'heading', attrs: { level: 2 } },
  { text: 'Review records, confirm acceptance criteria, and document the final decision.' },
])
const englishEditor = { isDestroyed: false, state: { doc: englishDoc } }

const purposeBodyBlock = englishDoc._blocks.find((block) => block.text.includes('batch release'))
const batchIndex = purposeBodyBlock.text.indexOf('batch')
const wordSelectionEditor = {
  isDestroyed: false,
  state: {
    doc: englishDoc,
    selection: {
      from: purposeBodyBlock.pos + 1 + batchIndex,
      to: purposeBodyBlock.pos + 1 + batchIndex + 'batch'.length,
      empty: false,
    },
  },
}
const selectedWord = resolveTargetInEditor(wordSelectionEditor, {
  prompt: 'rewrite selected word',
  userPrompt: 'rewrite selected word',
  selection: wordSelectionEditor.state.selection,
  targetScope: 'selection',
})
assert.equal(selectedWord.text, 'batch')
assert.equal(selectedWord.sectionType, 'Word')

const purpose = resolveTargetInEditor(englishEditor, {
  prompt: 'rewrite the Purpose section',
  userPrompt: 'rewrite the Purpose section',
  targetScope: 'section',
})
assert.match(purpose.text, /Purpose/)
assert.match(purpose.text, /batch release/)
assert.doesNotMatch(purpose.text, /Applies to commercial/)
assert.ok(purpose.confidence >= 0.58)

const scopeSummary = resolveTargetInEditor(englishEditor, {
  prompt: 'summarize the scope section',
  userPrompt: 'summarize the scope section',
  targetScope: 'section',
})
assert.match(scopeSummary.text, /Scope/)
assert.match(scopeSummary.text, /commercial batches/)
assert.doesNotMatch(scopeSummary.text, /QA reviews/)

const importedParagraphHeadingDoc = makeDoc([
  { text: 'Purpose', type: 'heading', attrs: { level: 2 } },
  { text: 'This SOP describes the documentation system.' },
  { text: 'Scope', type: 'heading', attrs: { level: 2 } },
  { text: 'This SOP applies to all employees of Company to ensure knowledge about documentation and Quality Management.' },
  { text: 'Responsibilities' },
  { text: 'Task Responsibility Decision to generate a new Controlled Document Author and QA Writing Author', type: 'table' },
  { text: 'Procedure: Generation, Review, Approval and Management of Controlled Documents' },
  { text: 'Procedure body.' },
])
const importedParagraphHeadingEditor = { isDestroyed: false, state: { doc: importedParagraphHeadingDoc } }
const importedScope = resolveTargetInEditor(importedParagraphHeadingEditor, {
  prompt: 'Rewrite the "Scope" section in the current SOP style.',
  userPrompt: 'Rewrite the "Scope" section in the current SOP style.',
  targetScope: 'section',
})
assert.match(importedScope.text, /Scope/)
assert.match(importedScope.text, /applies to all employees/)
assert.doesNotMatch(importedScope.text, /Responsibilities/)
assert.doesNotMatch(importedScope.text, /Decision to generate/)

const numberedSubsection = resolveTargetInEditor(englishEditor, {
  prompt: 'explain section 2.1',
  userPrompt: 'explain section 2.1',
  targetScope: 'section',
})
assert.match(numberedSubsection.text, /2\.1 Responsibilities/)
assert.match(numberedSubsection.text, /Manufacturing resolves/)

const followupSameSection = resolveTargetInEditor(englishEditor, {
  prompt: 'make it shorter',
  userPrompt: 'make it shorter',
  sectionHint: 'Procedure',
  targetScope: 'section',
})
assert.match(followupSameSection.text, /Procedure/)
assert.match(followupSameSection.text, /acceptance criteria/)

const sectionIndex = buildEditorSectionIndex(englishEditor)
assert.ok(sectionIndex.some((section) => section.sectionName === 'Purpose'))
assert.ok(sectionIndex.some((section) => section.sectionName === '2.1 Responsibilities'))

const tocDoc = makeDoc([
  { text: 'Table of Contents', type: 'heading', attrs: { level: 1 } },
  { text: '1. Document History 3' },
  { text: '2. Abbreviations and Definitions 4' },
  { text: '2.1 Abbreviations 4' },
  { text: '2.2 Definitions & Terms 4' },
  { text: 'Document History', type: 'heading', attrs: { level: 1 } },
  { text: 'Version | Date | Change' },
  { text: '2. Abbreviations and Definitions', type: 'heading', attrs: { level: 1 } },
  { text: '2.1 Abbreviations', type: 'heading', attrs: { level: 2 } },
  { text: 'Abbreviation | Meaning' },
  { text: 'QA | Quality Assurance' },
  { text: 'SOP | Standard Operating Procedure' },
  { text: '2.2 Definitions & Terms', type: 'heading', attrs: { level: 2 } },
  { text: 'Defined terms appear here.' },
])
const tocEditor = { isDestroyed: false, state: { doc: tocDoc } }
const abbreviations = resolveTargetInEditor(tocEditor, {
  prompt: 'rewrite the abbreviation section table',
  userPrompt: 'rewrite the abbreviation section table',
  targetScope: 'section',
})
assert.equal(abbreviations.sectionName, '2.1 Abbreviations')
assert.match(abbreviations.text, /QA \| Quality Assurance/)
assert.doesNotMatch(abbreviations.text, /Table of Contents/)

const definitionsOnly = resolveTargetInEditor(tocEditor, {
  prompt: 'rewrite the definitions section',
  userPrompt: 'rewrite the definitions section',
  targetScope: 'section',
})
assert.equal(definitionsOnly.sectionName, '2.2 Definitions & Terms')
assert.match(definitionsOnly.text, /Defined terms appear here/)
assert.doesNotMatch(definitionsOnly.text, /QA \| Quality Assurance/)
assert.doesNotMatch(definitionsOnly.text, /Abbreviations and Definitions/)

const tableNode = {
  isBlock: true,
  textContent: 'QP Qualified Person QA Quality Assurance CMO Contract Manufacturing Organization',
  type: { name: 'table' },
  attrs: {},
  nodeSize: 'QP Qualified Person QA Quality Assurance CMO Contract Manufacturing Organization'.length + 2,
}
const nestedTableParagraph = {
  isBlock: true,
  textContent: 'QP Qualified Person',
  type: { name: 'paragraph' },
  attrs: {},
  nodeSize: 21,
}
const tableDocBlocks = [
  { node: makeNode('2.1 Abbreviations', 'heading', { level: 2 }), pos: 0, text: '2.1 Abbreviations' },
  { node: tableNode, pos: 21, text: tableNode.textContent, nested: [{ node: nestedTableParagraph, pos: 29 }] },
  { node: makeNode('2.2 Definitions', 'heading', { level: 2 }), pos: 130, text: '2.2 Definitions' },
  { node: makeNode('Definition body.'), pos: 148, text: 'Definition body.' },
]
const tableDoc = {
  content: { size: 140 },
  descendants(callback) {
    for (const block of tableDocBlocks) {
      const descend = callback(block.node, block.pos)
      if (descend !== false && Array.isArray(block.nested)) {
        for (const nested of block.nested) callback(nested.node, nested.pos)
      }
    }
  },
  textBetween(from, to, separator = '\n') {
    const parts = []
    for (const block of tableDocBlocks) {
      const start = block.pos + 1
      const end = start + block.text.length
      const overlapStart = Math.max(from, start)
      const overlapEnd = Math.min(to, end)
      if (overlapEnd > overlapStart) {
        parts.push(block.text.slice(overlapStart - start, overlapEnd - start))
      }
    }
    return parts.join(separator)
  },
}
const tableEditor = { isDestroyed: false, state: { doc: tableDoc } }
const abbreviationTable = resolveTargetInEditor(tableEditor, {
  prompt: 'rewrite the abbreviation section table',
  userPrompt: 'rewrite the abbreviation section table',
  targetScope: 'section',
})
assert.match(abbreviationTable.text, /CMO Contract Manufacturing Organization/)
assert.doesNotMatch(abbreviationTable.text, /Definition body/)
assert.ok(abbreviationTable.to >= tableNode.nodeSize + 21)

const staleAbbreviationSelectionEditor = {
  isDestroyed: false,
  state: {
    doc: tableDoc,
    selection: {
      from: 1,
      to: '2.1 Abbreviations'.length + 1,
      empty: false,
    },
  },
}
const chatNamedAbbreviations = resolveTargetInEditor(staleAbbreviationSelectionEditor, {
  prompt: 'no improve the full Abbreviations section',
  userPrompt: 'no improve the full Abbreviations section',
  selection: staleAbbreviationSelectionEditor.state.selection,
  targetScope: 'selection',
  targetType: 'section',
  targetLabel: '2.1 Abbreviations',
})
assert.match(chatNamedAbbreviations.text, /CMO Contract Manufacturing Organization/)
assert.doesNotMatch(chatNamedAbbreviations.text, /Definition body/)
assert.ok(chatNamedAbbreviations.text.length > '2.1 Abbreviations'.length)

const qualificationBlocks = makeDoc([
  { text: 'Qualification of other GMP/GDP service suppliers', type: 'heading', attrs: { level: 2 } },
  { text: 'QA evaluates service supplier risk and qualification requirements.' },
  { text: 'Table 1 Initial qualification requirements service suppliers', type: 'heading', attrs: { level: 3 } },
  { text: 'Supplier risk Service risk Low Moderate High No audit required Questionnaire On site audit', type: 'table' },
  { text: 'Evaluation and requalification of GMP/GDP service suppliers', type: 'heading', attrs: { level: 2 } },
  { text: 'Evaluation body.' },
])
const qualificationEditor = { isDestroyed: false, state: { doc: qualificationBlocks } }
const qualificationTableOnly = resolveTargetInEditor(qualificationEditor, {
  prompt: 'rewrite Table 1 Initial qualification requirements service suppliers',
  userPrompt: 'rewrite Table 1 Initial qualification requirements service suppliers',
  targetScope: 'section',
})
assert.equal(qualificationTableOnly.sectionType, 'Table')

const qualificationSection = resolveTargetInEditor(qualificationEditor, {
  prompt: 'rewrite the Table 1 Initial qualification requirements service suppliers section',
  userPrompt: 'rewrite the Table 1 Initial qualification requirements service suppliers section',
  targetScope: 'section',
})
assert.equal(qualificationSection.sectionName, 'Qualification of other GMP/GDP service suppliers')
assert.match(qualificationSection.text, /QA evaluates service supplier risk/)
assert.match(qualificationSection.text, /Supplier risk Service risk/)
assert.doesNotMatch(qualificationSection.text, /Evaluation body/)

const llmTableOnly = resolveTargetInEditor(qualificationEditor, {
  prompt: 'rewrite this',
  userPrompt: 'rewrite this',
  targetScope: 'table',
  targetType: 'table',
  targetLabel: 'Table 1 Initial qualification requirements service suppliers',
})
assert.equal(llmTableOnly.sectionType, 'Table')
assert.match(llmTableOnly.text, /Supplier risk Service risk/)

const llmTableSection = resolveTargetInEditor(qualificationEditor, {
  prompt: 'rewrite this',
  userPrompt: 'rewrite this',
  targetScope: 'table_section',
  targetType: 'table_section',
  targetLabel: 'Table 1 Initial qualification requirements service suppliers',
  owningSection: 'Qualification of other GMP/GDP service suppliers',
})
assert.equal(llmTableSection.sectionName, 'Qualification of other GMP/GDP service suppliers')
assert.match(llmTableSection.text, /QA evaluates service supplier risk/)
assert.doesNotMatch(llmTableSection.text, /Evaluation body/)

const llmDefinitions = resolveTargetInEditor(tocEditor, {
  prompt: 'rewrite this',
  userPrompt: 'rewrite this',
  targetScope: 'section',
  targetType: 'section',
  targetLabel: '2.2 Definitions & Terms',
})
assert.equal(llmDefinitions.sectionName, '2.2 Definitions & Terms')
assert.doesNotMatch(llmDefinitions.text, /QA \| Quality Assurance/)

const ambiguousDoc = makeDoc([
  { text: 'Risk Assessment', type: 'heading', attrs: { level: 2 } },
  { text: 'Risk assessment body.' },
  { text: 'Risk Analysis', type: 'heading', attrs: { level: 2 } },
  { text: 'Risk analysis body.' },
])
const ambiguousEditor = { isDestroyed: false, state: { doc: ambiguousDoc } }
assert.throws(
  () => resolveTargetInEditor(ambiguousEditor, {
    prompt: 'rewrite the risk section',
    userPrompt: 'rewrite the risk section',
    targetScope: 'section',
  }),
  /multiple matching sections|exact heading/i,
)

assert.throws(
  () => resolveTargetInEditor(englishEditor, {
    prompt: 'rewrite the validation section',
    userPrompt: 'rewrite the validation section',
    targetScope: 'section',
  }),
  /could not find that section/i,
)

const agentSelectedWord = TargetResolverAgent.resolve({
  editor: wordSelectionEditor,
  action: 'rewrite',
  userQuery: 'rewrite selected word',
  prompt: 'rewrite selected word',
  userPrompt: 'rewrite selected word',
  selection: wordSelectionEditor.state.selection,
  targetScope: 'selection',
})
assert.equal(agentSelectedWord.target.text, 'batch')
assert.equal(agentSelectedWord.resolution.scope, 'selection')
assert.equal(agentSelectedWord.resolution.needs_clarification, false)

const agentStaleSelection = TargetResolverAgent.resolve({
  editor: wordSelectionEditor,
  action: 'rewrite',
  userQuery: 'rewrite the Purpose section',
  prompt: 'rewrite the Purpose section',
  userPrompt: 'rewrite the Purpose section',
  selection: wordSelectionEditor.state.selection,
  targetScope: 'selection',
  targetType: 'selection',
  targetAnalysis: { target_type: 'selection', target_label: 'Purpose', confidence: 0.7 },
})
assert.match(agentStaleSelection.target.text, /Purpose/)
assert.match(agentStaleSelection.target.text, /batch release/)
assert.notEqual(agentStaleSelection.target.text, 'batch')
assert.equal(agentStaleSelection.resolution.scope, 'section')
assert.equal(agentStaleSelection.resolution.needs_clarification, false)

const agentCurrentSopStyleSection = TargetResolverAgent.resolve({
  editor: staleAbbreviationSelectionEditor,
  action: 'rewrite',
  userQuery: 'rewrite the Abbreviations and Definitions Abbreviations section in the current SOP style',
  prompt: 'rewrite the Abbreviations and Definitions Abbreviations section in the current SOP style',
  userPrompt: 'rewrite the Abbreviations and Definitions Abbreviations section in the current SOP style',
  selection: staleAbbreviationSelectionEditor.state.selection,
  targetScope: 'full_document',
  targetType: 'full_document',
  targetAnalysis: { target_type: 'full_document', target_label: 'Abbreviations', confidence: 0.64 },
})
assert.equal(agentCurrentSopStyleSection.resolution.scope, 'section')
assert.equal(agentCurrentSopStyleSection.target.isFullDoc, false)
assert.match(agentCurrentSopStyleSection.target.text, /CMO Contract Manufacturing Organization/)
assert.doesNotMatch(agentCurrentSopStyleSection.target.text, /Definition body/)

const agentSectionHintOverridesFullScope = TargetResolverAgent.resolve({
  editor: staleAbbreviationSelectionEditor,
  action: 'rewrite',
  userQuery: 'rewrite in the current SOP style',
  prompt: 'rewrite in the current SOP style',
  userPrompt: 'rewrite in the current SOP style',
  selection: staleAbbreviationSelectionEditor.state.selection,
  sectionHint: '2.1 Abbreviations',
  targetScope: 'full_document',
  targetType: 'full_document',
  targetAnalysis: { target_type: 'full_document', target_label: '', confidence: 0.62 },
})
assert.equal(agentSectionHintOverridesFullScope.resolution.scope, 'section')
assert.equal(agentSectionHintOverridesFullScope.target.isFullDoc, false)
assert.match(agentSectionHintOverridesFullScope.target.text, /CMO Contract Manufacturing Organization/)
assert.doesNotMatch(agentSectionHintOverridesFullScope.target.text, /Definition body/)

const agentImportedScope = TargetResolverAgent.resolve({
  editor: importedParagraphHeadingEditor,
  action: 'improve',
  userQuery: 'improve Scope',
  prompt: 'improve Scope',
  userPrompt: 'improve Scope',
  targetScope: 'section',
})
assert.match(agentImportedScope.target.text, /Scope/)
assert.match(agentImportedScope.target.text, /applies to all employees/)
assert.doesNotMatch(agentImportedScope.target.text, /Responsibilities/)
assert.equal(agentImportedScope.resolution.contains_table, false)

const agentTable = TargetResolverAgent.resolve({
  editor: qualificationEditor,
  action: 'rewrite',
  userQuery: 'rewrite Table 1 Initial qualification requirements service suppliers',
  prompt: 'rewrite Table 1 Initial qualification requirements service suppliers',
  userPrompt: 'rewrite Table 1 Initial qualification requirements service suppliers',
  targetScope: 'table',
})
assert.equal(agentTable.target.sectionType, 'Table')
assert.equal(agentTable.resolution.scope, 'table')
assert.equal(agentTable.resolution.contains_table, true)

const documentHistoryTableDoc = makeDoc([
  { text: 'Purpose', type: 'heading', attrs: { level: 2 } },
  { text: 'Purpose body.' },
  { text: 'Scope', type: 'heading', attrs: { level: 2 } },
  { text: 'Scope body mentions documents and history.' },
  { text: 'Document History', type: 'heading', attrs: { level: 2 } },
  { text: 'Version Effective Since Replaces Version Description of Change 01 2024-01-01 n.a. Initial version', type: 'table' },
  { text: 'Abbreviations', type: 'heading', attrs: { level: 2 } },
  { text: 'QA Quality Assurance' },
])
const documentHistoryTableEditor = { isDestroyed: false, state: { doc: documentHistoryTableDoc } }
const agentDocumentHistoryTable = TargetResolverAgent.resolve({
  editor: documentHistoryTableEditor,
  action: 'improve',
  userQuery: 'improve the document history table',
  prompt: 'improve the document history table',
  userPrompt: 'improve the document history table',
  targetScope: 'table',
  targetAnalysis: { target_type: 'section', target_label: 'Scope', confidence: 0.71 },
})
assert.equal(agentDocumentHistoryTable.target.sectionType, 'Table')
assert.equal(agentDocumentHistoryTable.resolution.scope, 'table')
assert.match(agentDocumentHistoryTable.target.text, /Description of Change/)
assert.doesNotMatch(agentDocumentHistoryTable.target.text, /Scope body/)

const agentDocumentHistoryTableById = TargetResolverAgent.resolve({
  editor: documentHistoryTableEditor,
  action: 'improve',
  userQuery: 'improve the document history table',
  prompt: 'improve the document history table',
  userPrompt: 'improve the document history table',
  targetScope: 'table',
  targetType: 'table',
  targetId: 'table-1',
  targetAnalysis: {
    target_type: 'table',
    target_id: 'table-1',
    target_label: 'Scope',
    confidence: 0.91,
    source: 'deep_agent_target_resolver',
  },
})
assert.equal(agentDocumentHistoryTableById.target.sectionType, 'Table')
assert.equal(agentDocumentHistoryTableById.resolution.target_id, 'table-1')
assert.match(agentDocumentHistoryTableById.target.text, /Description of Change/)
assert.doesNotMatch(agentDocumentHistoryTableById.target.text, /Scope body/)

const parsedDocumentHistory = parseSopDocument(documentHistoryTableEditor)
assert.ok(parsedDocumentHistory.schema.includes('[table:table-1]'))
assert.equal(parsedDocumentHistory.nodes.find((node) => node.id === 'table-1')?.parent_id, 'section-3')

const stableIdDoc = makeDoc([
  { text: 'Purpose', type: 'heading', attrs: { id: 'heading-purpose' } },
  { text: 'Purpose body.', attrs: { id: 'paragraph-purpose' } },
  { text: 'Change History', type: 'heading', attrs: { id: 'heading-history' } },
  { text: 'Version Date Change 1 2024 Initial', type: 'table', attrs: { id: 'table-history' } },
])
const stableIdEditor = { isDestroyed: false, state: { doc: stableIdDoc } }
const parsedStableIds = parseSopDocument(stableIdEditor)
assert.ok(parsedStableIds.nodes.some((node) => node.id === 'heading-purpose' && node.type === 'section'))
assert.ok(parsedStableIds.nodes.some((node) => node.id === 'paragraph-purpose' && node.parent_id === 'heading-purpose'))
assert.ok(parsedStableIds.nodes.some((node) => node.id === 'table-history' && node.parent_id === 'heading-history'))

const mixedSectionDoc = makeDoc([
  { text: '4. Definitions', type: 'heading', attrs: { id: 'heading-definitions', level: 2 } },
  { text: 'Definitions introduction.', attrs: { id: 'paragraph-definitions' } },
  { text: 'Term Meaning SOP Standard operating procedure', type: 'table', attrs: { id: 'table-definitions' } },
  { text: '5. Scope', type: 'heading', attrs: { id: 'heading-scope', level: 2 } },
  { text: 'Scope body.', attrs: { id: 'paragraph-scope' } },
])
const mixedHeading = mixedSectionDoc._blocks[0]
const mixedParagraph = mixedSectionDoc._blocks[1]
const mixedSectionEditor = { isDestroyed: false, state: { doc: mixedSectionDoc } }
const headingOnlySelection = {
  from: mixedHeading.pos + 1,
  to: mixedHeading.pos + 1 + mixedHeading.text.length,
  text: mixedHeading.text,
  empty: false,
}
const selectedMixedSection = resolveTargetInEditor(mixedSectionEditor, {
  prompt: 'rewrite the selected section',
  userPrompt: 'rewrite the selected section',
  selection: headingOnlySelection,
  targetScope: 'selection',
  targetType: 'selection',
  preferFullSection: true,
})
assert.equal(selectedMixedSection.contains_table, true)
assert.equal(selectedMixedSection.sectionType, 'Table Section')
assert.equal(selectedMixedSection.resolved_target_type, 'table_section')
assert.match(selectedMixedSection.text, /Definitions introduction/)
assert.match(selectedMixedSection.text, /Standard operating procedure/)
assert.doesNotMatch(selectedMixedSection.text, /5\. Scope/)

const paragraphOnlySelection = resolveTargetInEditor(mixedSectionEditor, {
  prompt: 'rewrite the selected text',
  userPrompt: 'rewrite the selected text',
  selection: {
    from: mixedParagraph.pos + 1,
    to: mixedParagraph.pos + 1 + mixedParagraph.text.length,
    text: mixedParagraph.text,
    empty: false,
  },
  targetScope: 'selection',
  targetType: 'selection',
  preferFullSection: false,
})
assert.equal(paragraphOnlySelection.contains_table, false)
assert.equal(paragraphOnlySelection.resolved_target_type, 'selection')
assert.equal(paragraphOnlySelection.text, 'Definitions introduction.')

const selectedMixedSectionByAgent = TargetResolverAgent.resolve({
  editor: mixedSectionEditor,
  action: 'rewrite',
  userQuery: 'rewrite the selected section',
  prompt: 'rewrite the selected section',
  userPrompt: 'rewrite the selected section',
  selection: headingOnlySelection,
  targetScope: 'selection',
  targetType: 'selection',
  targetId: 'selection',
  targetAnalysis: {
    target_type: 'selection',
    target_id: 'selection',
    confidence: 0.99,
    requires_clarification: false,
  },
})
assert.equal(selectedMixedSectionByAgent.target.contains_table, true)
assert.equal(selectedMixedSectionByAgent.resolution.contains_table, true)
assert.equal(selectedMixedSectionByAgent.resolution.target_type, 'table_section')

const invalidDeepAgentTableId = TargetResolverAgent.resolve({
  editor: documentHistoryTableEditor,
  action: 'improve',
  userQuery: 'improve the document history table',
  prompt: 'improve the document history table',
  userPrompt: 'improve the document history table',
  targetScope: 'table',
  targetType: 'table',
  targetId: 'missing-table-id',
  targetAnalysis: {
    target_type: 'table',
    target_id: 'missing-table-id',
    target_label: 'Scope',
    confidence: 0.91,
    source: 'deep_agent_target_resolver',
  },
})
assert.equal(invalidDeepAgentTableId.resolution.needs_clarification, true)
assert.ok(invalidDeepAgentTableId.resolution.candidate_targets.every((candidate) => candidate.type === 'table'))
assert.ok(!invalidDeepAgentTableId.resolution.candidate_targets.some((candidate) => candidate.label === 'Scope'))

const highConfidenceWrongTypeId = TargetResolverAgent.resolve({
  editor: documentHistoryTableEditor,
  action: 'rewrite',
  userQuery: 'rewrite the Scope section',
  prompt: 'rewrite the Scope section',
  userPrompt: 'rewrite the Scope section',
  targetScope: 'section',
  targetType: 'section',
  targetId: 'table-1',
  targetAnalysis: {
    target_type: 'section',
    target_id: 'table-1',
    target_label: 'Scope',
    confidence: 0.99,
  },
})
assert.equal(highConfidenceWrongTypeId.resolution.needs_clarification, true)
assert.equal(highConfidenceWrongTypeId.resolution.code, 'deep_agent_target_type_mismatch')
assert.ok(highConfidenceWrongTypeId.resolution.candidate_targets.every((candidate) => candidate.type === 'section'))

const agentColumnIntentTargetsTable = TargetResolverAgent.resolve({
  editor: documentHistoryTableEditor,
  action: 'rewrite',
  userQuery: 'rewrite column Description of Change in Document History',
  prompt: 'rewrite column Description of Change in Document History',
  userPrompt: 'rewrite column Description of Change in Document History',
  targetScope: 'table',
})
assert.equal(agentColumnIntentTargetsTable.target.sectionType, 'Table')
assert.equal(agentColumnIntentTargetsTable.resolution.scope, 'table')
assert.match(agentColumnIntentTargetsTable.target.text, /Description of Change/)

const currentTableRowNeedsClarification = TargetResolverAgent.resolve({
  editor: documentHistoryTableEditor,
  action: 'improve',
  userQuery: 'improve the current table row',
  prompt: 'improve the current table row',
  userPrompt: 'improve the current table row',
  targetScope: 'table',
  targetType: 'table',
  targetAnalysis: {
    target_type: 'table',
    target_id: 'table-1',
    target_label: 'Document History',
    confidence: 0.91,
  },
})
assert.equal(currentTableRowNeedsClarification.resolution.needs_clarification, true)
assert.equal(currentTableRowNeedsClarification.resolution.code, 'table_span_requires_specific_target')
assert.ok(currentTableRowNeedsClarification.resolution.candidate_targets.every((candidate) => candidate.type === 'table'))

const responsibilitiesSummary = TargetResolverAgent.resolve({
  editor: englishEditor,
  action: 'summarize',
  userQuery: 'summarize the Responsibilities section',
  prompt: 'summarize the Responsibilities section',
  userPrompt: 'summarize the Responsibilities section',
  targetScope: 'section',
})
assert.match(responsibilitiesSummary.target.text, /2\.1 Responsibilities/)
assert.match(responsibilitiesSummary.target.text, /Manufacturing resolves/)
assert.doesNotMatch(responsibilitiesSummary.target.text, /Review records/)

const selectedSectionGapCheck = TargetResolverAgent.resolve({
  editor: zweckSelectionEditor,
  action: 'gap_check',
  userQuery: 'gap check the selected section',
  prompt: 'gap check the selected section',
  userPrompt: 'gap check the selected section',
  selection: zweckSelectionEditor.state.selection,
  targetScope: 'selection',
  targetType: 'selection',
  targetAnalysis: { target_type: 'selection', target_id: 'selection', confidence: 0.92 },
})
assert.equal(selectedSectionGapCheck.resolution.scope, 'selection')
assert.match(selectedSectionGapCheck.target.text, /1\. Zweck/)

const fullSopRewrite = TargetResolverAgent.resolve({
  editor: englishEditor,
  action: 'rewrite',
  userQuery: 'rewrite full SOP',
  prompt: 'rewrite full SOP',
  userPrompt: 'rewrite full SOP',
  targetScope: 'full_document',
  targetType: 'full_document',
  targetAnalysis: { target_type: 'full_document', target_id: 'doc_root', confidence: 0.98 },
})
assert.equal(fullSopRewrite.target.isFullDoc, true)
assert.equal(fullSopRewrite.resolution.target_id, 'doc_root')
assert.match(fullSopRewrite.target.text, /Batch Release/)

const agentAmbiguous = TargetResolverAgent.resolve({
  editor: ambiguousEditor,
  action: 'rewrite',
  userQuery: 'rewrite the risk section',
  prompt: 'rewrite the risk section',
  userPrompt: 'rewrite the risk section',
  targetScope: 'section',
})
assert.equal(agentAmbiguous.resolution.needs_clarification, true)
assert.ok(agentAmbiguous.resolution.candidate_targets.length >= 2)

console.log(JSON.stringify({
  ok: true,
  capa: { sectionName: capa.sectionName, chars: capa.text.length },
  zweck: { sectionName: zweck.sectionName, chars: zweck.text.length },
  decisions: { sectionName: decisions.sectionName, chars: decisions.text.length },
  embeddedCapa: { sectionName: embeddedCapa.sectionName, chars: embeddedCapa.text.length },
  purpose: { sectionName: purpose.sectionName, chars: purpose.text.length },
  scopeSummary: { sectionName: scopeSummary.sectionName, chars: scopeSummary.text.length },
  numberedSubsection: { sectionName: numberedSubsection.sectionName, chars: numberedSubsection.text.length },
  abbreviations: { sectionName: abbreviations.sectionName, chars: abbreviations.text.length },
}, null, 2))
