import assert from 'node:assert/strict'
import { captureEditorSelectionForAction } from '../frontend/src/utils/editScopeInference.js'
import { buildEditorSectionIndex, resolveTargetInEditor } from '../frontend/src/utils/editorTargetResolver.js'

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
assert.doesNotMatch(classifierSelectionScope.text, /CAPA-IT-021/)
assert.ok(classifierSelectionScope.text.length <= embeddedCapaMarker.text.length)

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

console.log(JSON.stringify({
  ok: true,
  capa: { sectionName: capa.sectionName, chars: capa.text.length },
  zweck: { sectionName: zweck.sectionName, chars: zweck.text.length },
  decisions: { sectionName: decisions.sectionName, chars: decisions.text.length },
  embeddedCapa: { sectionName: embeddedCapa.sectionName, chars: embeddedCapa.text.length },
  purpose: { sectionName: purpose.sectionName, chars: purpose.text.length },
  scopeSummary: { sectionName: scopeSummary.sectionName, chars: scopeSummary.text.length },
  numberedSubsection: { sectionName: numberedSubsection.sectionName, chars: numberedSubsection.text.length },
}, null, 2))
