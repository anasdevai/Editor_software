const ACTION_PATTERNS = [
  ['gap_check', /\b(?:gap\s*check|gap\s*analysis|compliance\s+(?:check|review)|l(?:u|ue|ü)cken(?:analyse|check|pr(?:u|ue|ü)fung)?)\b/i],
  ['summarize', /\b(?:summari[sz]e|summary|fasse\s+zusammen|zusammenfass|kurzfass)\b/i],
  ['improve', /\b(?:improve|enhance|optimi[sz]e|verbessern?|überarbeiten|ueberarbeiten)\b/i],
  ['rewrite', /\b(?:rewrite|re-?write|rephrase|redraft|umschreiben|neu\s+formulieren)\b/i],
  ['expand', /\b(?:expand|extend|add\s+detail|ausf(?:u|ue|ü)hrlicher)\b/i],
  ['shorten', /\b(?:shorten|shorter|concise|compress|verk(?:u|ue|ü)rzen|k(?:u|ue|ü)rzer)\b/i],
  ['fix_grammar', /\b(?:fix\s+grammar|grammar|proofread|spelling|rechtschreibung|grammatik)\b/i],
  ['explain', /\b(?:explain|describe|what\s+does|erkl(?:a|ae|ä)r)\b/i],
]

const FULL_DOCUMENT_RE =
  /\b(?:full|whole|entire|complete)\s+(?:sop|document|doc)\b|\b(?:rewrite|improve|summari[sz]e|gap\s*check)\s+(?:this\s+|the\s+|current\s+)?(?:sop|document|doc)\b/i

const EXPLICIT_SELECTION_RE =
  /\b(?:selected\s+(?:text|word|paragraph|section|sentence|line|content)|selection|highlighted|marked\s+text|this\s+selection)\b/i

const TABLE_RE = /\b(?:table|tabelle|tabular|matrix)\b/i
const SECTION_RE = /\b(?:section|sections|heading|abschnitt|kapitel)\b/i
const PARAGRAPH_RE = /\b(?:paragraph|para|absatz)\b/i
const SENTENCE_RE = /\b(?:sentence|satz)\b/i
const WORD_RE = /\b(?:word|term|phrase|wort|begriff)\b/i

export function classifyTargetIntent(userQuery = '', hints = {}) {
  const text = String(userQuery || '').trim()
  const lower = text.toLowerCase()

  const primary = ACTION_PATTERNS.find(([, pattern]) => pattern.test(text))
  const action = hints.action || primary?.[0] || 'rewrite'

  const namesTable = TABLE_RE.test(text)
  const namesSection = SECTION_RE.test(text) || /\b\d+(?:\.\d+)+\b/.test(text)
  const namesFullDocument = FULL_DOCUMENT_RE.test(text) && !namesTable && !namesSection

  let scope = hints.targetScope || 'unknown'
  if (namesFullDocument) scope = 'full_document'
  else if (EXPLICIT_SELECTION_RE.test(text)) scope = 'selection'
  else if (namesTable) scope = SECTION_RE.test(text) ? 'table_section' : 'table'
  else if (namesSection) scope = 'section'
  else if (PARAGRAPH_RE.test(text)) scope = 'paragraph'
  else if (SENTENCE_RE.test(text)) scope = 'sentence'
  else if (WORD_RE.test(text)) scope = 'word'

  const lengthConstraint = (() => {
    const lines = text.match(/\b(?:in|as|to)\s+(\d{1,2})\s+lines?\b/i) || text.match(/\b(\d{1,2})\s+lines?\b/i)
    if (lines?.[1]) return { type: 'lines', value: Number(lines[1]) }
    if (/\b(?:shorter|shorten|concise|brief|k(?:u|ue|ü)rzer)\b/i.test(text)) return { type: 'relative', value: 'shorter' }
    if (/\b(?:detailed|more detail|ausf(?:u|ue|ü)hrlich)\b/i.test(text)) return { type: 'relative', value: 'detailed' }
    return null
  })()

  const styleConstraint = (() => {
    const clientProfile = text.match(/\b(?:client|profile)\s*([A-Za-z0-9_-]+)\s+(?:style|profile)\b/i)
    if (clientProfile?.[1]) return { profile_hint: clientProfile[1], style_only: true }
    if (/\bformal\b/i.test(text)) return { tone: 'formal' }
    if (/\bsimpl(?:e|er|ify)\b/i.test(text)) return { tone: 'simple' }
    return null
  })()

  const evidenceRequirement =
    action === 'gap_check'
    || /\b(?:rag|evidence|source|citation|compliance|regulatory|compare|generation|generate)\b/i.test(lower)

  return {
    primary_action: action,
    intent: action,
    scope,
    target_scope: scope,
    length_constraint: lengthConstraint,
    style_constraint: styleConstraint,
    evidence_requirement: evidenceRequirement,
    explicit_selection: EXPLICIT_SELECTION_RE.test(text),
    names_table: namesTable,
    names_section: namesSection,
    names_full_document: namesFullDocument,
  }
}

export default classifyTargetIntent
