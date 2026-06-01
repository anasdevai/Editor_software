import { queryAI } from '../api/editorApi'
import { getKLAssistantContext } from './assistantContext'

/** Persisted interaction mode for the KL/KI sidebar assistant (global across routes). */
const KL_ASSISTANT_MODE_LS_KEY = 'cybrain_kl_assistant_mode_v1'

export function readKlAssistantMode() {
  if (typeof window === 'undefined') return 'action'
  try {
    const v = localStorage.getItem(KL_ASSISTANT_MODE_LS_KEY)
    if (v === 'query' || v === 'action') return v
  } catch {
    // ignore
  }
  return 'action'
}

export function writeKlAssistantMode(mode) {
  if (typeof window === 'undefined') return
  try {
    if (mode === 'query' || mode === 'action') {
      localStorage.setItem(KL_ASSISTANT_MODE_LS_KEY, mode)
    }
  } catch {
    // ignore
  }
}

const ROUTE_CONFIG = {
  '/sops': {
    category: 'sops',
    contextLabel: 'Kontext: SOP-Ansicht',
    suggestions: [
      'Welche SOP ist besonders relevant?',
      'Gab es Audit-Bezug?',
      'Was war die letzte Abweichung?',
      'Zusammenfassung letzter Woche',
    ],
  },
  '/deviations': {
    category: 'deviations',
    contextLabel: 'Kontext: Abweichungen',
    suggestions: [
      'Welche Abweichung ist kritisch?',
      'Zeige offene Abweichungen mit Impact',
      'Gibt es verknuepfte SOPs?',
      'Welche CAPA ist ueberfaellig?',
    ],
  },
  '/capa': {
    category: 'capas',
    contextLabel: 'Kontext: CAPA',
    suggestions: [
      'Welche CAPA ist am dringendsten?',
      'Welche CAPA ist noch offen?',
      'Welche CAPA ist mit Audits verknuepft?',
      'Was ist die naechste Eskalation?',
    ],
  },
  '/audits': {
    category: 'audits',
    contextLabel: 'Kontext: Audit Findings',
    suggestions: [
      'Welche Findings sind offen?',
      'Welche Findings sind kritisch?',
      'Zeige Audit-zu-SOP Bezug',
      'Welche Findings brauchen CAPA?',
    ],
  },
  '/decisions': {
    category: 'decisions',
    contextLabel: 'Kontext: Entscheidungen',
    suggestions: [
      'Welche Entscheidung ist zuletzt getroffen worden?',
      'Welche Entscheidung ist noch offen?',
      'Welche SOP ist davon betroffen?',
      'Zeige begruendete Entscheidungen',
    ],
  },
  '/knowledge': {
    category: undefined,
    contextLabel: 'Kontext: Wissenssuche',
    suggestions: [
      'Zeige relevante SOPs zum Thema',
      'Welche Quellen stuetzen das?',
      'Welche Risiken sind erkennbar?',
      'Fasse den Kontext kurz zusammen',
    ],
  },
  '/editor': {
    category: 'sops',
    contextLabel: 'Kontext: SOP Editor',
    suggestions: [
      'Analysiere den aktuellen SOP-Kontext',
      'Welche Verbesserungen sind noetig?',
      'Pruefe auf Compliance-Luecken',
      'Fasse den SOP-Inhalt zusammen',
    ],
  },
  '/profiles': {
    category: 'sops',
    contextLabel: 'Kontext: Profile Workspace',
    suggestions: [
      'Welche Profilversion ist aktiv?',
      'Wofuer wird dieses Profil verwendet?',
      'Welche Aenderungen gab es im Profil?',
      'Wie wirkt sich das Profil auf Rewrite aus?',
    ],
  },
}

const DEFAULT_CONFIG = {
  category: undefined,
  contextLabel: 'Kontext: Keine Analyse, Startscreen',
  suggestions: [
    'Welche SOP ist besonders relevant?',
    'Gab es Audit-Bezug?',
    'Was war die letzte Abweichung?',
    'Zusammenfassung letzter Woche',
  ],
}

export function nowTime() {
  return new Date().toLocaleTimeString('de-DE', { hour: '2-digit', minute: '2-digit' })
}

export function formatChatTimeFromIso(iso) {
  if (!iso) return nowTime()
  try {
    const d = new Date(iso)
    if (Number.isNaN(d.getTime())) return nowTime()
    return d.toLocaleTimeString('de-DE', { hour: '2-digit', minute: '2-digit' })
  } catch {
    return nowTime()
  }
}

/** Remove backend retrieval/system markers before rendering assistant text. */
export function sanitizeAnswerForDisplay(text) {
  let s = String(text || '').trim()
  if (!s) return ''

  const blockMarkers = ['---SUGGESTIONS---', '---CITATIONS---', 'LIVE_ASSISTANT_CONTEXT']
  for (const m of blockMarkers) {
    const idx = s.toLowerCase().indexOf(m.toLowerCase())
    if (idx >= 0) {
      const rest = s.slice(idx + m.length)
      const nextPara = rest.search(/\n\n(?:Summary|Details|Status|The |Die |Der )/i)
      s = (s.slice(0, idx) + (nextPara >= 0 ? rest.slice(nextPara) : '')).trim()
    }
  }

  const bracketedNames = ['LIVE_ASSISTANT_CONTEXT', 'LIVE_ASSISTANT', 'editor_context', 'RETRIEVED CONTEXT']
  for (const name of bracketedNames) {
    s = s.replace(new RegExp(`\\[\\s*${name}\\s*\\]`, 'gi'), ' ')
    s = s.replace(new RegExp(`\\b${name}\\b`, 'gi'), ' ')
  }

  const inlinePatterns = [
    /\s*\[REASONING\]\s*/gi,
    /\s*\[CONFIDENCE\]\s*/gi,
    /\s*\[ANSWER\]\s*/gi,
    /\bSCOPE=ACTIVE_SOP_ONLY\b/gi,
    /\bACTIVE_SOP_ID=[0-9a-fA-F-]{8,}\b/gi,
    /\bRAG_HINTS:\s*[^\n\]]*/gi,
    /\bPLANNED_ASSISTANT_ACTION:\s*[^\n\]]*/gi,
    /\s*\[?\s*Retrieval scope:\s*[^\n\]]+\]?\s*/gi,
  ]
  for (const re of inlinePatterns) {
    s = s.replace(re, ' ')
  }

  const dropLine = (line) => {
    const low = line.trim().toLowerCase()
    return (
      low.startsWith('live_assistant_context')
      || low.startsWith('- retrieval scope:')
      || low.startsWith('- active sop:')
      || low.startsWith('- linked ')
      || low.startsWith('rag_hints:')
    )
  }
  s = s
    .split(/\r?\n/)
    .filter((line) => !dropLine(line))
    .join('\n')
    .replace(/\s{2,}/g, ' ')
    .replace(/\n{3,}/g, '\n\n')
    .trim()

  return s
}

export function toHtml(text) {
  if (!text) return '<p></p>'
  const raw = sanitizeAnswerForDisplay(String(text || '').trim())
  const escaped = raw
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')

  const lines = escaped
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)

  const sectionHeading = (line) =>
    /^(summary|details|status|cross-refs|cross refs|sources|references)\s*:/i.test(line)
  const bulletLine = (line) => /^[-*•]\s+/.test(line) || /^\d+[.)]\s+/.test(line)

  const parts = []
  let i = 0
  while (i < lines.length) {
    const line = lines[i]
    if (sectionHeading(line)) {
      const [labelRaw, firstBodyRaw = ''] = line.split(/:\s*/, 2)
      const label = labelRaw.trim()
      const firstBody = firstBodyRaw.trim()
      const bullets = []
      const paras = []
      if (firstBody) paras.push(firstBody)

      let j = i + 1
      while (j < lines.length && !sectionHeading(lines[j])) {
        const row = lines[j]
        if (bulletLine(row)) bullets.push(row.replace(/^([-*•]|\d+[.)])\s+/, '').trim())
        else paras.push(row)
        j += 1
      }

      parts.push(`<h4>${label}</h4>`)
      paras.forEach((p) => parts.push(`<p>${p}</p>`))
      if (bullets.length) {
        parts.push(`<ul>${bullets.map((b) => `<li>${b}</li>`).join('')}</ul>`)
      }
      i = j
      continue
    }

    if (bulletLine(line)) {
      const bullets = []
      let j = i
      while (j < lines.length && bulletLine(lines[j])) {
        bullets.push(lines[j].replace(/^([-*•]|\d+[.)])\s+/, '').trim())
        j += 1
      }
      parts.push(`<ul>${bullets.map((b) => `<li>${b}</li>`).join('')}</ul>`)
      i = j
      continue
    }

    parts.push(`<p>${line}</p>`)
    i += 1
  }

  return parts.join('')
}

export function stripHtml(html) {
  return String(html || '')
    .replace(/<br\s*\/?>/gi, '\n')
    .replace(/<[^>]+>/g, '')
    .trim()
}

function normalizeHistoryRows(chatHistory = []) {
  if (!Array.isArray(chatHistory)) return []
  return chatHistory
    .slice(-16)
    .map((row) => ({
      role: row?.role === 'assistant' || row?.role === 'ai' ? 'assistant' : 'user',
      content: sanitizeAnswerForDisplay(String(row?.content || row?.text || '').slice(0, 2400)),
    }))
    .filter((row) => row.content)
}

function buildEditorContextContract({
  sessionId,
  visibleQuestion,
  assistantContext,
  chatHistory,
}) {
  const current = assistantContext?.current_sop || {}
  const selected = assistantContext?.selected_section || {}
  const last = assistantContext?.last_action || null
  const focus = assistantContext?.last_focus || null
  return {
    session_id: sessionId || null,
    user_message: visibleQuestion,
    sop_context: {
      sop_id: current.id || assistantContext?.active_sop_id || null,
      title: current.title || '',
      version: current.version || '',
      owner: current.owner || '',
      status: current.status || '',
      created_at: current.created_at || null,
      updated_at: current.updated_at || assistantContext?.context_updated_at || null,
      tags: Array.isArray(current.tags) ? current.tags : [],
      sections: Array.isArray(current.sections) ? current.sections : [],
      full_text: current.full_text || assistantContext?.editor_excerpt || '',
      word_count: current.word_count || 0,
      compliance_standards: Array.isArray(current.compliance_standards) ? current.compliance_standards : [],
    },
    selected_section: {
      id: selected.id || null,
      label: selected.label || selected.name || null,
      content: selected.content || selected.text_excerpt || null,
    },
    conversation_history: normalizeHistoryRows(
      chatHistory.length
        ? chatHistory
        : assistantContext?.conversation_history || [],
    ),
    active_scope: assistantContext?.active_scope || null,
    instruction_memory: Array.isArray(assistantContext?.instruction_memory)
      ? assistantContext.instruction_memory
      : [],
    last_affected_scope: last
      ? {
          level: last.target_scope === 'full_document' ? 'full' : (last.target_scope || null),
          target_id: last.sop_id || null,
          target_label: last.section_name || last.sop_title || null,
          line_number: null,
        }
      : focus
        ? {
            level: focus.target_scope === 'full_document' ? 'full' : (focus.target_scope || null),
            target_id: focus.sop_id || null,
            target_label: focus.section_name || focus.sop_title || null,
            line_number: null,
          }
      : {
          level: null,
          target_id: null,
          target_label: null,
          line_number: null,
        },
  }
}

function matchRouteConfig(pathname = '/') {
  const matchedKey = Object.keys(ROUTE_CONFIG).find((route) => pathname.startsWith(route))
  return matchedKey ? ROUTE_CONFIG[matchedKey] : DEFAULT_CONFIG
}

/** Strip legacy injected SOP context prefix from stored/displayed user messages. */
export function toVisibleUserMessage(text) {
  const raw = String(text || '').trim()
  if (!raw) return ''
  const prefixMatch = raw.match(/^Active SOP context:\s*[\s\S]+?\.\s*User request:\s*/i)
  if (prefixMatch) {
    const visible = raw.slice(prefixMatch[0].length).trim()
    return visible || raw
  }
  return raw
}

export function getAssistantRouteMeta(pathname = '/') {
  return matchRouteConfig(pathname)
}

export async function runUnifiedAssistantQuery({
  question,
  pathname = '/',
  chatHistory = [],
  assistantActionConfirmation = null,
  surface = 'global_chatbot',
  sessionId = null,
  assistantMode = null,
  assistantContextOverride = null,
}) {
  const routeMeta = matchRouteConfig(pathname)
  const visibleQuestion = String(question || '').trim()
  const assistantContext = assistantContextOverride || getKLAssistantContext(pathname)
  const editorContextContract = buildEditorContextContract({
    sessionId,
    visibleQuestion,
    assistantContext,
    chatHistory,
  })
  const mode =
    assistantMode === 'query' || assistantMode === 'action'
      ? assistantMode
      : surface === 'kl_assistant'
        ? readKlAssistantMode()
        : 'action'
  return queryAI(visibleQuestion, {
    chat_history: chatHistory,
    category: routeMeta.category,
    assistant_context: {
      ...assistantContext,
      editor_context_contract: editorContextContract,
    },
    assistant_action_confirmation: assistantActionConfirmation,
    surface,
    route: pathname,
    session_id: sessionId || undefined,
    assistant_mode: mode,
  })
}
