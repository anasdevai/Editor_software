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
  const bulletLine = (line) => /^[-*•]\s+/.test(line) || /^\d+[\.\)]\s+/.test(line)

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
        if (bulletLine(row)) bullets.push(row.replace(/^([-*•]|\d+[\.\)])\s+/, '').trim())
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
        bullets.push(lines[j].replace(/^([-*•]|\d+[\.\)])\s+/, '').trim())
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
}) {
  const routeMeta = matchRouteConfig(pathname)
  const visibleQuestion = String(question || '').trim()
  const assistantContext = getKLAssistantContext(pathname)
  const mode =
    assistantMode === 'query' || assistantMode === 'action'
      ? assistantMode
      : surface === 'kl_assistant'
        ? readKlAssistantMode()
        : 'action'
  return queryAI(visibleQuestion, {
    chat_history: chatHistory,
    category: routeMeta.category,
    assistant_context: assistantContext,
    assistant_action_confirmation: assistantActionConfirmation,
    surface,
    route: pathname,
    session_id: sessionId || undefined,
    assistant_mode: mode,
  })
}
