import React, { useState, useEffect, useRef, useCallback, memo } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { Send, Zap } from 'lucide-react'
import {
  nowTime,
  runUnifiedAssistantQuery,
  getAssistantRouteMeta,
  toVisibleUserMessage,
  toHtml,
  formatChatTimeFromIso,
} from '../../utils/chatAssistant'
import {
  createDocument,
  getChatSessionMessages,
} from '../../api/editorApi'
import { htmlToPlainText, deriveSopTitleFromText, plainTextToTiptapDoc } from '../../utils/chatSopSave'
import {
  getKLAssistantContext,
  getAssistantContextStorageKeys,
  resetAssistantStateOnce,
  saveAssistantLastAction,
} from '../../utils/assistantContext'
import {
  EDITOR_AI_ACTIONS,
  EDITOR_AI_ACTION_STATUS,
  describeEditorAiResult,
  dispatchEditorAiActionRequest,
  getActiveEditorDocumentId,
  hasActiveSopEditor,
  makeEditorAiRequestId,
  SOP_EDITOR_CONTEXT_EVENT,
  subscribeEditorAiActionResult,
} from '../../utils/editorAiBridge'
import {
  buildEnrichedActionPrompt,
  classifyAssistantMessage,
  planEditorActionExecution,
} from '../../utils/assistantIntentRouter'
import EditorChatActions from './EditorChatActions'
import { dispatchActionsTabRun } from '../../utils/editorActionsBridge'
import { runEditorGapCheck } from '../../utils/editorGapCheck'
import { buildGapCheckSidebarReport } from '../../utils/actionsTabGapReport'
import './DashboardComponents.css'

const LS_SESSION_BY_PATH = 'cybrain_kl_chat_session_by_path'

resetAssistantStateOnce()

function readSessionIdForPath(pathname) {
  try {
    const raw = localStorage.getItem(LS_SESSION_BY_PATH)
    const j = raw ? JSON.parse(raw) : {}
    const sid = j?.[pathname]
    return sid && String(sid).trim() ? String(sid).trim() : null
  } catch {
    return null
  }
}

function writeSessionIdForPath(pathname, sessionId) {
  try {
    const raw = localStorage.getItem(LS_SESSION_BY_PATH)
    const j = raw ? JSON.parse(raw) : {}
    j[pathname] = sessionId
    localStorage.setItem(LS_SESSION_BY_PATH, JSON.stringify(j))
  } catch {
    // ignore
  }
}

function defaultGreeting() {
  return [
    {
      id: `greeting-${Date.now()}`,
      role: 'ai',
      text: 'Chatbot ist verbunden. Stelle eine Frage oder bitte um eine Aktion (z. B. Rewrite, Improve, Gap Check, Zusammenfassung) — alles in diesem Chat.',
      tags: [],
      time: nowTime(),
    },
  ]
}

function mapSourcesToWidgetTags(sources) {
  if (!Array.isArray(sources)) return []
  return sources.slice(0, 5).map((s, idx) => s?.label || s?.id || `Quelle ${idx + 1}`)
}

function dbMessagesToWidget(rows) {
  if (!Array.isArray(rows) || rows.length === 0) return defaultGreeting()
  return rows.map((m) => ({
    id: m.id,
    role: m.role === 'user' ? 'user' : 'ai',
    text:
      m.role === 'user'
        ? toVisibleUserMessage(m.content)
        : String(m.content || ''),
    tags: m.role === 'user' ? [] : mapSourcesToWidgetTags(m.sources),
    time: formatChatTimeFromIso(m.created_at),
  }))
}

const CHAT_INPUT_PLACEHOLDER_EDITOR =
  'Frage oder Aktion eingeben (z. B. Rewrite, Übersetzung ins Englische)…'
const CHAT_INPUT_PLACEHOLDER_DEFAULT = 'Frage zur SOP stellen…'

/** Message list isolated from composer keystrokes to avoid sidebar layout flicker. */
const AIWidgetMessageList = memo(function AIWidgetMessageList({
  messages,
  sending,
  assistantMode,
  messagesScrollRef,
  chatEndRef,
  onCreateSop,
}) {
  return (
    <div className="ai-messages-section" ref={messagesScrollRef}>
      {messages.map((m, idx) => (
        <div
          key={m.id}
          className={`ai-chat-message ${m.role}${m.isError ? ' error' : ''}${idx === 0 && String(m.id).startsWith('greeting') ? ' ai-greeting-bubble' : ''}`}
        >
          {idx === 0 && String(m.id).startsWith('greeting') ? (
            <p className="ai-greeting-text">{m.text}</p>
          ) : (
            <div className="ai-message-content" dangerouslySetInnerHTML={{ __html: toHtml(m.text) }} />
          )}
          {m.tags && m.tags.length > 0 && (
            <div className="ai-message-tags">
              {m.tags.map((tag) => (
                <span key={tag} className="ai-message-tag">
                  {tag}
                </span>
              ))}
            </div>
          )}
          {m.role === 'ai' && !m.isError && assistantMode === 'action' && /purpose|zweck|scope|geltungsbereich|procedure|verfahren|responsibilities|verantwortlichkeiten/i.test(m.text) && (
            <button
              className="ai-kontext-btn"
              type="button"
              style={{ marginTop: '10px', padding: '6px 12px', fontSize: '11px', minHeight: 'auto', borderRadius: '4px' }}
              onClick={() => onCreateSop(m.text)}
            >
              Als SOP speichern
            </button>
          )}
        </div>
      ))}

      {sending ? (
        <div
          className="ai-typing-indicator"
          role="status"
          aria-live="polite"
          aria-label="Antwort wird generiert"
        >
          <span />
          <span />
          <span />
        </div>
      ) : null}
      <div ref={chatEndRef} />
    </div>
  )
})

/** Footer (suggestions + input) — only re-renders when compose-related props change. */
const AIWidgetComposeFooter = memo(function AIWidgetComposeFooter({
  input,
  onInputChange,
  onSend,
  sending,
  sopEditorActive,
  suggestions,
  onSuggestionClick,
}) {
  const placeholder = sopEditorActive ? CHAT_INPUT_PLACEHOLDER_EDITOR : CHAT_INPUT_PLACEHOLDER_DEFAULT

  return (
    <div className="ai-widget-chat-footer">
      <div className="ai-widget-divider" />

      <div className="ai-quick-section">
        <h4 className="ai-quick-title">Schnelle Fragen</h4>
        <div className="ai-quick-list">
          {suggestions.map((text) => (
            <button
              key={text}
              type="button"
              className="ai-quick-item"
              onClick={() => onSuggestionClick(text)}
              disabled={sending}
            >
              {text}
            </button>
          ))}
        </div>
      </div>

      <div className="ai-widget-divider" />

      <div className="ai-bottom-input-section">
        <div className="ai-bottom-input-group">
          <input
            type="text"
            placeholder={placeholder}
            className="ai-bottom-input"
            value={input}
            onChange={onInputChange}
            onKeyDown={(e) => e.key === 'Enter' && onSend()}
            disabled={sending}
          />
          <button
            type="button"
            className="ai-bottom-send-btn"
            onClick={onSend}
            disabled={sending || !input.trim()}
            aria-label="Senden"
          >
            <Send size={14} />
          </button>
        </div>
      </div>
    </div>
  )
})

function AIWidget() {
  const location = useLocation()
  const navigate = useNavigate()
  const routeMeta = getAssistantRouteMeta(location.pathname)
  const [messages, setMessages] = useState(() => defaultGreeting())
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const [pendingDeleteAction, setPendingDeleteAction] = useState(null)
  const [actionToast, setActionToast] = useState('')
  const chatEndRef = useRef(null)
  const messagesScrollRef = useRef(null)
  const messagesRef = useRef(messages)
  /** requestId -> { messageId, action } for in-flight editor bridge requests. */
  const pendingBridgeRef = useRef(new Map())
  const suggestions = routeMeta.suggestions
  const [sopEditorActive, setSopEditorActive] = useState(() => hasActiveSopEditor(location.pathname))
  const [liveAssistantContext, setLiveAssistantContext] = useState(() => getKLAssistantContext(location.pathname))
  const assistantMode = sopEditorActive ? 'action' : 'query'

  useEffect(() => {
    const syncEditorContext = () => {
      setSopEditorActive(hasActiveSopEditor(location.pathname))
      setLiveAssistantContext(getKLAssistantContext(location.pathname))
    }
    syncEditorContext()
    window.addEventListener(SOP_EDITOR_CONTEXT_EVENT, syncEditorContext)
    window.addEventListener('storage', syncEditorContext)
    return () => {
      window.removeEventListener(SOP_EDITOR_CONTEXT_EVENT, syncEditorContext)
      window.removeEventListener('storage', syncEditorContext)
    }
  }, [location.pathname])

  const currentSop = liveAssistantContext?.current_sop || {}
  const selectedSection = liveAssistantContext?.selected_section || {}
  const currentSopLabel =
    String(currentSop?.sop_number || currentSop?.title || '').trim() || ''
  const selectedSectionLabel = String(selectedSection?.name || '').trim() || ''
  const dynamicSuggestions = sopEditorActive
    ? [
        selectedSectionLabel
          ? `Fasse den Abschnitt "${selectedSectionLabel}" zusammen`
          : 'Fasse die aktuelle SOP kurz zusammen',
        selectedSectionLabel
          ? `Rewrite den Abschnitt "${selectedSectionLabel}" im aktuellen SOP-Stil`
          : 'Rewrite diese SOP im aktuellen Unternehmensstil',
        selectedSectionLabel
          ? `Prüfe "${selectedSectionLabel}" auf Compliance-Lücken`
          : 'Prüfe diese SOP auf Compliance-Lücken',
        selectedSectionLabel
          ? `Verbessere "${selectedSectionLabel}" ohne die SOP-Struktur zu ändern`
          : 'Verbessere diese SOP ohne die Struktur zu ändern',
      ]
    : suggestions
  const visibleSuggestions = dynamicSuggestions.filter(Boolean).slice(0, 4)

  useEffect(() => {
    if (assistantMode === 'query') setPendingDeleteAction(null)
  }, [assistantMode])

  const emitSOPRefresh = useCallback((reason, sopId) => {
    if (typeof window === 'undefined') return
    window.dispatchEvent(
      new CustomEvent('sops-refresh-request', {
        detail: { reason, sop_id: sopId || null },
      }),
    )
  }, [])

  const showToast = useCallback((text) => {
    setActionToast(text)
    window.setTimeout(() => setActionToast(''), 2400)
  }, [])
  const clearAssistantActiveContext = useCallback(() => {
    const keys = getAssistantContextStorageKeys()
    localStorage.removeItem('current_document_id')
    try {
      const editorRaw = localStorage.getItem(keys.editor)
      if (editorRaw) {
        const parsed = JSON.parse(editorRaw)
        const next = { ...(parsed || {}), sop: {}, linked: {}, editor_text: '' }
        localStorage.setItem(keys.editor, JSON.stringify(next))
      }
    } catch {
      // ignore storage parse failures
    }
    console.info('[assistant-delete-ui] cleared active assistant context')
  }, [])

  useEffect(() => {
    messagesRef.current = messages
  }, [messages])

  const lastMessageScrollKeyRef = useRef('')

  useEffect(() => {
    const scrollKey = `${messages.length}:${sending ? 1 : 0}`
    if (scrollKey === lastMessageScrollKeyRef.current) return
    lastMessageScrollKeyRef.current = scrollKey

    const el = messagesScrollRef.current
    if (!el) return

    requestAnimationFrame(() => {
      el.scrollTop = el.scrollHeight
    })
  }, [messages, sending])

  const loadChatHistory = useCallback(async () => {
    const path = location.pathname
    const sid = readSessionIdForPath(path)
    if (!sid) {
      setMessages(defaultGreeting())
      return
    }
    try {
      const rows = await getChatSessionMessages(sid)
      setMessages(dbMessagesToWidget(rows))
    } catch (e) {
      console.error('[chat-history-load] AIWidget messages', e)
      setMessages(defaultGreeting())
    }
  }, [location.pathname])

  useEffect(() => {
    loadChatHistory()
  }, [loadChatHistory])

  useEffect(() => {
    const pending = pendingBridgeRef.current
    const unsubscribe = subscribeEditorAiActionResult((detail) => {
      const requestId = detail?.requestId
      if (!requestId) return
      const entry = pending.get(requestId)
      if (!entry) return
      console.info('[kl-editor-bridge-received]', { requestId, status: detail?.status, action: detail?.action })
      pending.delete(requestId)
      const statusText = describeEditorAiResult(detail)
      const isError = detail?.status === EDITOR_AI_ACTION_STATUS.ERROR
      setMessages((prev) => prev.map((m) => (
        m.id === entry.messageId
          ? { ...m, text: statusText, isError, _pendingBridge: false, time: nowTime() }
          : m
      )))
      saveAssistantLastAction({
        action: detail?.action || entry.action,
        target_scope: detail?.applied_scope || 'selection',
        section_name: detail?.section_name || '',
        sop_id: detail?.sop_id || getActiveEditorDocumentId(),
        suggested_text_excerpt: statusText,
        status: detail?.status || '',
        source: 'sidebar_bridge_result',
      })
    })
    return () => {
      unsubscribe()
      pending.clear()
    }
  }, [])

  const bridgeStatusText = (intent) => {
    if (intent === EDITOR_AI_ACTIONS.REWRITE) {
      return 'Rewrite wird im Editor vorbereitet. Prüfe die Inline-Vorschau und wähle Accept oder Reject unten.'
    }
    if (intent === EDITOR_AI_ACTIONS.IMPROVE) {
      return 'Verbesserung wird im Editor vorbereitet. Prüfe die Inline-Vorschau und wähle Accept oder Reject unten.'
    }
    if (intent === EDITOR_AI_ACTIONS.GAP_CHECK) {
      return 'Gap Check läuft…'
    }
    if (intent === EDITOR_AI_ACTIONS.READ) return 'Bestätige aktive SOP im Editor…'
    if (intent === EDITOR_AI_ACTIONS.SUMMARIZE) {
      return 'Zusammenfassung wird als Inline-Vorschau im Editor vorbereitet. Prüfe Accept oder Reject unten.'
    }
    if (intent === EDITOR_AI_ACTIONS.ANALYZE) return 'Analyse wird im Editor vorbereitet…'
    if (intent === EDITOR_AI_ACTIONS.COMPARE) return 'Versionsvergleich wird geöffnet…'
    return 'Editor-Aktion wird vorbereitet…'
  }

  /**
   * Route editor-side work from chat using semantic intent classification.
   * Returns true when the message was handled (user message already in the thread).
   */
  const routeClassifiedEditorAction = useCallback(async (text, classification, opts = {}) => {
    const { explicitAction = null } = opts
    if (!hasActiveSopEditor(location.pathname)) return false

    const plan = planEditorActionExecution(classification, { explicitAction })
    const intent = plan.intent
    if (!intent) return false

    const actionPrompt = buildEnrichedActionPrompt(text, classification)
    const { sectionHint, targetScope } = plan.snapshotOptions

    if (intent === EDITOR_AI_ACTIONS.GAP_CHECK) {
      setMessages((prev) => [
        ...prev,
        {
          id: `gap-pending-${Date.now()}`,
          role: 'ai',
          text: bridgeStatusText(intent),
          tags: ['Gap Check'],
          time: nowTime(),
        },
      ])
      try {
        const { result, target } = await runEditorGapCheck({ instruction: actionPrompt })
          const report = buildGapCheckSidebarReport(result)
          const parts = report.sections.map((s) => {
            if (s.gapItems?.length) {
              return `${s.title}\n${s.gapItems.map((g) => `- ${g.issue}${g.recommendation ? `\n  → ${g.recommendation}` : ''}`).join('\n')}`
            }
            return s.body ? `${s.title}\n${s.body}` : s.title
          })
          const plain = parts.filter(Boolean).join('\n\n') || report.analysisPlain
        setMessages((prev) => [
          ...prev.filter((m) => !String(m.id).startsWith('gap-pending-')),
          {
            id: `gap-chat-${Date.now()}`,
            role: 'ai',
            text: `Gap check — ${target.sectionName}${target.isFullDoc ? ' (full SOP)' : ''}\n\n${plain}`,
            tags: ['Gap Check'],
            time: nowTime(),
          },
        ])
        saveAssistantLastAction({
          action: 'gap_check',
          target_scope: target.isFullDoc ? 'full_document' : 'section',
          section_name: target.sectionName || '',
          request_prompt: text,
          original_text_excerpt: target.text || '',
          suggested_text_excerpt: plain || '',
          status: 'suggested',
          source: 'sidebar_gap_check',
          sop_id: getActiveEditorDocumentId(),
        })
        return true
      } catch (err) {
        setMessages((prev) => [
          ...prev.filter((m) => !String(m.id).startsWith('gap-pending-')),
          {
            id: `gap-err-${Date.now()}`,
            role: 'ai',
            text: err?.message || 'Gap check failed.',
            isError: true,
            time: nowTime(),
          },
        ])
        return true
      }
    }

    if (plan.useInline) {
      saveAssistantLastAction({
        action: plan.inlineAction || intent,
        target_scope: sectionHint ? 'section' : (targetScope || 'selection'),
        section_name: sectionHint || '',
        request_prompt: text,
        status: 'requested',
        source: 'sidebar_inline_action',
        sop_id: getActiveEditorDocumentId(),
      })
      setMessages((prev) => [
        ...prev,
        {
          id: `editor-action-${Date.now()}`,
          role: 'ai',
          text: bridgeStatusText(plan.inlineAction || intent),
          tags: [],
          time: nowTime(),
        },
      ])
      dispatchActionsTabRun({
        action: plan.inlineAction || intent,
        prompt: actionPrompt,
        sectionHint,
        targetScope,
      })
      return true
    }

    if (!plan.useBridge) return false

    const activeDocumentId = getActiveEditorDocumentId()
    if (!activeDocumentId) {
      setMessages((prev) => [
        ...prev,
        {
          id: `no-sop-${Date.now()}`,
          role: 'ai',
          text: 'Please open an SOP in the editor first.',
          tags: [],
          time: nowTime(),
        },
      ])
      return true
    }

    const placeholderId = `bridge-${Date.now()}`
    const placeholderMsg = {
      id: placeholderId,
      role: 'ai',
      text: bridgeStatusText(intent),
      tags: [],
      time: nowTime(),
      _pendingBridge: true,
    }
    setMessages((prev) => [...prev, placeholderMsg])

    const requestId = makeEditorAiRequestId()
    pendingBridgeRef.current.set(requestId, { messageId: placeholderId, action: intent })
    saveAssistantLastAction({
      action: intent,
      target_scope: targetScope || 'selection',
      section_name: sectionHint || '',
      request_prompt: text,
      status: 'requested',
      source: 'sidebar_bridge_action',
      sop_id: activeDocumentId,
    })
    dispatchEditorAiActionRequest({
      action: intent,
      prompt: actionPrompt,
      requestId,
      source: 'kl_assistant',
    })

    window.setTimeout(() => {
      const stillPending = pendingBridgeRef.current.get(requestId)
      if (!stillPending) return
      pendingBridgeRef.current.delete(requestId)
      setMessages((prev) => prev.map((m) => (
        m.id === stillPending.messageId
          ? { ...m, text: 'Editor-Aktion hat zu lange gedauert. Bitte erneut versuchen.', isError: true, _pendingBridge: false }
          : m
      )))
    }, 360000)

    return true
  }, [location.pathname, setMessages])

  const sendMessage = useCallback(async (text, opts = {}) => {
    const trimmed = text.trim()
    if (!trimmed || sending) return

    const userMsg = { id: Date.now(), role: 'user', text: trimmed }
    setMessages((prev) => [...prev, userMsg])
    setInput('')
    setSending(true)

    try {
      if (!opts.assistantActionConfirmation) {
        const classification = await classifyAssistantMessage({
          message: trimmed,
          pathname: location.pathname,
          recentMessages: messagesRef.current.map((msg) => ({
            role: msg.role === 'ai' ? 'assistant' : 'user',
            content: msg.text,
          })),
        })

        if (classification.flow === 'clarify' && classification.clarification_question) {
          setMessages((prev) => [
            ...prev,
            {
              id: Date.now() + 1,
              role: 'ai',
              text: classification.clarification_question,
              time: nowTime(),
            },
          ])
          return
        }

        if (classification.flow === 'editor_action' || classification.flow === 'follow_up_action') {
          const handled = await routeClassifiedEditorAction(trimmed, classification, opts)
          if (handled) return
        }
      }
      const chatHistoryPayload = [
        ...messagesRef.current.map((msg) => ({
          role: msg.role === 'ai' ? 'assistant' : 'user',
          content: msg.text,
        })),
        { role: 'user', content: trimmed },
      ]
      const sid = readSessionIdForPath(location.pathname)
      const result = await runUnifiedAssistantQuery({
        question: trimmed,
        pathname: location.pathname,
        chatHistory: chatHistoryPayload,
        assistantActionConfirmation: opts.assistantActionConfirmation || null,
        surface: 'kl_assistant',
        sessionId: sid,
        assistantMode,
      })
      const action = result?.assistant_action
      if (assistantMode === 'action' && action?.requires_confirmation && action?.type === 'delete_sop') {
        setPendingDeleteAction({
          question: trimmed,
          action,
        })
      } else {
        setPendingDeleteAction(null)
      }
      if (assistantMode === 'action' && action?.ok && action?.type === 'create_sop' && action?.sop_id) {
        emitSOPRefresh('create', action.sop_id)
        showToast('SOP created successfully')
        navigate(`/editor/${action.sop_id}`)
      }
      if (assistantMode === 'action' && action?.ok && action?.type === 'update_sop') {
        showToast('SOP updated successfully')
      }
      if (assistantMode === 'action' && action?.ok && action?.type === 'delete_sop') {
        emitSOPRefresh('delete', action.sop_id)
        showToast('SOP deleted successfully')
        console.info('[assistant-delete-ui] delete success', action)
        const activeId = localStorage.getItem('current_document_id')
        if (activeId && action?.sop_id && String(activeId) === String(action.sop_id)) {
          clearAssistantActiveContext()
          navigate('/sops')
        }
      }
      if (result?.session_id) {
        if (assistantMode === 'action') {
          saveAssistantLastAction({
            action: result?.intent || result?.action || 'chat',
            target_scope: 'chat',
            sop_id: getActiveEditorDocumentId(),
            request_prompt: trimmed,
            suggested_text_excerpt: result.answer || result.text || result.response || '',
            status: 'answered',
            source: 'sidebar_query_response',
          })
        }
        writeSessionIdForPath(location.pathname, result.session_id)
        const rows = await getChatSessionMessages(result.session_id)
        setMessages(dbMessagesToWidget(rows))
      } else {
        if (assistantMode === 'action') {
          saveAssistantLastAction({
            action: result?.intent || result?.action || 'chat',
            target_scope: 'chat',
            sop_id: getActiveEditorDocumentId(),
            request_prompt: trimmed,
            suggested_text_excerpt: result.answer || result.text || result.response || '',
            status: 'answered',
            source: 'sidebar_query_response',
          })
        }
        const aiMsg = {
          id: Date.now() + 1,
          role: 'ai',
          text: result.answer || result.text || result.response || '—',
          tags: mapSourcesToWidgetTags(result.sources || result.citations),
          time: nowTime(),
        }
        setMessages((prev) => [...prev, aiMsg])
      }
    } catch (err) {
      // Graceful error message in chat
      const errMsg = {
        id: Date.now() + 1,
        role: 'ai',
        text: `Fehler: ${err.message || 'Unbekannter Fehler'}`,
        isError: true,
        time: nowTime(),
      }
      setMessages(prev => [...prev, errMsg])
    } finally {
      setSending(false)
    }
  }, [sending, location.pathname, navigate, emitSOPRefresh, showToast, clearAssistantActiveContext, assistantMode, routeClassifiedEditorAction])

  const handleSend = () => sendMessage(input)

  const handleInputChange = useCallback((e) => {
    setInput(e.target.value)
  }, [])

  // Clicking a suggestion triggers the actual query immediately
  const handleSuggestionClick = (text) => sendMessage(text)

  const handleCreateSOP = useCallback(async (messageText) => {
    if (assistantMode === 'query') return
    try {
      if (!messageText) return
      const htmlText = toHtml(messageText)
      const plain = htmlToPlainText(htmlText)
      const title = deriveSopTitleFromText(plain)
      const docJson = plainTextToTiptapDoc(plain)
      
      const created = await createDocument({
        title,
        doc_type: 'sop',
        doc_json: docJson,
        metadata_json: {
          sopStatus: 'draft',
          sopMetadata: {
            title,
            author: 'AI Assistant',
            reviewer: '',
            riskLevel: 'Medium',
            department: 'Quality',
            documentId: '',
            references: [],
            reviewDate: '',
            effectiveDate: '',
            regulatoryReferences: [],
          },
          auditTrail: [
            {
              action: 'generated_from_chatbot',
              note: 'SOP created from KL Assistant-generated content.',
              actor: 'AI Assistant',
              createdAt: new Date().toISOString(),
            },
          ],
        },
      })
      if (created?.id) {
        navigate(`/editor/${created.id}`)
      }
    } catch (err) {
      console.error('Failed to create SOP from AIWidget:', err)
    }
  }, [navigate, assistantMode])

  const contextLabel = routeMeta.contextLabel

  const confirmDelete = async () => {
    if (!pendingDeleteAction) return
    await sendMessage(pendingDeleteAction.question, {
      assistantActionConfirmation: {
        action: 'delete_sop',
        confirmed: true,
      },
    })
  }

  return (
    <div className="ai-widget-container">
      {actionToast ? (
        <div className="assistant-action-toast" role="status" aria-live="polite">
          {actionToast}
        </div>
      ) : null}
      {/* Header (n_93fb9) */}
      <div className="ai-widget-header-section">
        {/* Title row with status dot, title, and Aktiv badge (n_00925, n_93f3c, n_36ff5, n_a93c8, n_cf5e4) */}
        <div className="ai-widget-header-row">
          <div className="ai-widget-title-group">
            <span className="ai-status-dot" />
            <h3 className="ai-widget-title">KI Assistent</h3>
          </div>
          <span className="ai-aktiv-badge">Aktiv</span>
        </div>
        <div className="ai-widget-divider" />
      </div>

      {/* Context section (n_e1120) */}
      <div className="ai-context-section">
        {/* Context label row (n_36782, n_8a4b0, n_1632b) */}
        <div className="ai-context-row">
          <Zap size={14} className="ai-context-icon" />
          <span className="ai-context-label">{contextLabel}</span>
        </div>
        {sopEditorActive && currentSopLabel ? (
          <div className="ai-context-row" style={{ marginTop: 8 }}>
            <span className="ai-context-label" style={{ fontSize: 12 }}>
              Aktive SOP: {currentSopLabel}
              {selectedSectionLabel ? ` • Abschnitt: ${selectedSectionLabel}` : ''}
            </span>
          </div>
        ) : null}
      </div>

      <div className="ai-widget-divider" />

      <AIWidgetMessageList
        messages={messages}
        sending={sending}
        assistantMode={assistantMode}
        messagesScrollRef={messagesScrollRef}
        chatEndRef={chatEndRef}
        onCreateSop={handleCreateSOP}
      />

      <div className="ai-widget-bottom-stack">
        <div className="ai-editor-actions-slot">
          {sopEditorActive ? <EditorChatActions /> : null}
        </div>

        <AIWidgetComposeFooter
          input={input}
          onInputChange={handleInputChange}
          onSend={handleSend}
          sending={sending}
          sopEditorActive={sopEditorActive}
          suggestions={visibleSuggestions}
          onSuggestionClick={handleSuggestionClick}
        />
      </div>

      {pendingDeleteAction ? (
        <div className="sop-delete-modal-overlay" role="presentation">
          <div className="sop-delete-modal" role="dialog" aria-modal="true" aria-labelledby="assistant-delete-title">
            <h3 id="assistant-delete-title" className="sop-delete-title">SOP wirklich löschen?</h3>
            <p className="sop-delete-message">
              Diese Aktion blendet die aktuell aktive SOP aus dem Workspace aus. Sie können den Löschvorgang jetzt bestätigen oder abbrechen.
            </p>
            <div className="sop-delete-actions">
              <button
                type="button"
                className="sop-delete-btn sop-delete-btn-cancel"
                onClick={() => setPendingDeleteAction(null)}
                disabled={sending}
              >
                Cancel
              </button>
              <button
                type="button"
                className="sop-delete-btn sop-delete-btn-confirm"
                onClick={confirmDelete}
                disabled={sending}
              >
                {sending ? 'Deleting...' : 'OK'}
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  )
}

export default memo(AIWidget)
