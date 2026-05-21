import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { ArrowLeft } from 'lucide-react'
import ConversationList from '../components/Chat/ConversationList'
import ChatPanel from '../components/Chat/ChatPanel'
import {
  createChatSession,
  createDocument,
  getChatSessionMessages,
  hasChatAuthToken,
  listChatSessions,
} from '../api/editorApi'
import {
  formatChatTimeFromIso,
  getAssistantRouteMeta,
  nowTime,
  runUnifiedAssistantQuery,
  stripHtml,
  toHtml,
  toVisibleUserMessage,
} from '../utils/chatAssistant'
import { deriveSopTitleFromText, htmlToPlainText, plainTextToTiptapDoc } from '../utils/chatSopSave'
import { getAssistantContextStorageKeys, resetAssistantStateOnce } from '../utils/assistantContext'
import './ChatPage.css'

/** Only persisted pointer: which server chat_sessions row is active (UUID). Full history loads from GET /api/chat/... */
const LS_ACTIVE_SESSION_ID = 'cybrain_chat_active_session_id'

resetAssistantStateOnce()

const WELCOME_ID = 'm-welcome'

function buildWelcomeMessage() {
  return {
    id: WELCOME_ID,
    sender: 'ai',
    time: nowTime(),
    content: '<p>Chatbot ist verbunden. Stelle eine Frage zu SOPs, Abweichungen, CAPAs, Audits oder Entscheidungen.</p>',
    tags: [],
    showActions: false,
  }
}

function mapDbRowsToMessages(rows) {
  if (!Array.isArray(rows) || rows.length === 0) {
    return [buildWelcomeMessage()]
  }
  return rows.map((m) => ({
    id: m.id,
    sender: m.role === 'user' ? 'user' : 'ai',
    time: formatChatTimeFromIso(m.created_at),
    content: toHtml(
      m.role === 'user' ? toVisibleUserMessage(m.content) : String(m.content || ''),
    ),
    tags: [],
    showActions: m.role === 'assistant',
  }))
}

function sessionRowToConversation(s) {
  return {
    id: s.id,
    serverSessionId: s.id,
    title: s.title || 'Chat',
    description: '',
    time: formatChatTimeFromIso(s.updated_at || s.created_at),
    dateGroup: 'Gespeichert',
    hasAlert: false,
    tags: [],
    messages: [],
    activeSources: [],
    contextTags: [],
    _messagesLoaded: false,
  }
}

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i

function createAnonDraftConversation() {
  const id = `anon-draft-${Date.now()}`
  return {
    id,
    serverSessionId: null,
    title: 'Neues Gespräch',
    description: '',
    time: nowTime(),
    dateGroup: 'Entwurf',
    hasAlert: false,
    tags: [],
    messages: [buildWelcomeMessage()],
    activeSources: [],
    contextTags: [],
    _messagesLoaded: true,
  }
}

/**
 * ChatPage — DB-backed chat; session_id pointer in localStorage; works with or without login.
 */
export default function ChatPage() {
  const location = useLocation()
  const navigate = useNavigate()
  const routeMeta = useMemo(() => getAssistantRouteMeta(location.pathname), [location.pathname])
  const [conversations, setConversations] = useState([])
  const [activeConvId, setActiveConvId] = useState(null)
  const [showChat, setShowChat] = useState(false)
  const [isSending, setIsSending] = useState(false)
  const [pendingDeleteAction, setPendingDeleteAction] = useState(null)
  const [actionToast, setActionToast] = useState('')
  const [historiesLoading, setHistoriesLoading] = useState(true)
  const [historiesError, setHistoriesError] = useState('')

  const loadMessagesIntoConversation = useCallback(async (sessionId, matchConvId = null) => {
    const sid = String(sessionId || '').trim()
    if (!sid) return
    const rows = await getChatSessionMessages(sid)
    const mapped = mapDbRowsToMessages(rows)
    const mid = matchConvId != null ? String(matchConvId) : null
    setConversations((prev) =>
      prev.map((c) => {
        const match = c.serverSessionId === sid || (mid && c.id === mid)
        return match ? { ...c, serverSessionId: sid, id: sid, messages: mapped, _messagesLoaded: true } : c
      }),
    )
  }, [])

  useEffect(() => {
    let cancelled = false
    async function boot() {
      setHistoriesLoading(true)
      setHistoriesError('')

      async function bootAnonymousFromStorage() {
        const preferred = (typeof localStorage !== 'undefined' && localStorage.getItem(LS_ACTIVE_SESSION_ID)) || ''
        const trimmed = preferred.trim()
        if (trimmed && UUID_RE.test(trimmed)) {
          try {
            const rows = await getChatSessionMessages(trimmed)
            if (cancelled) return
            const firstUser = Array.isArray(rows) ? rows.find((r) => r.role === 'user') : null
            const titleHint = firstUser?.content
              ? String(firstUser.content).replace(/\s+/g, ' ').trim().slice(0, 45)
              : 'Chat'
            const conv = {
              id: trimmed,
              serverSessionId: trimmed,
              title: titleHint || 'Chat',
              description: '',
              time: formatChatTimeFromIso(rows?.[0]?.created_at),
              dateGroup: 'Gespeichert',
              hasAlert: false,
              tags: [],
              messages: mapDbRowsToMessages(rows || []),
              activeSources: [],
              contextTags: [],
              _messagesLoaded: true,
            }
            setConversations([conv])
            setActiveConvId(trimmed)
            return
          } catch (e) {
            console.error('[chat-history-load] anon session from storage', e)
            try {
              localStorage.removeItem(LS_ACTIVE_SESSION_ID)
            } catch {
              // ignore
            }
          }
        }
        if (cancelled) return
        const draft = createAnonDraftConversation()
        setConversations([draft])
        setActiveConvId(draft.id)
      }

      if (hasChatAuthToken()) {
        try {
          const sessions = await listChatSessions()
          if (cancelled) return
          const mapped = (sessions || []).map(sessionRowToConversation)
          let preferred = localStorage.getItem(LS_ACTIVE_SESSION_ID)
          if (preferred && !mapped.some((c) => c.id === preferred)) {
            preferred = null
          }
          const activeId = preferred || mapped[0]?.id || null
          if (mapped.length === 0) {
            setConversations([])
            setActiveConvId(null)
          } else {
            setConversations(mapped)
            setActiveConvId(activeId || mapped[0].id)
            if (activeId || mapped[0].id) {
              const sid = activeId || mapped[0].id
              const rows = await getChatSessionMessages(sid)
              if (cancelled) return
              const msgs = mapDbRowsToMessages(rows)
              setConversations((prev) =>
                prev.map((c) => (c.id === sid ? { ...c, messages: msgs, _messagesLoaded: true } : c)),
              )
            }
          }
        } catch (err) {
          if (cancelled) return
          setHistoriesError(err?.message || 'Konnte Chat-Verlauf nicht laden.')
          await bootAnonymousFromStorage()
        }
      } else {
        await bootAnonymousFromStorage()
      }
      if (!cancelled) setHistoriesLoading(false)
    }
    boot()
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (!activeConvId) return
    if (String(activeConvId).startsWith('anon-draft-')) return
    const conv = conversations.find((c) => c.id === activeConvId)
    const sid = (conv?.serverSessionId || activeConvId || '').trim()
    if (sid && UUID_RE.test(sid)) {
      localStorage.setItem(LS_ACTIVE_SESSION_ID, sid)
    }
  }, [activeConvId, conversations])

  const activeConversation = useMemo(
    () => conversations.find((c) => c.id === activeConvId) || null,
    [conversations, activeConvId],
  )

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

  const handleSelect = useCallback(
    async (id) => {
      setActiveConvId(id)
      setShowChat(true)
      const conv = conversations.find((c) => c.id === id)
      if (conv?.serverSessionId && !conv._messagesLoaded) {
        try {
          await loadMessagesIntoConversation(conv.serverSessionId)
        } catch (e) {
          console.error('[chat-history-load] failed lazy messages', e)
        }
      }
    },
    [conversations, loadMessagesIntoConversation],
  )

  const handleBack = useCallback(() => {
    setShowChat(false)
  }, [])

  const handleNewConversation = useCallback(async () => {
    if (hasChatAuthToken()) {
      try {
        const created = await createChatSession({ title: 'Neues Gespräch' })
        const sid = created?.id
        if (!sid) return
        const next = {
          id: sid,
          serverSessionId: sid,
          title: created.title || 'Neues Gespräch',
          description: '',
          time: formatChatTimeFromIso(created.created_at),
          dateGroup: 'Gespeichert',
          hasAlert: false,
          tags: [],
          messages: [buildWelcomeMessage()],
          activeSources: [],
          contextTags: [],
          _messagesLoaded: true,
        }
        setConversations((prev) => [next, ...prev])
        setActiveConvId(sid)
        localStorage.setItem(LS_ACTIVE_SESSION_ID, sid)
        setShowChat(true)
      } catch (err) {
        console.error('[chat-history-session-create] failed', err)
        window.alert(err?.message || 'Neuer Chat konnte nicht angelegt werden.')
      }
      return
    }
    const next = createAnonDraftConversation()
    setConversations((prev) => [next, ...prev])
    setActiveConvId(next.id)
    try {
      localStorage.removeItem(LS_ACTIVE_SESSION_ID)
    } catch {
      // ignore
    }
    setShowChat(true)
  }, [])

  const handleSendMessage = useCallback(
    async (text, opts = {}) => {
      if (!activeConvId || !text?.trim() || isSending) return
      setIsSending(true)

      const userMsg = {
        id: `u-${Date.now()}`,
        sender: 'user',
        time: nowTime(),
        content: toHtml(text.trim()),
        tags: [],
        showActions: false,
      }

      setConversations((prev) =>
        prev.map((c) =>
          c.id === activeConvId
            ? {
                ...c,
                messages: [...(c.messages || []).filter((m) => m.id !== WELCOME_ID), userMsg],
                description: text.trim().slice(0, 80),
                time: nowTime(),
              }
            : c,
        ),
      )

      try {
        const msgsForHist = (activeConversation?.messages || []).filter((m) => m.id !== WELCOME_ID)
        const chatHistoryPayload = [
          ...msgsForHist.map((msg) => ({
            role: msg.sender === 'ai' ? 'assistant' : 'user',
            content: stripHtml(msg.content),
          })),
          { role: 'user', content: text.trim() },
        ].filter((item) => item.content)

        const result = await runUnifiedAssistantQuery({
          question: text.trim(),
          pathname: location.pathname,
          chatHistory: chatHistoryPayload,
          assistantActionConfirmation: opts.assistantActionConfirmation || null,
          surface: 'global_chatbot',
          sessionId: activeConversation?.serverSessionId || null,
        })

        const action = result?.assistant_action
        if (action?.requires_confirmation && action?.type === 'delete_sop') {
          setPendingDeleteAction({
            question: text.trim(),
            conversationId: activeConvId,
          })
        } else {
          setPendingDeleteAction(null)
        }
        if (action?.ok && action?.type === 'create_sop' && action?.sop_id) {
          emitSOPRefresh('create', action.sop_id)
          showToast('SOP created successfully')
          navigate(`/editor/${action.sop_id}`)
        }
        if (action?.ok && action?.type === 'update_sop') {
          showToast('SOP updated successfully')
        }
        if (action?.ok && action?.type === 'delete_sop') {
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
          const newSid = String(result.session_id).trim()
          localStorage.setItem(LS_ACTIVE_SESSION_ID, newSid)
          const rows = await getChatSessionMessages(newSid)
          const mapped = mapDbRowsToMessages(rows)
          setConversations((prev) =>
            prev.map((c) =>
              c.id === activeConvId
                ? {
                    ...c,
                    id: newSid,
                    serverSessionId: newSid,
                    title: c.title === 'Neues Gespräch' ? text.trim().slice(0, 45) : c.title,
                    messages: mapped,
                    _messagesLoaded: true,
                  }
                : c,
            ),
          )
          setActiveConvId(newSid)
        } else {
          const strictInventory = Boolean(result?.retrieval_stats?.strict_mode)
          const sourceTags = strictInventory
            ? []
            : (result.sources || []).slice(0, 5).map((s, idx) => ({
                id: `src-${Date.now()}-${idx}`,
                label: s.label || s.id || `Quelle ${idx + 1}`,
                type: (s.type || 'sop').toLowerCase(),
              }))
          const aiMsg = {
            id: `a-${Date.now()}`,
            sender: 'ai',
            time: nowTime(),
            content: toHtml(result.answer || 'Keine Antwort erhalten.'),
            tags: sourceTags,
            showActions: true,
          }
          setConversations((prev) =>
            prev.map((c) =>
              c.id === activeConvId
                ? {
                    ...c,
                    title: c.title === 'Neues Gespräch' ? text.trim().slice(0, 45) : c.title,
                    messages: [...(c.messages || []), aiMsg],
                    activeSources: sourceTags,
                    contextTags: sourceTags.slice(0, 2),
                  }
                : c,
            ),
          )
        }
      } catch (err) {
        const errMsg = {
          id: `e-${Date.now()}`,
          sender: 'ai',
          time: nowTime(),
          content: toHtml(`Fehler beim Chatbot-Aufruf: ${err.message || 'Unbekannter Fehler'}`),
          tags: [],
          showActions: false,
        }
        setConversations((prev) =>
          prev.map((c) =>
            c.id === activeConvId ? { ...c, messages: [...(c.messages || []), errMsg], hasAlert: true } : c,
          ),
        )
      } finally {
        setIsSending(false)
      }
    },
    [
      activeConvId,
      activeConversation?.messages,
      activeConversation?.serverSessionId,
      isSending,
      location.pathname,
      navigate,
      emitSOPRefresh,
      showToast,
      clearAssistantActiveContext,
      loadMessagesIntoConversation,
    ],
  )

  const confirmDeleteViaAssistant = useCallback(async () => {
    if (!pendingDeleteAction) return
    await handleSendMessage(pendingDeleteAction.question, {
      assistantActionConfirmation: {
        action: 'delete_sop',
        confirmed: true,
      },
    })
  }, [pendingDeleteAction, handleSendMessage])

  const handleMessageAction = useCallback(async (action, message) => {
    try {
      if (!message) return
      const text = stripHtml(message.content || '')
      if (!text) return

      if (action === 'copy') {
        await navigator.clipboard.writeText(text)
        return
      }

      if (action === 'export') {
        const blob = new Blob([text], { type: 'text/plain;charset=utf-8' })
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `chat-response-${Date.now()}.txt`
        a.click()
        URL.revokeObjectURL(url)
        return
      }

      if (action === 'open_sop') {
        const plain = htmlToPlainText(message.content || '')
        const title = deriveSopTitleFromText(plain)
        const docJson = plainTextToTiptapDoc(plain)
        let created
        try {
          created = await createDocument({
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
                  note: 'SOP created from chatbot-generated content.',
                  actor: 'AI Assistant',
                  createdAt: new Date().toISOString(),
                },
              ],
            },
          })
        } catch (createErr) {
          if (createErr?.status === 409) {
            window.alert(
              createErr.message ||
                'This SOP ID already exists. Please create a new version or choose another SOP ID.',
            )
            return
          }
          throw createErr
        }
        if (created?.id) {
          navigate(`/editor/${created.id}`)
        }
      }
    } catch (err) {
      console.error('Chat action failed:', err)
    }
  }, [navigate])

  const mobileClass = showChat ? 'chat-page--show-chat' : 'chat-page--show-list'

  return (
    <div className={`chat-page ${mobileClass}`}>
      {actionToast ? (
        <div className="assistant-action-toast" role="status" aria-live="polite">
          {actionToast}
        </div>
      ) : null}
      {historiesLoading ? (
        <div className="chat-page__detail" style={{ padding: 16 }}>
          <p>Lade Gespräche…</p>
        </div>
      ) : null}
      {historiesError ? (
        <div className="chat-page__detail" style={{ padding: 16, color: 'var(--error, #c00)' }}>
          <p>{historiesError}</p>
        </div>
      ) : null}

      <ConversationList
        conversations={conversations.length ? conversations : []}
        activeId={activeConvId}
        onSelect={handleSelect}
        onNewConversation={handleNewConversation}
      />

      <div className="chat-page__detail">
        {showChat && (
          <button className="chat-page__back-btn" onClick={handleBack}>
            <ArrowLeft size={16} />
            Zurück zur Liste
          </button>
        )}
        <ChatPanel
          conversation={
            activeConversation
              ? {
                  ...activeConversation,
                  subtitleParts: [
                    activeConversation.messages?.length
                      ? `${activeConversation.messages.length} Nachrichten`
                      : 'Noch keine Nachrichten',
                    isSending
                      ? 'Antwort wird generiert…'
                      : activeConversation?.serverSessionId
                        ? 'Server gespeichert'
                        : 'Neuer Chat',
                    routeMeta.contextLabel,
                  ],
                  dateDivider: 'Heute',
                }
              : null
          }
          onSendMessage={handleSendMessage}
          isAwaitingResponse={isSending}
          onMessageAction={handleMessageAction}
        />
      </div>
      {pendingDeleteAction ? (
        <div className="sop-delete-modal-overlay" role="presentation">
          <div className="sop-delete-modal" role="dialog" aria-modal="true" aria-labelledby="chat-delete-title">
            <h3 id="chat-delete-title" className="sop-delete-title">
              SOP wirklich löschen?
            </h3>
            <p className="sop-delete-message">
              Der KL Assistant wird die aktive SOP nach Ihrer Bestätigung sicher entfernen (Soft Delete).
            </p>
            <div className="sop-delete-actions">
              <button
                type="button"
                className="sop-delete-btn sop-delete-btn-cancel"
                onClick={() => setPendingDeleteAction(null)}
                disabled={isSending}
              >
                Cancel
              </button>
              <button
                type="button"
                className="sop-delete-btn sop-delete-btn-confirm"
                onClick={confirmDeleteViaAssistant}
                disabled={isSending}
              >
                {isSending ? 'Deleting...' : 'OK'}
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  )
}
