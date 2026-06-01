import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import AIComparisonModal from './AIComparisonModal'
import { performAIAction } from '../../api/editorApi'
import { formatAiSuggestionForUi } from '../../utils/aiOutputFormatter'
import {
  dispatchEditorSnapshotResponse,
  subscribeEditorInlineSuggestionApply,
  subscribeEditorInlineSuggestionClear,
  subscribeEditorInlineSuggestionShow,
  subscribeEditorSnapshotRequest,
  EDITOR_GAP_APPEND_EVENT,
  EDITOR_SCROLL_TO_RANGE_EVENT,
  EDITOR_SELECTION_QUERY_EVENT,
  EDITOR_SELECTION_RESPONSE_EVENT,
} from '../../utils/editorActionsBridge'
import {
  clearInlineAiSuggestion,
  setInlineAiSuggestion,
} from '../../utils/editorInlineSuggestionPlugin'
import { resolveTargetInEditor } from '../../utils/editorTargetResolver'
import {
  AI_ACTION_TRIGGERED_BY,
  EDITOR_AI_ACTIONS,
  EDITOR_AI_ACTION_STATUS,
  dispatchEditorAiActionResult,
  subscribeEditorAiActionRequest,
} from '../../utils/editorAiBridge'

const INLINE_SHOWN_EVENT = 'editor-actions-inline-shown'
const INLINE_APPLIED_EVENT = 'editor-actions-inline-applied'

const ACTION_TEXT_WARNING_CHARS = 7000

const stripHtml = (value) =>
  String(value || '')
    .replace(/<br\s*\/?>/gi, '\n')
    .replace(/<\/p>/gi, '\n')
    .replace(/<\/div>/gi, '\n')
    .replace(/<[^>]+>/g, '')
    .replace(/\n{3,}/g, '\n\n')
    .trim()

const buildFindingsHtml = (structuredData) => {
  if (!structuredData || typeof structuredData !== 'object') return ''
  const gaps = Array.isArray(structuredData.gaps) ? structuredData.gaps : []
  const items = gaps.length > 0
    ? gaps
    : [{
      issue: structuredData.issue,
      explanation: structuredData.explanation,
      recommendation: structuredData.recommendation,
    }].filter((entry) => entry && (entry.issue || entry.explanation || entry.recommendation))

  if (items.length === 0) return ''

  const itemsHtml = items
    .map((gap) => {
      const issue = gap?.issue ? `<p><strong>Issue:</strong> ${gap.issue}</p>` : ''
      const explanation = gap?.explanation ? `<p><strong>Explanation:</strong> ${gap.explanation}</p>` : ''
      const recommendation = gap?.recommendation
        ? `<p><strong>Recommendation:</strong> ${gap.recommendation}</p>`
        : ''
      return `<li>${issue}${explanation}${recommendation}</li>`
    })
    .join('')

  return `<h3>AI Gap Check Findings</h3><ul>${itemsHtml}</ul>`
}

const ALLOWED_ACTIONS = new Set([
  EDITOR_AI_ACTIONS.REWRITE,
  EDITOR_AI_ACTIONS.IMPROVE,
  EDITOR_AI_ACTIONS.GAP_CHECK,
  EDITOR_AI_ACTIONS.SUMMARIZE,
  EDITOR_AI_ACTIONS.ANALYZE,
])

/**
 * Bridges KL/KI Assistant action requests into the live SOP editor.
 *
 * Subscribes to {@link EDITOR_AI_ACTION_REQUEST_EVENT} dispatched by chat
 * surfaces (AIWidget, ChatPage). For rewrite / improve / gap_check it runs
 * `/api/ai/action` on the current selection (or whole document when no
 * selection exists), shows the standard {@link AIComparisonModal}, then
 * applies the result into the editor on accept. A
 * {@link EDITOR_AI_ACTION_RESULT_EVENT} is emitted so the chat surface can
 * report status back to the user.
 *
 * The component renders nothing besides the modal portal.
 */
const EditorAIBridge = ({
  editor,
  documentId,
  sopMetadata,
  isEditable = true,
  onPreviewSessionChange,
  onAfterApply,
  onVersionCompareRequest,
}) => {
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [aiResult, setAIResult] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  /** Snapshot of the request that opened the current modal. */
  const activeRequestRef = useRef(null)
  /** Range in the editor that should receive the accepted content. */
  const targetRangeRef = useRef(null)
  /** Tracks whether we are currently using the full document as the source. */
  const isFullDocRef = useRef(false)
  const inFlightRef = useRef(false)
  const editorRef = useRef(editor)
  const sopMetadataRef = useRef(sopMetadata)
  const documentIdRef = useRef(documentId)
  const isEditableRef = useRef(isEditable)
  /** Pending inline suggestion from the sidebar Actions tab. */
  const inlinePendingRef = useRef(null)

  useEffect(() => { editorRef.current = editor }, [editor])
  useEffect(() => { sopMetadataRef.current = sopMetadata }, [sopMetadata])
  useEffect(() => { documentIdRef.current = documentId }, [documentId])
  useEffect(() => { isEditableRef.current = isEditable }, [isEditable])

  const notifyPreviewSession = useCallback((active) => {
    if (typeof onPreviewSessionChange === 'function') {
      onPreviewSessionChange(active)
    }
  }, [onPreviewSessionChange])

  const emitResult = useCallback((detail) => {
    dispatchEditorAiActionResult(detail)
  }, [])

  const closeModal = useCallback(() => {
    setIsModalOpen(false)
    setAIResult(null)
    activeRequestRef.current = null
    targetRangeRef.current = null
    isFullDocRef.current = false
    notifyPreviewSession(false)
  }, [notifyPreviewSession])

  const sopTitle = useMemo(() => {
    const metadata = sopMetadata || {}
    return (metadata.title || metadata.documentId || 'Untitled SOP').toString().trim() || 'Untitled SOP'
  }, [sopMetadata])

  const runActionRequest = useCallback(async (request) => {
    const { action, requestId } = request || {}
    if (action === EDITOR_AI_ACTIONS.COMPARE) {
      const liveEditor = editorRef.current
      if (!liveEditor || liveEditor.isDestroyed) {
        emitResult({
          ...request,
          status: EDITOR_AI_ACTION_STATUS.NOT_AVAILABLE,
          reason: 'editor_unavailable',
          message: 'Editor nicht bereit.',
        })
        console.warn('[kl-editor-action-failed]', { action, requestId, reason: 'editor_unavailable' })
        return
      }
      if (typeof onVersionCompareRequest !== 'function') {
        emitResult({
          ...request,
          status: EDITOR_AI_ACTION_STATUS.ERROR,
          message: 'Versionsvergleich ist hier nicht verfügbar.',
        })
        console.warn('[kl-editor-action-failed]', { action, requestId, reason: 'no_compare_handler' })
        return
      }
      try {
        console.info('[kl-editor-bridge-received]', { action, requestId, phase: 'compare' })
        await Promise.resolve(onVersionCompareRequest())
        emitResult({
          ...request,
          status: EDITOR_AI_ACTION_STATUS.DISPLAYED,
          action: EDITOR_AI_ACTIONS.COMPARE,
        })
      } catch (err) {
        console.error('[kl-editor-action-failed]', err)
        emitResult({
          ...request,
          status: EDITOR_AI_ACTION_STATUS.ERROR,
          message: err?.message || 'Versionsvergleich fehlgeschlagen.',
        })
      }
      return
    }

    if (!ALLOWED_ACTIONS.has(action)) {
      emitResult({
        ...request,
        status: EDITOR_AI_ACTION_STATUS.NOT_AVAILABLE,
        reason: 'unsupported_action',
      })
      return
    }
    const liveEditor = editorRef.current
    if (!liveEditor || liveEditor.isDestroyed || !isEditableRef.current) {
      emitResult({
        ...request,
        status: EDITOR_AI_ACTION_STATUS.NOT_AVAILABLE,
        reason: 'editor_unavailable',
      })
      return
    }

    if (inFlightRef.current) {
      emitResult({
        ...request,
        status: EDITOR_AI_ACTION_STATUS.ERROR,
        message: 'Es läuft bereits eine Editor-Aktion.',
      })
      return
    }

    const { state } = liveEditor
    const { selection } = state
    const hasSelection = Boolean(selection && !selection.empty)
    const selectionPayload = hasSelection
      ? { from: selection.from, to: selection.to, empty: false }
      : { empty: true }

    let from = 0
    let to = state.doc.content.size
    let selectedText = state.doc.textBetween(from, to, '\n').trim()
    let isFullDoc = true
    let sectionName = 'Full SOP'
    let sectionType = 'Full Document'

    const actionPrompt = String(request?.prompt || '').trim()
    if (actionPrompt) {
      try {
        const resolved = resolveTargetInEditor(liveEditor, {
          prompt: actionPrompt,
          userPrompt: String(request?.userPrompt || ''),
          selection: selectionPayload,
          sectionHint: String(request?.sectionHint || ''),
          targetScope: String(request?.targetScope || ''),
          lineNumber: request?.lineNumber ?? null,
          recordId: String(request?.recordId || ''),
          preferFullSection: Boolean(request?.preferFullSection),
        })
        if (resolved?.text && resolved.from != null && resolved.to != null) {
          from = resolved.from
          to = resolved.to
          selectedText = resolved.text
          isFullDoc = Boolean(resolved.isFullDoc)
          sectionName = resolved.sectionName || sectionName
          sectionType = resolved.sectionType || sectionType
        }
      } catch (err) {
        emitResult({
          ...request,
          status: EDITOR_AI_ACTION_STATUS.ERROR,
          message: err?.message || 'Could not resolve target in the open SOP.',
        })
        return
      }
    } else if (hasSelection) {
      from = selection.from
      to = selection.to
      const fragment = state.doc.textBetween(from, to, '\n').trim()
      if (fragment.length > 0) {
        selectedText = fragment
        isFullDoc = false
        sectionName = 'Selected text'
        sectionType = 'Paragraph'
      }
    }

    if (!selectedText) {
      emitResult({
        ...request,
        status: EDITOR_AI_ACTION_STATUS.NOT_AVAILABLE,
        reason: 'empty_document',
      })
      return
    }

    if (selectedText.length > ACTION_TEXT_WARNING_CHARS) {
      const proceed = window.confirm(
        'Dieser SOP-Inhalt ist möglicherweise zu lang für das lokale LLM und kann mit Kontextlimit-Fehlern abbrechen.\n\nMit der Aktion fortfahren?',
      )
      if (!proceed) {
        emitResult({
          ...request,
          status: EDITOR_AI_ACTION_STATUS.CANCELLED,
          reason: 'user_declined_long_text',
        })
        return
      }
    }

    inFlightRef.current = true
    activeRequestRef.current = request
    targetRangeRef.current = { from, to }
    isFullDocRef.current = isFullDoc
    notifyPreviewSession(true)
    setIsLoading(true)

    try {
      console.info('[kl-editor-bridge-received]', {
        action,
        requestId,
        documentId: documentIdRef.current,
        textLen: selectedText.length,
        isFullDoc,
        source: request?.source || 'unknown',
      })
      const result = await performAIAction({
        action,
        text: selectedText,
        document_id: documentIdRef.current || sopMetadataRef.current?.documentId || null,
        section_id: `kl-assistant-${requestId || Date.now()}`,
        sop_title: sopTitle,
        section_name: sectionName,
        section_type: sectionType,
        edit_scope: isFullDoc ? 'full_document' : 'section_only',
        sop_entity_id: documentIdRef.current || null,
        triggered_by: AI_ACTION_TRIGGERED_BY.KL_ASSISTANT,
        instruction: actionPrompt || null,
        learn_to_profile: Boolean(request?.learn_to_profile),
      })

      const safeSuggestedHtml = formatAiSuggestionForUi({
        action: result?.action || action,
        suggestedText: result?.suggested_text,
        structuredData: result?.structured_data,
      })

      setAIResult({
        ...result,
        action: result?.action || action,
        suggested_text: safeSuggestedHtml,
        section_name: sectionName,
      })
      setIsModalOpen(true)
      console.info('[kl-editor-action-modal-open]', { action: result?.action || action, requestId, isFullDoc })
    } catch (err) {
      console.error('[kl-editor-action-failed]', err)
      const message = err?.message || 'Editor-Aktion fehlgeschlagen.'
      emitResult({
        ...request,
        status: EDITOR_AI_ACTION_STATUS.ERROR,
        message,
      })
      notifyPreviewSession(false)
      activeRequestRef.current = null
      targetRangeRef.current = null
      isFullDocRef.current = false
      window.alert(message)
    } finally {
      inFlightRef.current = false
      setIsLoading(false)
    }
  }, [emitResult, notifyPreviewSession, sopTitle, onVersionCompareRequest])

  const handleReadRequest = useCallback((request) => {
    const liveEditor = editorRef.current
    if (!liveEditor || liveEditor.isDestroyed) {
      emitResult({
        ...request,
        status: EDITOR_AI_ACTION_STATUS.NOT_AVAILABLE,
        reason: 'editor_unavailable',
      })
      return
    }
    const metadata = sopMetadataRef.current || {}
    const preview = (liveEditor.getText() || '').slice(0, 400)
    emitResult({
      ...request,
      status: EDITOR_AI_ACTION_STATUS.DISPLAYED,
      sop_id: documentIdRef.current || null,
      sop_title: metadata.title || '',
      sop_number: metadata.documentId || '',
      preview,
    })
  }, [emitResult])

  const resolveDocRange = useCallback((detail) => {
    const liveEditor = editorRef.current
    if (!liveEditor || liveEditor.isDestroyed) return null
    const size = liveEditor.state.doc.content.size
    const from = Number.isFinite(detail.from) ? detail.from : 0
    const to = Number.isFinite(detail.to) ? detail.to : size
    return {
      from: Math.max(0, Math.min(from, size)),
      to: Math.max(from, Math.min(to, size)),
    }
  }, [])

  const emitInlineShown = useCallback((requestId, range) => {
    const liveEditor = editorRef.current
    let toolbarCoords = null
    if (liveEditor?.view && range) {
      try {
        const coords = liveEditor.view.coordsAtPos(range.to)
        toolbarCoords = { top: coords.top + window.scrollY, left: coords.left + window.scrollX }
      } catch {
        toolbarCoords = null
      }
    }
    window.dispatchEvent(
      new CustomEvent(INLINE_SHOWN_EVENT, {
        detail: {
          requestId,
          toolbarCoords,
          from: range?.from,
          to: range?.to,
        },
      }),
    )
  }, [])

  const emitInlineApplied = useCallback((requestId, ok, message = '') => {
    window.dispatchEvent(
      new CustomEvent(INLINE_APPLIED_EVENT, {
        detail: { requestId, ok, message },
      }),
    )
  }, [])

  useEffect(() => {
    const onSelectionQuery = (event) => {
      const requestId = event.detail?.requestId
      const liveEditor = editorRef.current
      if (!requestId) return
      let hasSelection = false
      if (liveEditor && !liveEditor.isDestroyed && isEditableRef.current) {
        const sel = liveEditor.state.selection
        hasSelection = Boolean(sel && !sel.empty)
      }
      window.dispatchEvent(
        new CustomEvent(EDITOR_SELECTION_RESPONSE_EVENT, {
          detail: { requestId, hasSelection },
        }),
      )
    }
    window.addEventListener(EDITOR_SELECTION_QUERY_EVENT, onSelectionQuery)
    return () => window.removeEventListener(EDITOR_SELECTION_QUERY_EVENT, onSelectionQuery)
  }, [])

  useEffect(() => {
    const unsubSnapshot = subscribeEditorSnapshotRequest(({
      requestId,
      prompt,
      userPrompt,
      sectionHint,
      targetScope,
      lineNumber,
      recordId,
      preferFullSection,
    }) => {
      const liveEditor = editorRef.current
      if (!liveEditor || liveEditor.isDestroyed || !isEditableRef.current) {
        dispatchEditorSnapshotResponse({
          requestId,
          ok: false,
          message: 'Editor is not available or is read-only.',
        })
        return
      }
      const { state } = liveEditor
      const { selection } = state
      const hasSelection = Boolean(selection && !selection.empty)
      const selectionPayload = hasSelection
        ? {
            from: selection.from,
            to: selection.to,
            text: state.doc.textBetween(selection.from, selection.to, '\n'),
            empty: false,
          }
        : { empty: true }

      try {
        const target = resolveTargetInEditor(liveEditor, {
          prompt: String(prompt || ''),
          userPrompt: String(userPrompt || ''),
          selection: selectionPayload,
          sectionHint: String(sectionHint || ''),
          targetScope: String(targetScope || ''),
          lineNumber: lineNumber ?? null,
          recordId: String(recordId || ''),
          preferFullSection: Boolean(preferFullSection),
        })
        if (!target?.text || target.from == null || target.to == null) {
          dispatchEditorSnapshotResponse({
            requestId,
            ok: false,
            error: 'Could not find that heading or paragraph in the open SOP. Check the text or select it in the editor.',
          })
          return
        }
        dispatchEditorSnapshotResponse({
          requestId,
          ok: true,
          target,
          fullText: state.doc.textBetween(0, state.doc.content.size, '\n'),
          docSize: state.doc.content.size,
          selection: selectionPayload,
          sopTitle: (sopMetadataRef.current?.title || 'Untitled SOP').toString(),
          sopNumber: (sopMetadataRef.current?.documentId || '').toString(),
        })
      } catch (err) {
        dispatchEditorSnapshotResponse({
          requestId,
          ok: false,
          error: err?.message || 'Could not resolve target in editor.',
        })
      }
    })

    const unsubShow = subscribeEditorInlineSuggestionShow((detail) => {
      const liveEditor = editorRef.current
      const requestId = detail?.requestId
      if (!requestId || !liveEditor || liveEditor.isDestroyed || !isEditableRef.current) {
        emitInlineShown(requestId, null)
        return
      }

      if (inlinePendingRef.current?.requestId && inlinePendingRef.current.requestId !== requestId) {
        clearInlineAiSuggestion(liveEditor)
      }

      let range = resolveDocRange(detail)
      const docSize = liveEditor.state.doc.content.size
      if (detail.isFullDoc && docSize > 0) {
        range = { from: 0, to: Math.max(docSize, 1) }
      }
      if (!range || range.to <= range.from) {
        emitInlineShown(requestId, null)
        return
      }

      const suggestedPlain = String(detail.suggestedPlain || '').trim()
      if (!suggestedPlain) {
        emitInlineShown(requestId, null)
        return
      }

      inlinePendingRef.current = {
        requestId,
        ...range,
        suggestedPlain,
        suggestedHtml: detail.suggestedHtml || null,
        acceptedContent: detail.acceptedContent || null,
        selectedFraction: Number(detail.selectedFraction) || 0,
        structuredData: detail.structuredData || null,
        action: detail.action,
        isFullDoc: Boolean(detail.isFullDoc),
        originalText: detail.originalText || liveEditor.state.doc.textBetween(range.from, range.to, '\n'),
      }

      notifyPreviewSession(true)
      setInlineAiSuggestion(liveEditor, {
        from: range.from,
        to: range.to,
        suggestedPlain,
        suggestedHtml: detail.suggestedHtml || null,
      })
      try {
        liveEditor.commands.focus()
        liveEditor.commands.setTextSelection({ from: range.from, to: range.to })
        liveEditor.commands.scrollIntoView()
      } catch {
        // non-fatal
      }
      emitInlineShown(requestId, range)
    })

    const unsubClear = subscribeEditorInlineSuggestionClear(({ requestId }) => {
      const liveEditor = editorRef.current
      const pending = inlinePendingRef.current
      if (requestId && pending?.requestId && pending.requestId !== requestId) return
      if (liveEditor && !liveEditor.isDestroyed) {
        clearInlineAiSuggestion(liveEditor)
      }
      inlinePendingRef.current = null
      notifyPreviewSession(false)
    })

    const unsubApply = subscribeEditorInlineSuggestionApply(({ requestId }) => {
      const liveEditor = editorRef.current
      const pending = inlinePendingRef.current
      if (!pending || pending.requestId !== requestId) {
        emitInlineApplied(requestId, false, 'No pending suggestion to apply.')
        return
      }
      if (!liveEditor || liveEditor.isDestroyed) {
        emitInlineApplied(requestId, false, 'Editor is not available.')
        return
      }

      try {
        const {
          from,
          to,
          suggestedPlain,
          suggestedHtml,
          acceptedContent,
          isFullDoc,
          action,
          structuredData,
        } = pending
        const insertPayload =
          acceptedContent
          || (isFullDoc ? suggestedHtml : suggestedPlain)
          || suggestedHtml
          || suggestedPlain

        if (isFullDoc) {
          liveEditor.commands.setContent(insertPayload || '<p></p>', false)
        } else if (typeof insertPayload === 'string' && /<\/?[a-z]/i.test(insertPayload)) {
          liveEditor.chain().focus().insertContentAt({ from, to }, insertPayload).run()
        } else {
          liveEditor.chain().focus().insertContentAt({ from, to }, insertPayload || '').run()
        }
        clearInlineAiSuggestion(liveEditor)
        inlinePendingRef.current = null
        notifyPreviewSession(false)
        emitInlineApplied(requestId, true)
        if (typeof onAfterApply === 'function') {
          onAfterApply({
            action,
            applied_scope: isFullDoc ? 'full_document' : 'selection',
            source: 'actions_tab',
            suggestion_id: structuredData?.suggestion_id || null,
          })
        }
      } catch (err) {
        console.error('[editor-actions-bridge] apply failed', err)
        emitInlineApplied(requestId, false, err?.message || 'Could not apply suggestion.')
      }
    })

    const onScrollToRange = (event) => {
      const liveEditor = editorRef.current
      const { from, to } = event.detail || {}
      if (!liveEditor || liveEditor.isDestroyed || from == null || to == null) return
      try {
        liveEditor.chain().focus().setTextSelection({ from, to }).scrollIntoView().run()
      } catch (err) {
        console.warn('[editor-actions-bridge] scrollIntoView failed', err)
      }
    }
    window.addEventListener(EDITOR_SCROLL_TO_RANGE_EVENT, onScrollToRange)

    const onGapAppend = (event) => {
      const liveEditor = editorRef.current
      const html = event.detail?.html
      if (!liveEditor || liveEditor.isDestroyed || !html) return
      try {
        const docEnd = liveEditor.state.doc.content.size
        const appendix = /<h3/i.test(String(html))
          ? html
          : `<h3>AI Gap Check Findings</h3>${html}`
        liveEditor.chain().focus().insertContentAt(docEnd, appendix, { updateSelection: false }).run()
      } catch (err) {
        console.warn('[editor-actions-bridge] gap append failed', err)
      }
    }
    window.addEventListener(EDITOR_GAP_APPEND_EVENT, onGapAppend)

    return () => {
      unsubSnapshot()
      unsubShow()
      unsubClear()
      unsubApply()
      window.removeEventListener(EDITOR_SCROLL_TO_RANGE_EVENT, onScrollToRange)
      window.removeEventListener(EDITOR_GAP_APPEND_EVENT, onGapAppend)
    }
  }, [editor, emitInlineApplied, emitInlineShown, notifyPreviewSession, onAfterApply, resolveDocRange])

  useEffect(() => {
    const unsubscribe = subscribeEditorAiActionRequest((request) => {
      if (!request || !request.action) return
      console.info('[kl-editor-bridge-received]', {
        action: request.action,
        requestId: request.requestId,
        source: request.source,
      })
      if (request.action === EDITOR_AI_ACTIONS.READ) {
        handleReadRequest(request)
        return
      }
      runActionRequest(request)
    })
    return unsubscribe
  }, [handleReadRequest, runActionRequest])

  const handleAccept = useCallback(() => {
    const liveEditor = editorRef.current
    const request = activeRequestRef.current
    const target = targetRangeRef.current
    if (!liveEditor || liveEditor.isDestroyed || !aiResult || !request) {
      closeModal()
      return
    }

    const action = String(aiResult.action || request.action || '').toLowerCase()
    const suggestedHtml = aiResult.suggested_text || ''
    const structuredData = aiResult.structured_data || {}

    try {
      if (action === EDITOR_AI_ACTIONS.GAP_CHECK) {
        const appendix = buildFindingsHtml(structuredData) || `<h3>AI Gap Check Findings</h3>${suggestedHtml}`
        const docEnd = liveEditor.state.doc.content.size
        liveEditor
          .chain()
          .focus()
          .insertContentAt(docEnd, appendix, { updateSelection: false })
          .run()
        console.info('[kl-editor-action-inserted]', { action, scope: 'append', requestId: request?.requestId })
      } else if (isFullDocRef.current) {
        let payloadHtml = suggestedHtml
        if (action === EDITOR_AI_ACTIONS.REWRITE) {
          payloadHtml = formatAiSuggestionForUi({
            action,
            suggestedText: structuredData?.rewritten_text || aiResult?.suggested_text,
            structuredData,
          })
        } else if (
          action === EDITOR_AI_ACTIONS.IMPROVE
          || action === EDITOR_AI_ACTIONS.SUMMARIZE
          || action === EDITOR_AI_ACTIONS.ANALYZE
        ) {
          const improvedSource =
            structuredData?.improved_text || structuredData?.improved_version || aiResult?.suggested_text
          payloadHtml = formatAiSuggestionForUi({
            action: EDITOR_AI_ACTIONS.IMPROVE,
            suggestedText: improvedSource,
            structuredData,
          })
        }
        liveEditor.commands.setContent(payloadHtml || suggestedHtml || '<p></p>', false)
        console.info('[kl-editor-action-inserted]', { action, scope: 'full_document', requestId: request?.requestId })
      } else {
        const from = target?.from ?? 0
        const to = target?.to ?? liveEditor.state.doc.content.size
        let plainContent = ''
        if (action === EDITOR_AI_ACTIONS.REWRITE) {
          plainContent = stripHtml(structuredData?.rewritten_text || aiResult?.suggested_text)
        } else if (
          action === EDITOR_AI_ACTIONS.IMPROVE
          || action === EDITOR_AI_ACTIONS.SUMMARIZE
          || action === EDITOR_AI_ACTIONS.ANALYZE
        ) {
          plainContent = stripHtml(
            structuredData?.improved_text || structuredData?.improved_version || aiResult?.suggested_text,
          )
        } else {
          plainContent = stripHtml(aiResult?.suggested_text)
        }
        liveEditor
          .chain()
          .focus()
          .insertContentAt({ from, to }, plainContent || '')
          .run()
        console.info('[kl-editor-action-inserted]', { action, scope: 'selection', requestId: request?.requestId })
      }

      emitResult({
        ...request,
        status: EDITOR_AI_ACTION_STATUS.APPLIED,
        action,
        applied_scope: isFullDocRef.current ? 'full_document' : 'selection',
        sop_id: documentIdRef.current || null,
      })
      console.info('[kl-editor-action-accepted]', { action, requestId: request?.requestId })

      if (typeof onAfterApply === 'function') {
        try {
          onAfterApply({
            action,
            applied_scope: isFullDocRef.current ? 'full_document' : 'selection',
            suggestion_id: structuredData?.suggestion_id || null,
          })
        } catch (err) {
          console.error('[editor-ai-bridge] onAfterApply failed', err)
        }
      }
    } catch (err) {
      console.error('[editor-ai-bridge] failed to apply suggestion', err)
      emitResult({
        ...request,
        status: EDITOR_AI_ACTION_STATUS.ERROR,
        message: err?.message || 'Konnte Vorschlag nicht im Editor anwenden.',
      })
    } finally {
      closeModal()
    }
  }, [aiResult, closeModal, emitResult, onAfterApply])

  const handleReject = useCallback(() => {
    const request = activeRequestRef.current
    const action = String(aiResult?.action || request?.action || '').toLowerCase()
    if (request) {
      emitResult({
        ...request,
        status: EDITOR_AI_ACTION_STATUS.CANCELLED,
        action,
      })
    }
    closeModal()
  }, [aiResult, closeModal, emitResult])

  return (
    <>
      <AIComparisonModal
        isOpen={isModalOpen}
        onClose={handleReject}
        action={aiResult?.action}
        originalText={aiResult?.original_text}
        suggestedText={aiResult?.suggested_text}
        explanation={aiResult?.explanation}
        structuredData={aiResult?.structured_data}
        onAccept={handleAccept}
        sectionName={aiResult?.section_name}
        sopTitle={sopTitle}
      />
      {isLoading ? (
        <div className="editor-ai-bridge-loading" role="status" aria-live="polite">
          <div className="editor-ai-bridge-loading__inner">
            <span className="editor-ai-bridge-loading__spinner" />
            <span>KI-Assistent bearbeitet die SOP…</span>
          </div>
        </div>
      ) : null}
    </>
  )
}

export default EditorAIBridge
