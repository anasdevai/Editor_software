import React, { useCallback, useEffect, useRef, useState, memo } from 'react'
import { useLocation } from 'react-router-dom'
import { Check, X } from 'lucide-react'
import { performAIAction } from '../../api/editorApi'
import { selectionLooksLikeFormattedAiReport } from '../../utils/aiActionSelection'
import { buildGapCheckSidebarReport } from '../../utils/actionsTabGapReport'
import {
  buildAcceptedInsertContent,
  buildInlineSuggestionHtml,
  normalizeAiActionResult,
} from '../../utils/editorAiActionShared'
import { buildActionSummary } from '../../utils/actionsTabSummary'
import {
  applyEditorInlineSuggestion,
  appendGapFindingsToEditor,
  clearEditorInlineSuggestion,
  requestEditorSnapshot,
  scrollEditorToRange,
  showEditorInlineSuggestion,
  subscribeActionsTabRun,
} from '../../utils/editorActionsBridge'
import {
  AI_ACTION_TRIGGERED_BY,
  getActiveEditorDocumentId,
  hasActiveSopEditor,
} from '../../utils/editorAiBridge'
import { inferEditScope } from '../../utils/editScopeInference'
import { wantsFullSopIntent } from '../../utils/sopActionIntent'
import { runEditorGapCheck } from '../../utils/editorGapCheck'
import { sanitizeRenderedHtml } from '../../utils/aiOutputFormatter'
import { saveAssistantLastAction } from '../../utils/assistantContext'

const ACTION_TEXT_WARNING_CHARS = 7000
const INLINE_SHOWN_EVENT = 'editor-actions-inline-shown'
const INLINE_APPLIED_EVENT = 'editor-actions-inline-applied'

/**
 * Runs rewrite / improve / gap_check from chat (via dispatchActionsTabRun) and
 * shows accept/reject review UI inside the KI Assistant chat panel.
 */
function EditorChatActions({ onRunStart, onRunComplete, onRunError }) {
  const location = useLocation()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [pending, setPending] = useState(null)
  const inFlightRef = useRef(false)
  const pendingRef = useRef(null)

  useEffect(() => {
    pendingRef.current = pending
  }, [pending])

  const clearPending = useCallback((requestId) => {
    const current = pendingRef.current
    if (requestId && current?.requestId && current.requestId !== requestId) return
    if (current?.action !== 'gap_check') {
      clearEditorInlineSuggestion(current?.requestId)
    }
    setPending(null)
  }, [])

  const runGapCheck = useCallback(async (instructionText = '', targetOptions = {}) => {
    const instruction = String(instructionText || '').trim()
    const { target, result, normalized } = await runEditorGapCheck({
      instruction,
      targetOptions,
      triggeredBy: AI_ACTION_TRIGGERED_BY.KL_ASSISTANT,
    })
    const report = buildGapCheckSidebarReport(result)
    const plainReport = report.sections
      .map((section) => {
        const body = section.body ? `${section.title}\n${section.body}` : section.title
        const gaps = (section.gapItems || [])
          .map((gap) => `- ${gap.issue}${gap.recommendation ? ` -> ${gap.recommendation}` : ''}`)
          .join('\n')
        return [body, gaps].filter(Boolean).join('\n')
      })
      .filter(Boolean)
      .join('\n\n')

    setPending({
      requestId: `gap-${Date.now().toString(36)}`,
      action: 'gap_check',
      sectionName: target.sectionName,
      isFullDoc: Boolean(target.isFullDoc),
      gapReport: report,
      previewHtml: normalized.suggestedHtml,
      range: { from: target.from, to: target.to },
      appendHtml: normalized.suggestedHtml,
    })
    saveAssistantLastAction({
      action: 'gap_check',
      target_scope: target.isFullDoc ? 'full_document' : 'section',
      section_name: target.sectionName || '',
      sop_id: getActiveEditorDocumentId(),
      request_prompt: instruction,
      original_text_excerpt: target.text || '',
      suggested_text_excerpt: plainReport || normalized.suggestedPlain || '',
      status: 'suggested',
      source: 'editor_actions_tab',
    })
  }, [])

  const runInlineContentAction = useCallback(async (action, instructionText = '', targetOptions = {}) => {
    const documentId = getActiveEditorDocumentId()
    const instruction = String(instructionText || '').trim()
    const snapshot = await requestEditorSnapshot({
      prompt: instruction,
      userPrompt: targetOptions.userPrompt || '',
      sectionHint: targetOptions.sectionHint || '',
      targetScope: targetOptions.targetScope || '',
      lineNumber: targetOptions.lineNumber ?? null,
      recordId: targetOptions.recordId || '',
      preferFullSection: Boolean(targetOptions.preferFullSection),
    })
    let target = snapshot.target
    if (target?.from == null || target?.to == null || !target?.text) {
      throw new Error(snapshot.error || 'Could not find that heading or paragraph in the open SOP.')
    }

    const override = targetOptions.sourceContentOverride
    if (override?.enabled && override?.content) {
      target = {
        ...target,
        text: String(override.content).trim(),
        sectionName: target.sectionName || targetOptions.sectionHint || 'Refined section',
        sectionType: 'Section',
      }
    }

    if (selectionLooksLikeFormattedAiReport(target.text)) {
      throw new Error('That region looks like AI report output. Select original SOP prose instead.')
    }

    if (target.text.length > ACTION_TEXT_WARNING_CHARS) {
      const proceed = window.confirm('This section is long and may time out. Continue?')
      if (!proceed) return
    }

    scrollEditorToRange(target.from, target.to)

    const docSize = snapshot.docSize || target.to
    const selectedFraction =
      !target.isFullDoc && docSize > 0
        ? Math.abs(target.to - target.from) / docSize
        : target.isFullDoc
          ? 1
          : 0.3

    const result = await performAIAction({
      action,
      text: target.text,
      document_id: documentId,
      instruction,
      section_id: `${target.from}-${target.to}`,
      sop_title: snapshot.sopTitle || 'Untitled SOP',
      section_name: target.sectionName || 'Selected text',
      section_type: target.isFullDoc || wantsFullSopIntent(instruction)
        ? 'Full Document'
        : target.sectionType || 'Paragraph',
      edit_scope: target.isFullDoc || wantsFullSopIntent(instruction)
        ? 'full_document'
        : inferEditScope({
            text: target.text,
            from: target.from,
            to: target.to,
            docSize: snapshot.docSize || target.to,
            instruction,
          }),
      sop_entity_id: documentId,
      triggered_by: AI_ACTION_TRIGGERED_BY.KL_ASSISTANT,
      learn_to_profile: /\b(?:learn|save|update)\b[\s\S]*\b(?:profile|style)\b/i.test(instruction),
    })

    const normalized = normalizeAiActionResult(action, result)
    if (!normalized.suggestedPlain) {
      throw new Error('No suggestion returned.')
    }

    const acceptedContent = buildAcceptedInsertContent(normalized.raw, {
      selectedFraction,
      isFullDoc: Boolean(target.isFullDoc),
    })
    const inlineHtml = buildInlineSuggestionHtml(normalized)
    const requestId = `actions-${Date.now().toString(36)}`

    await new Promise((resolve, reject) => {
      const timer = window.setTimeout(() => {
        window.removeEventListener(INLINE_SHOWN_EVENT, onShow)
        reject(new Error('Could not show inline suggestion at the target location.'))
      }, 12000)

      const onShow = (event) => {
        if (event.detail?.requestId !== requestId) return
        window.clearTimeout(timer)
        window.removeEventListener(INLINE_SHOWN_EVENT, onShow)
        resolve(event.detail || {})
      }
      window.addEventListener(INLINE_SHOWN_EVENT, onShow)
      showEditorInlineSuggestion({
        requestId,
        from: target.from,
        to: target.to,
        originalText: target.text,
        suggestedPlain: normalized.suggestedPlain,
        suggestedHtml: inlineHtml,
        structuredData: normalized.structured,
        action,
        isFullDoc: Boolean(target.isFullDoc),
        acceptedContent,
        selectedFraction,
      })
    })

    setPending({
      requestId,
      action,
      sectionName: target.sectionName,
      isFullDoc: Boolean(target.isFullDoc),
      summarySections: buildActionSummary(action, result),
      previewHtml: inlineHtml,
      suggestedPlain: normalized.suggestedPlain || '',
      originalText: target.text || '',
      range: { from: target.from, to: target.to },
    })
    saveAssistantLastAction({
      action,
      target_scope: target.isFullDoc ? 'full_document' : 'section',
      section_name: target.sectionName || '',
      sop_id: documentId,
      request_prompt: instruction,
      original_text_excerpt: target.text || '',
      suggested_text_excerpt: normalized.suggestedPlain || '',
      status: 'suggested',
      source: 'editor_actions_tab',
    })
  }, [])

  const runAction = useCallback(async (action, instructionText = '', targetOptions = {}) => {
    if (inFlightRef.current) return
    if (!hasActiveSopEditor(location.pathname)) {
      const msg = 'Open an SOP in the editor first.'
      setError(msg)
      onRunError?.(msg)
      return
    }

    if (!getActiveEditorDocumentId()) {
      const msg = 'No active SOP. Open a document in the editor first.'
      setError(msg)
      onRunError?.(msg)
      return
    }

    const instruction = String(instructionText || '').trim()
    if (!instruction) {
      const msg = 'Describe what to run on the open SOP (e.g. rewrite this section, gap check CAPAs).'
      setError(msg)
      onRunError?.(msg)
      return
    }

    clearPending()
    inFlightRef.current = true
    setLoading(true)
    setError('')
    onRunStart?.(action, instruction)

    try {
      if (action === 'gap_check') {
        await runGapCheck(instruction, targetOptions)
      } else if (action === 'rewrite' || action === 'improve' || action === 'summarize') {
        await runInlineContentAction(action, instruction, targetOptions)
      } else {
        throw new Error(`Unsupported inline action: ${action}`)
      }
      onRunComplete?.(action, instruction)
    } catch (err) {
      const msg = err?.message || 'Action failed.'
      setError(msg)
      clearPending()
      onRunError?.(msg)
    } finally {
      inFlightRef.current = false
      setLoading(false)
    }
  }, [location.pathname, clearPending, runGapCheck, runInlineContentAction, onRunStart, onRunComplete, onRunError])

  const runActionWithTarget = useCallback(
    (action, instructionText, targetOptions = {}) => {
      if (action === 'gap_check') {
        return runAction('gap_check', instructionText, targetOptions)
      }
      return runAction(action, instructionText, targetOptions)
    },
    [runAction],
  )

  useEffect(() => {
    const unsubscribe = subscribeActionsTabRun(({
      action,
      prompt: runPrompt,
      userPrompt,
      sectionHint,
      targetScope,
      lineNumber,
      recordId,
      preferFullSection,
      sourceContentOverride,
    }) => {
      const normalizedAction =
        action === 'gap_check'
          ? 'gap_check'
          : action === 'improve'
            ? 'improve'
            : action === 'summarize'
              ? 'summarize'
              : 'rewrite'
      runActionWithTarget(normalizedAction, runPrompt || '', {
        userPrompt: userPrompt || '',
        sectionHint: sectionHint || recordId || '',
        targetScope: targetScope || '',
        lineNumber: lineNumber ?? null,
        recordId: recordId || '',
        preferFullSection: Boolean(preferFullSection),
        sourceContentOverride: sourceContentOverride || null,
      })
    })
    return unsubscribe
  }, [runActionWithTarget])

  const handleAccept = useCallback(() => {
    if (!pending?.requestId || pending.action === 'gap_check') return
    applyEditorInlineSuggestion(pending.requestId)
  }, [pending])

  const handleAppendGap = useCallback(() => {
    if (!pending?.appendHtml) return
    appendGapFindingsToEditor(pending.appendHtml)
    saveAssistantLastAction({
      action: 'gap_check',
      target_scope: pending?.isFullDoc ? 'full_document' : 'section',
      section_name: pending?.sectionName || '',
      sop_id: getActiveEditorDocumentId(),
      suggested_text_excerpt: pending?.gapReport?.analysisPlain || '',
      status: 'applied',
      source: 'editor_actions_tab',
    })
    setPending(null)
    setError('')
  }, [pending])

  const handleReject = useCallback(() => {
    clearPending()
  }, [clearPending])

  useEffect(() => {
    const onApplied = (event) => {
      const { requestId, ok, message } = event.detail || {}
      if (!pendingRef.current || pendingRef.current.requestId !== requestId) return
      if (!ok) {
        setError(message || 'Could not apply suggestion.')
        return
      }
      saveAssistantLastAction({
        action: pendingRef.current.action,
        target_scope: pendingRef.current.isFullDoc ? 'full_document' : 'section',
        section_name: pendingRef.current.sectionName || '',
        sop_id: getActiveEditorDocumentId(),
        original_text_excerpt: pendingRef.current.originalText || '',
        suggested_text_excerpt: pendingRef.current.suggestedPlain || pendingRef.current.previewHtml || '',
        status: 'applied',
        source: 'editor_actions_tab',
      })
      setPending(null)
      setError('')
    }
    window.addEventListener(INLINE_APPLIED_EVENT, onApplied)
    return () => window.removeEventListener(INLINE_APPLIED_EVENT, onApplied)
  }, [])

  useEffect(() => () => clearPending(), [clearPending])

  if (!loading && !error && !pending) return null

  const isGapPending = pending?.action === 'gap_check'

  return (
    <div className="ai-chat-editor-actions">
      {loading ? <p className="ai-actions-tab__status" role="status">Running editor action…</p> : null}
      {error ? <p className="ai-actions-tab__error" role="alert">{error}</p> : null}

      {pending ? (
        <div className={`ai-actions-tab__review${isGapPending ? ' ai-actions-tab__review--gap' : ''}`}>
          <div className="ai-actions-tab__review-header">
            <h4 className="ai-actions-tab__review-title">
              {isGapPending ? 'Gap check report' : 'Review at target location'}
            </h4>
            <span className="ai-actions-tab__review-scope">{pending.sectionName}</span>
          </div>

          {!isGapPending ? (
            <p className="ai-actions-tab__pending-hint">
              In the editor: <span className="ai-actions-tab__strike-sample">removed</span> →
              <span className="ai-actions-tab__add-sample"> suggested</span>. Accept replaces only that range.
            </p>
          ) : (
            <p className="ai-actions-tab__pending-hint">
              Full compliance gap analysis for this scope. Accept appends findings to the SOP.
            </p>
          )}

          {isGapPending && pending.gapReport?.sections?.map((section) => (
            <div key={section.id} className="ai-actions-tab__summary-block ai-actions-tab__gap-block">
              <h5 className="ai-actions-tab__summary-title">{section.title}</h5>
              {section.body ? <p className="ai-actions-tab__summary-body ai-actions-tab__gap-body">{section.body}</p> : null}
              {section.gapItems?.map((gap, index) => (
                <div key={`${section.id}-gap-${index}`} className="ai-actions-tab__gap-item">
                  <p className="ai-actions-tab__gap-issue">{gap.issue}</p>
                  {gap.explanation ? <p className="ai-actions-tab__gap-meta">{gap.explanation}</p> : null}
                  {gap.recommendation ? <p className="ai-actions-tab__gap-rec">{gap.recommendation}</p> : null}
                </div>
              ))}
            </div>
          ))}

          {!isGapPending
            ? pending.summarySections?.map((section) => (
                <div key={section.id} className="ai-actions-tab__summary-block">
                  <h5 className="ai-actions-tab__summary-title">{section.title}</h5>
                  {section.body ? <p className="ai-actions-tab__summary-body">{section.body}</p> : null}
                  {section.items?.length ? (
                    <ul className="ai-actions-tab__summary-list">
                      {section.items.map((item, index) => (
                        <li key={`${section.id}-${index}`}>{item}</li>
                      ))}
                    </ul>
                  ) : null}
                </div>
              ))
            : null}

          {pending.previewHtml ? (
            <details className="ai-actions-tab__preview" open={isGapPending}>
              <summary>{isGapPending ? 'Full formatted report' : 'Preview new SOP text'}</summary>
              <div
                className="ai-actions-tab__preview-html tiptap ai-actions-tab__gap-html"
                dangerouslySetInnerHTML={{ __html: sanitizeRenderedHtml(pending.previewHtml) }}
              />
            </details>
          ) : null}

          <div className="ai-actions-tab__decision" role="group" aria-label={isGapPending ? 'Gap check actions' : 'Accept or reject'}>
            {isGapPending ? (
              <>
                <button type="button" className="ai-inline-suggestion-toolbar__btn ai-inline-suggestion-toolbar__btn--reject" onClick={handleReject}>
                  <X size={14} />
                  Close
                </button>
                <button
                  type="button"
                  className="ai-inline-suggestion-toolbar__btn ai-inline-suggestion-toolbar__btn--accept"
                  onClick={handleAppendGap}
                >
                  <Check size={14} />
                  Append to SOP
                </button>
              </>
            ) : (
              <>
                <button type="button" className="ai-inline-suggestion-toolbar__btn ai-inline-suggestion-toolbar__btn--reject" onClick={handleReject}>
                  <X size={14} />
                  Reject
                </button>
                <button type="button" className="ai-inline-suggestion-toolbar__btn ai-inline-suggestion-toolbar__btn--accept" onClick={handleAccept}>
                  <Check size={14} />
                  Accept
                </button>
              </>
            )}
          </div>
        </div>
      ) : null}
    </div>
  )
}

export default memo(EditorChatActions)
