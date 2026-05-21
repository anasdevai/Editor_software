/**
 * Build sidebar summary for Actions-tab rewrite/improve results.
 */

const stripHtml = (value) =>
  String(value || '')
    .replace(/<[^>]+>/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()

export function looksLikeChatBriefing(text) {
  const t = String(text || '').toLowerCase()
  if (t.length < 120) return false
  const hasSummaryDetails = t.includes('summary') && (t.includes('details') || t.includes('status'))
  const hasPrescriptiveBrief =
    /\bthe rewrite of\b/.test(t)
    && /\bmust\b/.test(t)
    && (t.includes('procedure') || t.includes('sop') || t.includes('mandatory'))
  return hasSummaryDetails || hasPrescriptiveBrief
}

export function extractSuggestedPlainForInline(action, result) {
  const structured = result?.structured_data || {}
  const actionKey = String(action || '').toLowerCase()

  let plain = ''
  if (actionKey === 'rewrite') {
    plain = stripHtml(structured.rewritten_text || '')
    if (!plain && structured.rewritten_text) {
      plain = String(structured.rewritten_text).trim()
    }
  } else if (actionKey === 'improve') {
    plain = stripHtml(structured.improved_text || structured.improved_version || '')
  }

  if (!plain) {
    const fallback = stripHtml(result?.suggested_text || '')
    if (fallback && !looksLikeChatBriefing(fallback)) {
      plain = fallback
    }
  }

  if (!plain && looksLikeChatBriefing(result?.suggested_text || result?.explanation)) {
    return {
      plain: '',
      error:
        'The server returned a narrative briefing instead of editable SOP text. Ask to rewrite the section in chat while the SOP is open in the editor.',
    }
  }

  if (!plain) {
    return { plain: '', error: 'No rewrite text was returned. Try a shorter selection or check the LLM connection.' }
  }

  return { plain, error: null }
}

export function buildActionSummary(action, result) {
  const structured = result?.structured_data || {}
  const sections = []

  const explanation = stripHtml(result?.explanation || '')
  if (explanation) {
    sections.push({ id: 'summary', title: 'What this implements', body: explanation })
  }

  const changes = Array.isArray(structured.structural_changes)
    ? structured.structural_changes
    : structured.structural_changes
      ? [structured.structural_changes]
      : Array.isArray(structured.changes_made)
        ? structured.changes_made
        : []

  if (changes.length > 0) {
    sections.push({
      id: 'changes',
      title: action === 'improve' ? 'Improvements' : 'Structural changes',
      items: changes.map((c) => stripHtml(c)).filter(Boolean),
    })
  }

  const rationale = stripHtml(structured.rationale || structured.compliance_note || '')
  if (rationale) {
    sections.push({ id: 'rationale', title: 'Rationale', body: rationale })
  }

  if (structured.rewrite_mode) {
    sections.push({
      id: 'mode',
      title: 'Mode',
      body: String(structured.rewrite_mode),
    })
  }

  return sections
}
