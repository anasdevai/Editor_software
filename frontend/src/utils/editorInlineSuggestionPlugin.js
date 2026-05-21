import { Plugin, PluginKey } from '@tiptap/pm/state'
import { Decoration, DecorationSet } from '@tiptap/pm/view'

export const inlineAiSuggestionKey = new PluginKey('inlineAiSuggestion')

function buildDecorationSet(state, active) {
  if (!active) return DecorationSet.empty

  const { from, to, suggestedPlain, suggestedHtml } = active
  const safeFrom = Math.max(0, Math.min(from, state.doc.content.size))
  const safeTo = Math.max(safeFrom, Math.min(to, state.doc.content.size))
  const decorations = []

  if (safeTo > safeFrom) {
    decorations.push(
      Decoration.inline(safeFrom, safeTo, {
        class: 'ai-inline-diff-removed',
        'data-ai-diff': 'removed',
      }),
    )
  }

  const insertion = document.createElement('div')
  insertion.className = 'ai-inline-diff-added-block tiptap'
  insertion.setAttribute('data-ai-diff', 'added')
  if (suggestedHtml && /<\/?[a-z]/i.test(String(suggestedHtml))) {
    insertion.innerHTML = String(suggestedHtml)
  } else {
    insertion.textContent = String(suggestedPlain || '')
  }

  decorations.push(
    Decoration.widget(safeTo, insertion, {
      side: 1,
      key: `ai-suggestion-${safeFrom}-${safeTo}`,
      stopEvent: () => true,
    }),
  )

  return DecorationSet.create(state.doc, decorations)
}

export function createInlineAiSuggestionPlugin() {
  return new Plugin({
    key: inlineAiSuggestionKey,
    state: {
      init() {
        return { active: null }
      },
      apply(tr, value) {
        const meta = tr.getMeta(inlineAiSuggestionKey)
        if (meta?.type === 'clear') {
          return { active: null }
        }
        if (meta?.type === 'set' && meta.payload) {
          return { active: meta.payload }
        }
        if (!value.active) return value
        // User typing invalidates inline preview — remapping decorations on every
        // keystroke caused decoration rebuilds and visible editor flicker.
        if (tr.docChanged && !meta) {
          return { active: null }
        }
        return value
      },
    },
    props: {
      decorations(state) {
        const pluginState = inlineAiSuggestionKey.getState(state)
        return buildDecorationSet(state, pluginState?.active)
      },
    },
  })
}

export function setInlineAiSuggestion(editor, payload) {
  if (!editor || editor.isDestroyed) return false
  const tr = editor.state.tr
    .setMeta(inlineAiSuggestionKey, { type: 'set', payload })
    .setMeta('addToHistory', false)
  editor.view.dispatch(tr)
  return true
}

export function clearInlineAiSuggestion(editor) {
  if (!editor || editor.isDestroyed) return false
  const tr = editor.state.tr.setMeta(inlineAiSuggestionKey, { type: 'clear' })
  editor.view.dispatch(tr)
  return true
}

export function getInlineAiSuggestionState(editor) {
  if (!editor || editor.isDestroyed) return null
  return inlineAiSuggestionKey.getState(editor.state)?.active || null
}
