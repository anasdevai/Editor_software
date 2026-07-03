import { Plugin, PluginKey } from '@tiptap/pm/state'
import { Decoration, DecorationSet } from '@tiptap/pm/view'

export const inlineAiSuggestionKey = new PluginKey('inlineAiSuggestion')

function addTableRangeDecorations(state, from, to, decorations, className) {
  const safeFrom = Math.max(0, Math.min(from, state.doc.content.size))
  const safeTo = Math.max(safeFrom, Math.min(to, state.doc.content.size))
  if (safeTo <= safeFrom) return

  try {
    state.doc.nodesBetween(safeFrom, safeTo, (node, pos) => {
      const nodeName = node.type?.name
      const nodeFrom = pos
      const nodeTo = pos + node.nodeSize
      const overlaps = nodeTo >= safeFrom && nodeFrom <= safeTo
      if (!overlaps) return true

      if (nodeName === 'table') {
        decorations.push(Decoration.node(nodeFrom, nodeTo, { class: className }))
        return true
      }
      if (nodeName === 'tableCell' || nodeName === 'tableHeader') {
        decorations.push(Decoration.node(nodeFrom, nodeTo, { class: `${className}__cell` }))
        return false
      }
      return true
    })
  } catch {
    // Decorations are best-effort and must never block editor rendering.
  }
}

function buildDecorationSet(state, active) {
  const decorations = []
  const { from: selectionFrom, to: selectionTo, empty } = state.selection || {}
  if (!empty && selectionTo > selectionFrom) {
    addTableRangeDecorations(state, selectionFrom, selectionTo, decorations, 'pm-table-selection-active')
  }

  if (!active) return decorations.length ? DecorationSet.create(state.doc, decorations) : DecorationSet.empty

  const { from, to, suggestedPlain, suggestedHtml } = active
  const safeFrom = Math.max(0, Math.min(from, state.doc.content.size))
  const safeTo = Math.max(safeFrom, Math.min(to, state.doc.content.size))

  if (safeTo > safeFrom) {
    decorations.push(
      Decoration.inline(safeFrom, safeTo, {
        class: 'ai-inline-diff-removed',
        'data-ai-diff': 'removed',
      }),
    )
    addTableRangeDecorations(state, safeFrom, safeTo, decorations, 'ai-inline-diff-table-target')
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
