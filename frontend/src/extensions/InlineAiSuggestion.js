import { Extension } from '@tiptap/core'
import { createInlineAiSuggestionPlugin } from '../utils/editorInlineSuggestionPlugin'

/** TipTap extension for Actions-tab inline rewrite/improve diff decorations. */
export const InlineAiSuggestion = Extension.create({
  name: 'inlineAiSuggestion',

  addProseMirrorPlugins() {
    return [createInlineAiSuggestionPlugin()]
  },
})
