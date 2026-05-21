import React, { memo, Suspense, lazy } from 'react'
import { EditorContent } from '@tiptap/react'
import EditorAIBridge from './EditorAIBridge'

const AIAssistantBubbleMenu = lazy(() => import('./AIAssistantBubbleMenu'))

/**
 * Memoized TipTap surface — isolates ProseMirror DOM from unrelated EditorPage state
 * (autosave timestamps, sidebar metadata, etc.).
 */
const EditorTypingSurface = memo(function EditorTypingSurface({
  editor,
  isEditable,
  aiSopContext,
  documentId,
  onPreviewSessionChange,
  onAfterApply,
  onVersionCompareRequest,
}) {
  if (!editor || editor.isDestroyed) return null

  return (
    <div className="figma-paper editor-typing-surface">
      <EditorContent editor={editor} />
      <Suspense fallback={null}>
        <AIAssistantBubbleMenu
          editor={editor}
          sopMetadata={aiSopContext}
          isEditable={isEditable}
          onPreviewSessionChange={onPreviewSessionChange}
        />
      </Suspense>
      <EditorAIBridge
        editor={editor}
        documentId={documentId}
        sopMetadata={aiSopContext}
        isEditable={isEditable}
        onPreviewSessionChange={onPreviewSessionChange}
        onAfterApply={onAfterApply}
        onVersionCompareRequest={onVersionCompareRequest}
      />
    </div>
  )
})

export default EditorTypingSurface
