import React, { memo, useEffect, useState } from 'react'
import { debounce } from 'lodash'
import StatusBar from './StatusBar'

function computeDocStats(editor) {
  if (!editor || editor.isDestroyed) {
    return { wordCount: 0, charCount: 0, blockCount: 0 }
  }
  const plainText = editor.getText() || ''
  let blockCount = 0
  editor.state.doc.descendants((node) => {
    if (
      node.type?.name
      && ['paragraph', 'heading', 'bulletList', 'orderedList', 'table'].includes(node.type.name)
    ) {
      blockCount += 1
    }
  })
  return {
    wordCount: plainText.split(/\s+/).filter(Boolean).length,
    charCount: plainText.length,
    blockCount,
  }
}

/**
 * Status bar with document stats driven by debounced editor events —
 * avoids scanning the doc on every parent React render (reduces flicker while typing).
 */
function EditorDocStatsBar({ editor, lastSaved, isSaving, profile, onProfileChange, workflowStatus }) {
  const [stats, setStats] = useState(() => computeDocStats(editor))

  useEffect(() => {
    if (!editor || editor.isDestroyed) return undefined

    const pushStats = debounce(() => {
      setStats(computeDocStats(editor))
    }, 280)

    setStats(computeDocStats(editor))
    editor.on('update', pushStats)

    return () => {
      editor.off('update', pushStats)
      pushStats.cancel()
    }
  }, [editor])

  return (
    <StatusBar
      wordCount={stats.wordCount}
      charCount={stats.charCount}
      blockCount={stats.blockCount}
      lastSaved={lastSaved}
      isSaving={isSaving}
      profile={profile}
      onProfileChange={onProfileChange}
      workflowStatus={workflowStatus}
    />
  )
}

export default memo(EditorDocStatsBar)
