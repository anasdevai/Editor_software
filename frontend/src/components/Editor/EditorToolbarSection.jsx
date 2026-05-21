import React from 'react'
import {
  Bold,
  ChevronDown,
  Import,
  Italic,
  List,
  ListOrdered,
  Loader2,
  Redo2,
  Strikethrough,
  Table2,
  Underline as UnderlineIcon,
  Undo2,
} from 'lucide-react'

export default function EditorToolbarSection({
  versionSelectValue,
  versions,
  loadVersionHandler,
  buildVersionLabel,
  onPreviewOpen,
  onCreateVersion,
  onSave,
  isHistoricalView,
  isSaving,
  t,
  editor,
  openLinkModal,
  compareBaseValue,
  compareTargetValue,
  setCompareBaseVersionId,
  setCompareTargetVersionId,
  openCompareViewer,
  documentId,
  compareBaseVersionId,
  compareTargetVersionId,
  isImporting,
  triggerImport,
  insertPlaceholder,
  language,
  setLanguage,
}) {
  return (
    <>
      <section className="figma-card figma-actions-bar">
        <div className="figma-version-select">
          <select value={versionSelectValue} onChange={(event) => loadVersionHandler(event.target.value)} disabled={versions.length === 0}>
            {versions.length === 0 ? <option value="">v1 (draft)</option> : null}
            {versions.map((item) => (<option key={item.id} value={item.id}>{buildVersionLabel(item)}</option>))}
          </select>
          <ChevronDown size={16} />
        </div>
        <button type="button" className="figma-btn figma-btn-muted" onClick={onPreviewOpen}>{t.previewExport}</button>
        <button type="button" className="figma-btn figma-btn-muted" onClick={onCreateVersion}>{t.newVersion}</button>
        <button type="button" className="figma-btn figma-btn-primary" onClick={onSave} disabled={isHistoricalView}>
          {isHistoricalView ? t.readOnly : isSaving ? t.saving : t.save}
        </button>
      </section>

      <section className="figma-card figma-toolbar-card">
        <div className="figma-toolbar-row">
          <button type="button" className="figma-icon-btn" onClick={() => editor.chain().focus().undo().run()} disabled={isHistoricalView}><Undo2 size={16} /></button>
          <button type="button" className="figma-icon-btn" onClick={() => editor.chain().focus().redo().run()} disabled={isHistoricalView}><Redo2 size={16} /></button>
          <div className="figma-divider" />
          <button type="button" className={`figma-icon-btn${editor.isActive('bold') ? ' active' : ''}`} onClick={() => editor.chain().focus().toggleBold().run()} disabled={isHistoricalView}><Bold size={16} /></button>
          <button type="button" className={`figma-icon-btn${editor.isActive('italic') ? ' active' : ''}`} onClick={() => editor.chain().focus().toggleItalic().run()} disabled={isHistoricalView}><Italic size={16} /></button>
          <button type="button" className={`figma-icon-btn${editor.isActive('underline') ? ' active' : ''}`} onClick={() => editor.chain().focus().toggleUnderline().run()} disabled={isHistoricalView}><UnderlineIcon size={16} /></button>
          <button type="button" className={`figma-icon-btn${editor.isActive('strike') ? ' active' : ''}`} onClick={() => editor.chain().focus().toggleStrike().run()} disabled={isHistoricalView}><Strikethrough size={16} /></button>
          <button type="button" className={`figma-icon-btn${editor.isActive('bulletList') ? ' active' : ''}`} onClick={() => editor.chain().focus().toggleBulletList().run()} disabled={isHistoricalView}><List size={16} /></button>
          <button type="button" className={`figma-icon-btn${editor.isActive('orderedList') ? ' active' : ''}`} onClick={() => editor.chain().focus().toggleOrderedList().run()} disabled={isHistoricalView}><ListOrdered size={16} /></button>
          <button type="button" className={`figma-btn figma-btn-small${editor.isActive('link') ? ' active' : ''}`} onClick={openLinkModal} disabled={isHistoricalView}>{t.insertUrl}</button>
        </div>

        <div className="figma-toolbar-row">
          <button type="button" className="figma-btn figma-btn-small" onClick={() => editor.chain().focus().insertTable({ rows: 3, cols: 3, withHeaderRow: true }).run()} disabled={isHistoricalView}><Table2 size={16} />{t.insertTable}</button>
          <button type="button" className="figma-btn figma-btn-small" onClick={() => editor.chain().focus().deleteTable().run()} disabled={isHistoricalView || !editor.can().chain().focus().deleteTable().run()}>{t.deleteTable}</button>
          <div className="figma-compare-group">
            <div className="figma-compare-select">
              <select value={compareBaseValue} onChange={(event) => setCompareBaseVersionId(event.target.value)} disabled={versions.length === 0}>
                {versions.length === 0 ? <option value="">{t.base}: v1</option> : null}
                {versions.map((item) => <option key={`base-${item.id}`} value={item.id}>{t.base}: v{item.versionNumber || 1}</option>)}
              </select>
            </div>
            <span>{t.vs}</span>
            <div className="figma-compare-select">
              <select value={compareTargetValue} onChange={(event) => setCompareTargetVersionId(event.target.value)} disabled={versions.length === 0}>
                {versions.length === 0 ? <option value="">{t.target}: v1</option> : null}
                {versions.map((item) => <option key={`target-${item.id}`} value={item.id}>{t.target}: v{item.versionNumber || 1}</option>)}
              </select>
            </div>
          </div>
          <button type="button" className="figma-btn figma-btn-primary" onClick={openCompareViewer} disabled={!documentId || !compareBaseVersionId || !compareTargetVersionId}>{t.compare}</button>
          <label className={`figma-btn figma-btn-small${isHistoricalView || isImporting ? ' disabled' : ''}`}>
            {isImporting ? <Loader2 size={15} className="figma-spin" /> : <Import size={15} />}
            {isImporting ? 'Importing SOP...' : 'Import SOP'}
            <input type="file" accept=".pdf,.docx,.txt,.md" hidden onChange={triggerImport} disabled={isHistoricalView || isImporting} />
          </label>
          <button type="button" className="figma-btn figma-btn-small" onClick={insertPlaceholder} disabled={isHistoricalView}>{t.insertPlaceholder}</button>
          <div className="figma-language-select">
            <select value={language} onChange={(event) => setLanguage(event.target.value)}>
              <option value="de">{t.german}</option>
              <option value="en">{t.english}</option>
            </select>
            <ChevronDown size={14} />
          </div>
        </div>
      </section>
    </>
  )
}
