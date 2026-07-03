import { extractText } from '../api/editorApi'
import { isEditorContentEmpty, mapBlocksToTipTapDoc } from './editorUtils'
import { formatOCRText } from './formatOCRText'
import { mapOCRBlocksToHTML } from './mapOCRBlocksToHTML'
import { DEFAULT_SOP_VERSION_METADATA } from './sopConstants'

export const SOP_IMPORT_ACCEPT = '.pdf,.docx,.txt,.md'

export async function extractSOPImport(file) {
  const response = await extractText(file)
  const elements = Array.isArray(response?.elements) && response.elements.length
    ? response.elements
    : null
  const blocks = Array.isArray(response?.blocks) ? response.blocks : []
  const contentSource = elements ?? blocks
  const text = response?.text || ''
  const metadata = normalizeSOPImportMetadata(response?.sop_metadata_ui)

  return {
    response,
    blocks: contentSource,
    elements: elements ?? [],
    rawBlocks: blocks,
    text,
    metadata,
    hasContent: Boolean(text.trim() || contentSource.length),
  }
}

export function validateSOPImportContent(importResult, message = 'No text content found in PDF.') {
  if (!importResult?.hasContent) {
    throw new Error(message)
  }
}

function buildImportDocJson(importResult) {
  const docJson = mapBlocksToTipTapDoc(importResult?.blocks, importResult?.text)
  if (!isEditorContentEmpty(docJson)) {
    return docJson
  }

  const fallbackDocJson = mapBlocksToTipTapDoc([], importResult?.text)
  if (!isEditorContentEmpty(fallbackDocJson)) {
    return fallbackDocJson
  }

  throw new Error('No editor-readable content extracted from file.')
}

export function normalizeSOPImportMetadata(rawMetadata) {
  if (!rawMetadata || typeof rawMetadata !== 'object') return {}
  const source =
    rawMetadata?.sopMetadata && typeof rawMetadata.sopMetadata === 'object'
      ? { ...rawMetadata, ...rawMetadata.sopMetadata }
      : { ...rawMetadata }
  const normalized = { ...source }
  const aliasMap = {
    sop_id: 'documentId',
    sopId: 'documentId',
    document_id: 'documentId',
    sop_number: 'documentId',
    external_id: 'documentId',
    doc_type: 'docType',
    document_type: 'docType',
    type: 'docType',
    documentType: 'docType',
    version: 'sopVersion',
    revision: 'sopVersion',
    document_revision: 'sopVersion',
    sop_version: 'sopVersion',
    sop_status: 'sopStatus',
    status: 'sopStatus',
    client_id: 'clientId',
    client: 'clientName',
    client_name: 'clientName',
    document_family: 'documentFamily',
    family: 'documentFamily',
    effective_date: 'effectiveDate',
    date: 'effectiveDate',
    review_date: 'reviewDate',
    risk_level: 'riskLevel',
    regulatory_references: 'regulatoryReferences',
    compliance_elements: 'complianceElements',
    terminology_keywords: 'terminologyKeywords',
    keywords: 'terminologyKeywords',
  }

  Object.entries(aliasMap).forEach(([sourceKey, targetKey]) => {
    if (!normalized[targetKey] && source[sourceKey] != null) {
      normalized[targetKey] = source[sourceKey]
    }
  })

  if (import.meta?.env?.DEV) {
    console.debug('[SOP Status Debug] normalized import metadata', {
      rawStatus: source?.status || null,
      rawSopStatus: source?.sopStatus || null,
      normalizedSopStatus: normalized?.sopStatus || null,
      documentId: normalized?.documentId || null,
      title: normalized?.title || null,
    })
  }

  return normalized
}

const escapeRegExp = (value = '') => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

export function normalizeSOPTitleForDisplay(title = '', documentId = '') {
  const rawTitle = String(title || '').trim()
  const docId = String(documentId || '').trim()
  if (!rawTitle) return ''
  if (!docId) return rawTitle

  const idPattern = escapeRegExp(docId)
  const prefixRegex = new RegExp(`^${idPattern}\\s*(?:[-–—:]\\s*)?`, 'i')
  const cleaned = rawTitle.replace(prefixRegex, '').trim()
  return cleaned || rawTitle
}

export function buildSOPDisplayLabel(metadata = {}, fallback = '') {
  const cleanTitle = normalizeSOPTitleForDisplay(metadata.title, metadata.documentId)
  return [
    metadata.documentId,
    cleanTitle,
    (metadata.sopVersion || '').trim(),
  ].filter(Boolean).join(' — ') || fallback
}

export function applySOPImportMetadata(previousMetadata, importedMetadata = {}) {
  const normalizedImported = normalizeSOPImportMetadata(importedMetadata)
  const managedFields = [
    'documentId',
    'title',
    'clientId',
    'clientName',
    'department',
    'docType',
    'category',
    'documentFamily',
    'sopVersion',
    'effectiveDate',
    'reviewDate',
    'riskLevel',
    'regulatoryReferences',
    'roles',
    'workflow',
    'complianceElements',
    'risks',
    'gaps',
    'terminologyKeywords',
  ]
  const next = { ...previousMetadata }

  // Reset managed import fields so missing extractor values become empty inputs.
  managedFields.forEach((key) => {
    next[key] = key === 'regulatoryReferences' ? [] : ''
  })

  managedFields.forEach((key) => {
    const incoming = normalizedImported[key]
    if (incoming == null) return
    if (key === 'regulatoryReferences') {
      if (Array.isArray(incoming)) {
        next[key] = incoming
      } else if (typeof incoming === 'string' && incoming.trim()) {
        next[key] = incoming.split('\n').map((item) => item.trim()).filter(Boolean)
      } else {
        next[key] = []
      }
      return
    }
    next[key] = incoming
  })

  next.title = normalizeSOPTitleForDisplay(next.title, next.documentId)

  if (!next.docType) next.docType = 'SOP'
  if (!next.documentFamily) next.documentFamily = next.docType || 'SOP'
  return next
}

export function prepareSOPMetadataJson(importedMetadata = {}, overrides = {}) {
  const resolvedSopStatus =
    importedMetadata.sopStatus
    || importedMetadata.status
    || DEFAULT_SOP_VERSION_METADATA.sopStatus
  return {
    sopStatus: resolvedSopStatus,
    sopMetadata: {
      ...DEFAULT_SOP_VERSION_METADATA.sopMetadata,
      ...importedMetadata,
      ...overrides,
    },
    auditTrail: [],
    versionNote: '',
  }
}

export async function prepareEditorSOPImport(file) {
  const importResult = await extractSOPImport(file)
  validateSOPImportContent(importResult, 'No text content found in uploaded file.')
  const docJson = buildImportDocJson(importResult)
  const html = importResult.blocks.length
    ? mapOCRBlocksToHTML(importResult.blocks, 'sop')
    : formatOCRText(importResult.text)

  if (!html || !String(html).trim()) {
    throw new Error('No structured content extracted from file.')
  }

  return {
    ...importResult,
    docJson,
    html,
    tabLabel: buildSOPDisplayLabel(importResult.metadata),
  }
}

export async function prepareNewSOPImport(file) {
  const importResult = await extractSOPImport(file)
  validateSOPImportContent(importResult)

  const fallbackTitle = file.name.replace(/\.[^/.]+$/, '') || 'Imported SOP'
  const resolvedTitle = normalizeSOPTitleForDisplay(
    importResult.metadata.title || '',
    importResult.metadata.documentId || '',
  ) || fallbackTitle
  const docJson = buildImportDocJson(importResult)

  return {
    ...importResult,
    docJson,
    resolvedTitle,
    metadataJson: prepareSOPMetadataJson(importResult.metadata, {
      author: 'System (Import)',
      reviewer: '',
    }),
    tabLabel: buildSOPDisplayLabel(importResult.metadata),
  }
}
