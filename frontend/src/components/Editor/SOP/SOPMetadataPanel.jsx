import { Settings } from 'lucide-react'
import { useLanguage } from '../../../context/LanguageContext'
import { useSOPConfig } from '../../../context/SOPConfigContext'
import './SOPMetadataPanel.css'

const FIELD_GROUPS = [
    {
        title: 'Basic Info',
        keys: ['documentId', 'title', 'docType', 'category', 'department'],
    },
    {
        title: 'Revision',
        keys: ['sopVersion', 'sopStatus', 'author', 'reviewer'],
    },
    {
        title: 'Dates',
        keys: ['effectiveDate', 'reviewDate'],
    },
    {
        title: 'Compliance',
        keys: ['riskLevel', 'regulatoryReferences'],
    },
]

const REQUIRED_FIELD_DEFS = [
    { key: 'documentId', type: 'text', label: 'Document ID / SOP ID', aliases: ['sop_id', 'sopId', 'document_id'] },
    { key: 'title', type: 'text', label: 'Title' },
    { key: 'docType', type: 'text', label: 'Document Type', aliases: ['doc_type', 'type', 'documentType'] },
    { key: 'category', type: 'text', label: 'Category' },
    { key: 'department', type: 'text', label: 'Department' },
    { key: 'sopVersion', type: 'text', label: 'Document Revision', aliases: ['version', 'document_revision', 'revision'] },
    { key: 'sopStatus', type: 'text', label: 'Status', source: 'status' },
    { key: 'author', type: 'text', label: 'Author' },
    { key: 'reviewer', type: 'text', label: 'Reviewer' },
    { key: 'effectiveDate', type: 'date', label: 'Effective Date', aliases: ['effective_date', 'date'] },
    { key: 'reviewDate', type: 'date', label: 'Review Date', aliases: ['review_date'] },
    { key: 'riskLevel', type: 'text', label: 'Risk Level', aliases: ['risk_level'] },
    {
        key: 'regulatoryReferences',
        type: 'textarea',
        label: 'Regulatory References',
        aliases: ['regulatory_references'],
        multiValue: true,
        separator: '\n',
    },
]

const EXTRA_FIELD_DEFS = [
    { key: 'roles', type: 'textarea', label: 'Roles', aliases: ['role'] },
    { key: 'workflow', type: 'textarea', label: 'Workflow' },
    { key: 'complianceElements', type: 'textarea', label: 'Compliance Elements', aliases: ['compliance_elements'] },
    { key: 'risks', type: 'textarea', label: 'Risks' },
    { key: 'gaps', type: 'textarea', label: 'Gaps' },
    { key: 'terminologyKeywords', type: 'textarea', label: 'Terminology / Keywords', aliases: ['terminology_keywords', 'keywords', 'terminology'] },
]

const hasValue = (value) => {
    if (Array.isArray(value)) return value.length > 0
    return value !== undefined && value !== null && String(value).trim() !== ''
}

const formatStatusForDisplay = (value) => {
    const normalized = String(value || '').trim().toLowerCase().replace(/[\s-]+/g, '_')
    const map = {
        draft: 'Draft',
        under_review: 'Under Review',
        changes_requested: 'Changes Requested',
        accepted: 'Accepted',
        rejected: 'Rejected',
        approved: 'Approved',
        effective: 'Effective',
        obsolete: 'Obsolete',
    }
    return map[normalized] || String(value || '').trim()
}

export default function SOPMetadataPanel({
    metadata,
    onChange,
    status = '',
    onStatusChange,
    isReadOnly = false,
    errors = {},
}) {
    const { t } = useLanguage()
    const config = useSOPConfig()
    const isDev = Boolean(import.meta?.env?.DEV)
    const metadataView = metadata?.sopMetadata && typeof metadata.sopMetadata === 'object'
        ? { ...metadata, ...metadata.sopMetadata }
        : (metadata || {})

    const handleFieldChange = (key, value) => {
        onChange?.({
            ...metadata,
            [key]: value,
        })
    }

    const configMetadataFields = Array.isArray(config?.metadataFields) ? config.metadataFields : []
    const configFieldByKey = new Map(configMetadataFields.map((field) => [field.key, field]))
    const requiredFields = REQUIRED_FIELD_DEFS.map((fallbackField) => ({
        ...fallbackField,
        ...(configFieldByKey.get(fallbackField.key) || {}),
        aliases: fallbackField.aliases || configFieldByKey.get(fallbackField.key)?.aliases || [],
        source: fallbackField.source || configFieldByKey.get(fallbackField.key)?.source,
        label: fallbackField.label || configFieldByKey.get(fallbackField.key)?.label || fallbackField.key,
    }))
    const extraFields = EXTRA_FIELD_DEFS.filter((field) =>
        [field.key, ...(field.aliases || [])].some((key) => hasValue(metadataView?.[key])),
    )
    const requiredFieldByKey = new Map(requiredFields.map((field) => [field.key, field]))

    const getFieldValue = (fieldDef) => {
        if (fieldDef.source === 'status') return formatStatusForDisplay(status)
        const keys = [fieldDef.key, ...(fieldDef.aliases || [])]
        for (const key of keys) {
            const value = metadataView?.[key]
            if (hasValue(value)) return value
        }
        return fieldDef.multiValue ? [] : ''
    }

    const handlePanelFieldChange = (fieldDef, value) => {
        if (fieldDef.source === 'status') {
            onStatusChange?.(value)
            return
        }
        handleFieldChange(fieldDef.key, value)
    }

    /**
     * Render a single metadata field based on its config definition.
     */
    const renderField = (fieldDef) => {
        const { key, type, label, multiValue, separator } = fieldDef
        const hasError = errors[key]
        const displayLabel = t[label] || label

        let inputElement = null

        // Multi-value textarea (e.g. regulatoryReferences)
        if (type === 'textarea' && multiValue) {
            const fieldValue = getFieldValue(fieldDef)
            const arrayValue = Array.isArray(fieldValue) ? fieldValue : String(fieldValue || '').split(separator || '\n').filter(Boolean)
            inputElement = (
                <textarea
                    placeholder={displayLabel}
                    className="sidebar-textarea sop-textarea"
                    value={arrayValue.join(separator || '\n')}
                    onChange={(e) =>
                        handlePanelFieldChange(
                            fieldDef,
                            e.target.value
                                .split(separator || '\n')
                                .map((item) => item.trim())
                                .filter(Boolean)
                        )
                    }
                    disabled={isReadOnly}
                    rows={4}
                />
            )
        } else if (type === 'textarea') {
            // Regular textarea
            inputElement = (
                <textarea
                    placeholder={displayLabel}
                    className="sidebar-textarea sop-textarea"
                    value={getFieldValue(fieldDef) || ''}
                    onChange={(e) => handlePanelFieldChange(fieldDef, e.target.value)}
                    disabled={isReadOnly}
                    rows={4}
                />
            )
        } else if (type === 'date') {
            // Date input
            inputElement = (
                <input
                    type="date"
                    className="sidebar-input sop-input"
                    value={getFieldValue(fieldDef) || ''}
                    placeholder="mm/dd/yyyy"
                    onChange={(e) => handlePanelFieldChange(fieldDef, e.target.value)}
                    disabled={isReadOnly}
                />
            )
        } else {
            // Default: text input
            inputElement = (
                <input
                    type="text"
                    className="sidebar-input sop-input"
                    placeholder={displayLabel}
                    value={getFieldValue(fieldDef) || ''}
                    onChange={(e) => handlePanelFieldChange(fieldDef, e.target.value)}
                    disabled={isReadOnly}
                />
            )
        }

        return (
            <div key={key} className={`sop-field-group sop-field-${key}${type === 'textarea' ? ' sop-field-wide' : ''}`}>
                <label className="sop-field-label">{displayLabel}</label>
                {inputElement}
                {hasError && (
                    <p className="error-text">{hasError}</p>
                )}
            </div>
        )
    }

    const fallbackFieldByKey = new Map(REQUIRED_FIELD_DEFS.map((field) => [field.key, field]))
    const guaranteedSections = FIELD_GROUPS.map((group) => ({
        title: group.title,
        fields: group.keys
            .map((key) => requiredFieldByKey.get(key) || fallbackFieldByKey.get(key))
            .filter(Boolean),
    }))
    const guaranteedFieldKeys = guaranteedSections.flatMap((section) => section.fields.map((field) => field.key))
    const requiredFieldPresence = guaranteedFieldKeys.reduce((acc, key) => {
        const field = requiredFieldByKey.get(key) || fallbackFieldByKey.get(key)
        const aliases = field?.aliases || []
        const candidateKeys = [key, ...aliases]
        acc[key] = candidateKeys.some((candidate) => hasValue(metadataView?.[candidate]))
        return acc
    }, {})

    if (isDev) {
        console.log('[SOPMetadataPanel] render', {
            FIELD_GROUPS,
            renderedSections: guaranteedSections.map((s) => ({ title: s.title, keys: s.fields.map((f) => f.key) })),
            renderedSectionCount: guaranteedSections.length,
            metadataValues: {
                documentId: metadataView?.documentId || metadataView?.sop_id || metadataView?.document_id || '',
                title: metadataView?.title || '',
                docType: metadataView?.docType || metadataView?.doc_type || metadataView?.documentType || '',
                category: metadataView?.category || '',
                department: metadataView?.department || '',
                sopVersion: metadataView?.sopVersion || metadataView?.version || metadataView?.document_revision || '',
                sopStatus: status || metadataView?.sopStatus || metadataView?.status || '',
                author: metadataView?.author || '',
                reviewer: metadataView?.reviewer || '',
                effectiveDate: metadataView?.effectiveDate || metadataView?.effective_date || '',
                reviewDate: metadataView?.reviewDate || metadataView?.review_date || '',
                riskLevel: metadataView?.riskLevel || metadataView?.risk_level || '',
                regulatoryReferences: metadataView?.regulatoryReferences || metadataView?.regulatory_references || [],
            },
            requiredFieldPresence,
            finalJsxSectionCount: guaranteedSections.length + (extraFields.length > 0 ? 1 : 0),
        })
        console.debug('[SOP Status Debug] SOPMetadataPanel status prop/render', {
            statusProp: status,
            renderedStatusField: formatStatusForDisplay(status),
        })
    }

    return (
        <div className="sidebar-card sidebar-card-emphasis sop-panel-card">
            <h3 className="sidebar-title sop-panel-title">
                <Settings size={18} />
                {t.sopMetadata}
            </h3>
            {isDev ? (
                <div
                    style={{
                        marginBottom: 10,
                        fontSize: 11,
                        color: '#6b7280',
                        fontWeight: 600,
                    }}
                >
                    DEBUG: MetadataPanel body mounted
                </div>
            ) : null}

            <div className="sop-metadata-sections">
                {guaranteedSections.map((group) => (
                    <section key={group.title} className="sop-metadata-section">
                        <div className="sop-metadata-section-title">{group.title}</div>
                        <div className="sop-metadata-grid">
                            {group.fields.map(renderField)}
                        </div>
                    </section>
                ))}
                {extraFields.length > 0 ? (
                    <section className="sop-metadata-section">
                        <div className="sop-metadata-section-title">Detected Fields</div>
                        <div className="sop-metadata-grid">
                            {extraFields.map(renderField)}
                        </div>
                    </section>
                ) : null}
            </div>
        </div>
    )
}

