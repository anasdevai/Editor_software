/**
 * VersionDiffEngine.js
 * 
 * Core logic for comparing two Tiptap document JSON structures.
 * It identifies additions, deletions, and modifications at the block level
 * (paragraphs, headings, tables) based on unique 'block-id' attributes.
 */

import { compareBlocks } from './BlockComparator';
import { compareTableNodes } from './TableComparator';

/**
 * Creates a fast-lookup map of block IDs to their respective node objects.
 * @param {Array} contentArray - Array of ProseMirror/Tiptap node objects.
 * @returns {Map} A Map where keys are block-ids and values are the nodes.
 */
const buildIdMap = (contentArray = []) => {
    const map = new Map();
    contentArray.forEach((node, index) => {
        const blockId = node?.attrs?.['block-id'];
        if (blockId) map.set(blockId, { node, index });
    });
    return map;
};

/**
 * Compares an old document JSON against a new document JSON to generate a unified 
 * difference view structure. Iterates through the new document to find matches in 
 * the old document, falling back to marking nodes as 'added' or 'removed'.
 * 
 * @param {Object} oldJson - The baseline Tiptap document JSON.
 * @param {Object} newJson - The current/target Tiptap document JSON.
 * @returns {Object} A new document JSON structurally containing 'diffStatus' metadata.
 */
export const generateDocumentDiff = (oldJson, newJson) => {
    // Return empty document if either input is invalid
    if (!oldJson || !newJson) return { type: 'doc', content: [] };

    const oldContent = oldJson.content || [];
    const newContent = newJson.content || [];

    // Map old nodes by their block-id for O(1) lookups
    const oldMap = buildIdMap(oldContent);
    // Track which old nodes have been checked to find deletions later
    const processedOldIndexes = new Set();
    const diffResult = [];

    const compareNodePair = (oldNode, newNode) => {
        if (newNode?.type === 'table') {
            const tableDiff = compareTableNodes(oldNode, newNode);
            return tableDiff.isChanged ? { ...tableDiff.node, diffStatus: 'modified' } : newNode;
        }
        const diff = compareBlocks(oldNode, newNode);
        return diff.isChanged ? { ...diff.node, diffStatus: 'modified' } : newNode;
    };

    const findFallbackOldIndex = (startIndex, preferredType) => {
        for (let i = startIndex; i < oldContent.length; i++) {
            if (processedOldIndexes.has(i)) continue;
            const candidate = oldContent[i];
            if (candidate?.attrs?.['block-id']) continue;
            if (!preferredType || candidate?.type === preferredType) return i;
        }
        for (let i = 0; i < oldContent.length; i++) {
            if (processedOldIndexes.has(i)) continue;
            const candidate = oldContent[i];
            if (candidate?.attrs?.['block-id']) continue;
            return i;
        }
        return -1;
    };

    // 1. Iterate through new content to find additions and modifications
    newContent.forEach((newNode, newIndex) => {
        const blockId = newNode?.attrs?.['block-id'];

        if (blockId) {
            const oldEntry = oldMap.get(blockId);

            if (oldEntry?.node) {
                processedOldIndexes.add(oldEntry.index);
                diffResult.push(compareNodePair(oldEntry.node, newNode));
            } else {
                diffResult.push({ ...newNode, diffStatus: 'added' });
            }
            return;
        }

        // Fallback path for nodes without block-id: compare by document order.
        const sameIndexOld = oldContent[newIndex];
        if (
            sameIndexOld &&
            !processedOldIndexes.has(newIndex) &&
            !sameIndexOld?.attrs?.['block-id']
        ) {
            processedOldIndexes.add(newIndex);
            diffResult.push(compareNodePair(sameIndexOld, newNode));
            return;
        }

        const fallbackOldIndex = findFallbackOldIndex(newIndex, newNode?.type);
        if (fallbackOldIndex >= 0) {
            processedOldIndexes.add(fallbackOldIndex);
            diffResult.push(compareNodePair(oldContent[fallbackOldIndex], newNode));
            return;
        }

        diffResult.push({ ...newNode, diffStatus: 'added' });
    });

    // 2. Identify and append removed blocks from the old content
    oldContent.forEach((oldNode, oldIndex) => {
        if (!processedOldIndexes.has(oldIndex)) {
            diffResult.push({ ...oldNode, diffStatus: 'removed' });
        }
    });

    return { type: 'doc', content: diffResult };
};
