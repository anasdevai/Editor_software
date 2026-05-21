export const escapeHtml = (text = '') =>
    String(text)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');

const cleanLine = (line = '') =>
    String(line)
        .replace(/\t/g, ' ')
        .replace(/[ ]{2,}/g, ' ')
        .trim();

const isMostlyUppercase = (line = '') => {
    const letters = (line.match(/[A-Za-zÄÖÜ]/g) || []).length;
    const uppers = (line.match(/[A-ZÄÖÜ]/g) || []).length;
    if (!letters) return false;
    return letters >= 4 && uppers / letters > 0.7;
};

const isSectionHeading = (line = '') =>
    /^§\s*\d+/i.test(line) ||
    /^section\s+\d+/i.test(line) ||
    /^article\s+\d+/i.test(line);

const isClauseHeading = (line = '') =>
    /^\d+(\.\d+)*[.)]?\s+/.test(line) ||
    /^\d+([.)]|\s*[,a-z]\))/i.test(line) ||
    /^[a-zA-Z][.)]\s+/.test(line);

const isBulletLine = (line = '') =>
    /^[-–•*]\s+/.test(line);

const isNumberedListLine = (line = '') =>
    /^\(?\d+\)?[.)]\s+/.test(line) ||
    /^[a-zA-Z][.)]\s+/.test(line);

const isStandaloneLabel = (line = '') =>
    /^[A-ZÄÖÜ][A-Za-zÄÖÜäöüß\s/&()-]{0,60}:$/.test(line);

const isKeyValueLine = (line = '') =>
    /^[A-ZÄÖÜ][A-Za-zÄÖÜäöüß\s/&()-]{0,40}:\s+\S+/.test(line);

const isTableLikeLine = (line = '') =>
    /\|/.test(line) || /\S+\s{2,}\S+/.test(line);

const isShortHeadingLike = (line = '') => {
    const cleaned = cleanLine(line);
    if (!cleaned) return false;
    if (cleaned.length > 80) return false;
    if (/[.!?]$/.test(cleaned)) return false;
    return isMostlyUppercase(cleaned);
};

const shouldMergeWithPrevious = (prev = '', current = '') => {
    if (!prev || !current) return false;

    if (isSectionHeading(prev) || isSectionHeading(current)) return false;
    if (isClauseHeading(prev) || isClauseHeading(current)) return false;
    if (isBulletLine(prev) || isBulletLine(current)) return false;
    if (isNumberedListLine(prev) || isNumberedListLine(current)) return false;
    if (isStandaloneLabel(prev) || isStandaloneLabel(current)) return false;
    if (isKeyValueLine(prev) || isKeyValueLine(current)) return false;
    if (isShortHeadingLike(prev) || isShortHeadingLike(current)) return false;

    if (/[,;:]$/.test(prev)) return true;
    if (/[-‐-‒–—]$/.test(prev)) return true;
    if (/^[a-zäöüß(]/.test(current)) return true;
    if (prev.length < 45 && !/[.!?]$/.test(prev)) return true;

    return false;
};

export const normalizeOCRLines = (text = '') => {
    const rawLines = String(text)
        .replace(/\r\n/g, '\n')
        .replace(/\r/g, '\n')
        .split('\n')
        .map(cleanLine);

    const merged = [];

    for (const line of rawLines) {
        if (!line) {
            if (merged[merged.length - 1] !== '') {
                merged.push('');
            }
            continue;
        }

        const prev = merged[merged.length - 1];

        if (typeof prev === 'string' && shouldMergeWithPrevious(prev, line)) {
            const cleanedPrev = prev.replace(/[-‐-‒–—]$/, '').trim();
            const joined = `${cleanedPrev}${prev.match(/[-‐-‒–—]$/) ? '' : ' '}${line}`
                .replace(/\s+/g, ' ')
                .trim();

            merged[merged.length - 1] = joined;
        } else {
            merged.push(line);
        }
    }

    return merged.filter((line, index, arr) => {
        if (line !== '') return true;
        return arr[index - 1] !== '';
    });
};

export const formatOCRText = (text = '') => {
    const lines = normalizeOCRLines(text);

    const htmlParts = [];
    let paragraphBuffer = [];
    let bulletListBuffer = [];
    let orderedListBuffer = [];
    let tableBuffer = [];
    let titleEmitted = false;

    const flushParagraph = () => {
        if (!paragraphBuffer.length) return;

        const paragraphText = paragraphBuffer
            .join(' ')
            .replace(/\s+/g, ' ')
            .trim();

        if (paragraphText) {
            htmlParts.push(`<p>${escapeHtml(paragraphText)}</p>`);
        }

        paragraphBuffer = [];
    };

    const flushBulletList = () => {
        if (!bulletListBuffer.length) return;

        htmlParts.push(
            `<ul>${bulletListBuffer
                .map((item) => `<li>${escapeHtml(item)}</li>`)
                .join('')}</ul>`
        );

        bulletListBuffer = [];
    };

    const flushOrderedList = () => {
        if (!orderedListBuffer.length) return;

        htmlParts.push(
            `<ol>${orderedListBuffer
                .map((item) => `<li>${escapeHtml(item)}</li>`)
                .join('')}</ol>`
        );

        orderedListBuffer = [];
    };

    const flushAll = () => {
        flushParagraph();
        flushBulletList();
        flushOrderedList();
        if (tableBuffer.length) {
            const candidateRows = tableBuffer
                .map((line) => line.split(/\s{2,}|\|/).map((cell) => cleanLine(cell)).filter(Boolean))
            const rows = candidateRows.filter((row) => row.length > 1);
            if (rows.length) {
                htmlParts.push(
                    `<table class="ocr-table"><tbody>${rows
                        .map((row) => `<tr>${row.map((cell) => `<td>${escapeHtml(cell)}</td>`).join('')}</tr>`)
                        .join('')}</tbody></table>`
                );
            } else {
                candidateRows.forEach((row) => {
                    htmlParts.push(`<p>${escapeHtml(row.join(' '))}</p>`);
                });
            }
            tableBuffer = [];
        }
    };

    for (const line of lines) {
        if (!line) {
            flushAll();
            continue;
        }

        if (!titleEmitted && isShortHeadingLike(line) && line.length <= 90) {
            flushAll();
            htmlParts.push(`<h1 class="ocr-title">${escapeHtml(line)}</h1>`);
            titleEmitted = true;
            continue;
        }

        if (isSectionHeading(line)) {
            flushAll();
            htmlParts.push(`<h2 class="ocr-section-heading">${escapeHtml(line)}</h2>`);
            continue;
        }

        if (isClauseHeading(line)) {
            flushAll();
            htmlParts.push(`<h3 class="ocr-clause-heading">${escapeHtml(line)}</h3>`);
            continue;
        }

        if (isShortHeadingLike(line)) {
            flushAll();
            htmlParts.push(`<h3 class="ocr-heading">${escapeHtml(line)}</h3>`);
            continue;
        }

        if (isBulletLine(line)) {
            flushParagraph();
            flushOrderedList();
            bulletListBuffer.push(line.replace(/^[-–•*]\s+/, '').trim());
            continue;
        }

        if (isNumberedListLine(line)) {
            flushParagraph();
            flushBulletList();
            orderedListBuffer.push(line.replace(/^\(?[A-Za-z0-9]+\)?[.)]\s+/, '').trim());
            continue;
        }

        if (isStandaloneLabel(line)) {
            flushAll();
            htmlParts.push(`<p class="ocr-label"><strong>${escapeHtml(line)}</strong></p>`);
            continue;
        }

        if (isKeyValueLine(line)) {
            flushAll();
            const [label, ...rest] = line.split(':');
            const value = rest.join(':').trim();

            htmlParts.push(
                `<p class="ocr-key-value"><strong>${escapeHtml(label)}:</strong> ${escapeHtml(value)}</p>`
            );
            continue;
        }

        if (isTableLikeLine(line) && !isBulletLine(line) && !isNumberedListLine(line)) {
            flushParagraph();
            flushBulletList();
            flushOrderedList();
            tableBuffer.push(line);
            continue;
        }

        paragraphBuffer.push(line);
    }

    flushAll();

    return htmlParts.join('') || '<p></p>';
};