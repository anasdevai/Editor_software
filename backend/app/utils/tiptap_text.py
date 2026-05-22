"""Plain-text extraction shared by routes and semantic jobs.

The editor mostly stores TipTap / ProseMirror JSON, but imported SOPs can also
arrive in the normalized extraction shape used by the PDF pipeline:
```
{"sections": [{"title": "...", "content": [{"text": "..."}, {"items": [...]}]}]}
```
Keep this extractor schema-tolerant so NLP, RAG indexing, and assistant context
all see the same readable SOP text.
"""


def extract_plain_text_from_tiptap(doc_json: dict | None) -> str:
    if not isinstance(doc_json, dict):
        return ""
    out: list[str] = []

    def append_text(value):
        if value is None:
            return
        text = str(value).strip()
        if text:
            out.append(text)

    def walk(node):
        if isinstance(node, str):
            append_text(node)
            return
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if not isinstance(node, dict):
            return

        if node.get("type") == "text":
            append_text(node.get("text"))
        elif isinstance(node.get("text"), str):
            append_text(node.get("text"))

        if isinstance(node.get("title"), str):
            append_text(node.get("title"))

        for item in node.get("items", []) or []:
            walk(item)

        for child in node.get("content", []) or []:
            walk(child)

        for section in node.get("sections", []) or []:
            walk(section)

        for row in node.get("rows", []) or []:
            walk(row)

        for cell in node.get("cells", []) or []:
            walk(cell)

        for child in node.get("children", []) or []:
            walk(child)

    walk(doc_json)
    return " ".join(out).strip()
