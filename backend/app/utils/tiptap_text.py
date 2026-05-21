"""TipTap / ProseMirror plain-text extraction shared by routes and semantic jobs."""


def extract_plain_text_from_tiptap(doc_json: dict | None) -> str:
    if not isinstance(doc_json, dict):
        return ""
    out: list[str] = []

    def walk(node: dict):
        if not isinstance(node, dict):
            return
        if node.get("type") == "text" and node.get("text"):
            out.append(str(node.get("text")))
        for child in node.get("content", []) or []:
            walk(child)

    walk(doc_json)
    return " ".join(out).strip()
