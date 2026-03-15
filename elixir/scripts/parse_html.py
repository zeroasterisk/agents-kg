#!/usr/bin/env python3
import sys
import re

def _html_to_text(html: str) -> str:
    """Convert HTML to readable text using readability + basic cleanup."""
    try:
        from readability import Document
        doc = Document(html)
        summary = doc.summary()
        title = doc.title()
    except Exception:
        summary = html
        title = ""

    # Strip remaining HTML tags
    text = re.sub(r'<[^>]+>', '\n', summary)
    # Collapse whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()

    if title and not text.startswith(title):
        text = f"# {title}\n\n{text}"
    return text

if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            html_in = f.read()
    else:
        html_in = sys.stdin.read()
    print(_html_to_text(html_in))
