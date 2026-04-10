import re


def clean_text(text: str) -> str:
    """Clean raw PDF-extracted text."""

    # Replace multiple newlines with a single newline
    # PDFs often have \n\n\n\n between sections — collapse to one
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Replace multiple spaces/tabs with a single space
    # "hello     world" → "hello world"
    text = re.sub(r'[ \t]+', ' ', text)

    # Fix broken words: when a word is split across lines
    # "opti-\nmization" → "optimization"
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)

    # Remove leading/trailing whitespace from each line
    # "  hello  \n  world  " → "hello\nworld"
    lines = text.split('\n')
    lines = [line.strip() for line in lines]
    text = '\n'.join(lines)

    # Remove completely empty lines that are left over
    # After stripping, some lines become "" — remove consecutive empties
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Final strip
    text = text.strip()

    return text
