import os
import fitz  # PyMuPDF
from docx import Document as DocxDocument  # python-docx for Word files


def load_pdf(file_path: str) -> list[dict]:
    """Load a single PDF and extract text per page."""
    documents = []

    try:
        doc = fitz.open(file_path) # opens the PDF
    except Exception as e:
        print(f"⚠️ Could not open {file_path}: {e}")
        return []

    file_name = os.path.basename(file_path)  # gets the name of the file not path

    for page_num, page in enumerate(doc):  # looping for each page in document
        text = page.get_text().strip()  # gets the text from the page and strips the whitespace

        # Skip empty pages
        if not text:
            continue

        documents.append({  # structure of doc stored in list
            "text": text,
            "metadata": {
                "source": file_name,
                "page": page_num + 1  # 1-indexed for humans
            }
        })

    doc.close()
    print(f"  📄 {file_name}: {len(documents)} pages extracted")
    return documents


def load_docx(file_path: str) -> list[dict]:
    """Load a Word document and extract text."""
    documents = []
    file_name = os.path.basename(file_path)

    try:
        doc = DocxDocument(file_path)
    except Exception as e:
        print(f"⚠️ Could not open {file_path}: {e}")
        return []

    # combine all paragraphs into full text
    # Word docs don't have a "page" concept in the API
    full_text = "\n".join([para.text for para in doc.paragraphs])

    if full_text.strip():
        documents.append({
            "text": full_text,
            "metadata": {
                "source": file_name,
                "page": 1  # Word doesn't expose page numbers
            }
        })

    print(f"  📄 {file_name}: {len(documents)} section(s) extracted")
    return documents


def load_directory(dir_path: str) -> list[dict]:
    """Load all supported files (PDF, DOCX) from a directory."""
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    all_documents = []
    supported_count = 0
    skipped = []

    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        lower = file_name.lower()

        if lower.endswith(".pdf"):
            all_documents.extend(load_pdf(file_path))
            supported_count += 1
        elif lower.endswith(".docx"):
            all_documents.extend(load_docx(file_path))
            supported_count += 1
        else:
            skipped.append(file_name)

    if skipped:
        print(f"⏭️ Skipped unsupported files: {', '.join(skipped)}")

    if not all_documents:
        print(f"⚠️ No supported files found in {dir_path}")
    else:
        print(f"✅ Total: {len(all_documents)} pages/sections from {supported_count} file(s)")

    return all_documents