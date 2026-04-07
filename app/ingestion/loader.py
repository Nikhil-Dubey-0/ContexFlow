import os
import fitz  # PyMuPDF


def load_pdf(file_path: str) -> list[dict]:
    """Load a single PDF and extract text per page."""
    documents = []

    try:
        doc = fitz.open(file_path)
    except Exception as e:
        print(f"⚠️ Could not open {file_path}: {e}")
        return []

    file_name = os.path.basename(file_path)

    for page_num, page in enumerate(doc):
        text = page.get_text().strip()

        # Skip empty pages
        if not text:
            continue

        documents.append({
            "text": text,
            "metadata": {
                "source": file_name,
                "page": page_num + 1  # 1-indexed for humans
            }
        })

    doc.close()
    print(f"  📄 {file_name}: {len(documents)} pages extracted")
    return documents


def load_directory(dir_path: str) -> list[dict]:
    """Load all PDFs from a directory."""
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    pdf_files = [f for f in os.listdir(dir_path) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"⚠️ No PDF files found in {dir_path}")
        return []

    print(f"📂 Found {len(pdf_files)} PDF(s) in {dir_path}")

    all_documents = []
    for pdf_file in pdf_files:
        file_path = os.path.join(dir_path, pdf_file)
        docs = load_pdf(file_path)
        all_documents.extend(docs)

    print(f"✅ Total: {len(all_documents)} pages from {len(pdf_files)} file(s)")
    return all_documents