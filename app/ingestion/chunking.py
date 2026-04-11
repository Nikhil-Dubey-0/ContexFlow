from app.utils.text_cleaning import clean_text

def chunk_text(text: str, chunk_size: int = 512, chunk_overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks using recursive character splitting.
    
    Tries to split at natural boundaries: paragraphs → lines → sentences → words
    """
    # separators in priority order — try the "biggest" breaks first
    separators = ["\n\n", "\n", ". ", " "]

    chunks = []
    # _recursive_split does the heavy lifting
    _recursive_split(text, chunk_size, chunk_overlap, separators, chunks)
    return chunks


def _recursive_split(text: str, chunk_size: int, chunk_overlap: int, 
                     separators: list[str], chunks: list[str]):
    """Recursively split text using the best available separator."""
    
    # base case: text fits in one chunk — just add it
    if len(text.strip()) <= chunk_size:
        if len(text.strip()) >= 30:          # skip tiny fragments
            chunks.append(text.strip())
        return

    # find the best separator that exists in this text
    separator = ""
    for sep in separators:
        if sep in text:
            separator = sep
            break

    # if no separator found, fall back to fixed-size splitting
    if not separator:
        step = chunk_size - chunk_overlap
        for start in range(0, len(text), step):
            chunk = text[start:start + chunk_size].strip()
            if len(chunk) >= 30:
                chunks.append(chunk)
        return

    # split text by the chosen separator
    parts = text.split(separator)

    # now combine parts back into chunks that fit within chunk_size
    current_chunk = ""
    for part in parts:
        # if adding this part would exceed chunk_size, save current chunk and start new one
        if len(current_chunk) + len(separator) + len(part) > chunk_size and current_chunk:
            if len(current_chunk.strip()) >= 30:
                chunks.append(current_chunk.strip())
            
            # start fresh — no overlap at natural boundaries
            # overlap only applies in the fixed-size fallback (line 36-41)
            current_chunk = part
        else:
            # keep building the current chunk
            if current_chunk:
                current_chunk += separator + part
            else:
                current_chunk = part

    # don't forget the last chunk
    if len(current_chunk.strip()) >= 30:
        chunks.append(current_chunk.strip())



def _extract_page_context(text: str) -> str:
    """Extract a short context prefix from the beginning of a page.
    
    Grabs the first 1-2 meaningful lines — usually a heading, course name,
    or section title. This gets prepended to every chunk from this page
    so chunks don't lose their parent context.
    
    Example:
        Page text starts with: "Deep Learning (BIT-326)\nCourse Category: PC\n..."
        Returns: "Deep Learning (BIT-326)"
    """
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    
    if not lines:
        return ""
    
    # take first 1-2 non-trivial lines as context (skip page numbers etc.)
    context_lines = []
    for line in lines[:3]:  # look at first 3 lines
        # skip lines that are just numbers (page numbers)
        if line.isdigit():
            continue
        # skip very short lines (likely fragments)
        if len(line) < 5:
            continue
        context_lines.append(line)
        if len(" | ".join(context_lines)) > 80:  # keep context short
            break
    
    return " | ".join(context_lines) if context_lines else ""


def chunk_documents(documents: list[dict], chunk_size: int = 512, chunk_overlap: int = 50) -> list[dict]:
    """Chunk a list of documents, preserving metadata.
    
    Takes the output of loader.load_directory() and returns
    a new list where each item is a single chunk with its metadata.
    
    Each chunk is enriched with context from the page header so it
    doesn't lose connection to its parent section/course.
    """
    all_chunks = []
    
    for doc in documents:
        # clean the text first before chunking
        cleaned_text = clean_text(doc["text"])
        
        # extract context from the page (first heading/course name)
        page_context = _extract_page_context(cleaned_text)
        
        # split this document's text into chunks
        text_chunks = chunk_text(cleaned_text, chunk_size, chunk_overlap)
        
        # create a new dict for each chunk, carrying over the original metadata
        for i, chunk in enumerate(text_chunks):
            # prepend context if it's not already in the chunk
            if page_context and page_context not in chunk:
                enriched_text = f"[{page_context}]\n{chunk}"
            else:
                enriched_text = chunk
            
            all_chunks.append({
                "text": enriched_text,
                "metadata": {
                    "source": doc["metadata"]["source"],      # original filename
                    "page": doc["metadata"]["page"],          # original page number
                    "chunk_index": i,                          # which chunk of this page
                }
            })
    
    return all_chunks
