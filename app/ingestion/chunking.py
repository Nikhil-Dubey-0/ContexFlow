from app.utils.text_cleaning import clean_text

def chunk_text(text: str, chunk_size: int = 512, chunk_overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks.
    
    Args:
        text: the full text to split
        chunk_size: max characters per chunk
        chunk_overlap: how many characters overlap between consecutive chunks
    """
    chunks = []
    
    # step = how far we move forward each iteration
    # if chunk_size=512 and overlap=50, step=462
    step = chunk_size - chunk_overlap
    
    # slide a window across the text
    for start in range(0, len(text), step):
        end = start + chunk_size        # end of this chunk
        chunk = text[start:end]         # slice the text (Python handles out-of-bounds gracefully)
        
        # skip tiny leftover chunks at the end (less than 50 chars = probably garbage)
        if len(chunk.strip()) < 50:
            continue
            
        chunks.append(chunk.strip())
    
    return chunks


def chunk_documents(documents: list[dict], chunk_size: int = 512, chunk_overlap: int = 50) -> list[dict]:
    """Chunk a list of documents, preserving metadata.
    
    Takes the output of loader.load_directory() and returns
    a new list where each item is a single chunk with its metadata.
    """
    all_chunks = []
    
    for doc in documents:
        # clean the text first before chunking
        cleaned_text = clean_text(doc["text"])
        
        # split this document's text into chunks
        text_chunks = chunk_text(cleaned_text, chunk_size, chunk_overlap)
        
        # create a new dict for each chunk, carrying over the original metadata
        for i, chunk in enumerate(text_chunks):
            all_chunks.append({
                "text": chunk,
                "metadata": {
                    "source": doc["metadata"]["source"],      # original filename
                    "page": doc["metadata"]["page"],          # original page number
                    "chunk_index": i,                          # which chunk of this page (0, 1, 2...)
                }
            })
    
    return all_chunks
