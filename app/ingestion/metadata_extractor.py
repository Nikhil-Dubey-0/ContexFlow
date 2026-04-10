'''
metadata extraction already happens in loader.py
"metadata": {
    "source": file_name,   # ← extracted here
    "page": page_num + 1   # ← extracted here
}


That file is in the original folder structure as a placeholder for advanced metadata like:

Document title, author, creation date (from PDF properties)
Section headers, table of contents detection
Language detection
'''