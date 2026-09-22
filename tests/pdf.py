import os
from pathlib import Path

import pymupdf


UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
FILE_PATH = f"{UPLOAD_DIR}/test.pdf"
print(FILE_PATH)
doc = pymupdf.open(FILE_PATH)
text = ""
for page in doc: # iterate the document pages
    text = page.get_text() # get plain text (is in UTF-8)

if len(doc) > 0:
    print(text)
doc.close()