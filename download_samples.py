import urllib.request
import os
from pathlib import Path

# Create directory
Path("data/pdfs").mkdir(parents=True, exist_ok=True)

# Simple PDF URLs that should work
samples = [
    ("https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf", "dummy.pdf"),
    ("http://www.africau.edu/images/default/sample.pdf", "sample.pdf"),
]

for url, filename in samples:
    filepath = os.path.join("data/pdfs", filename)
    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"[OK] Downloaded {filename}")
    except Exception as e:
        print(f"[FAIL] Could not download {filename}: {e}")

print("\nYou can also add your own PDF files to the data/pdfs/ directory")