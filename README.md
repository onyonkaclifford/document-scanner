![Tests workflow](https://github.com/onyonkaclifford/document-scanner/actions/workflows/tests.yml/badge.svg?branch=main)
![Lint workflow](https://github.com/onyonkaclifford/document-scanner/actions/workflows/lint.yml/badge.svg?branch=main)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/onyonkaclifford/document-scanner/blob/main/LICENSE)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Code style: flake8](https://img.shields.io/badge/code%20style-flake8-orange.svg)](https://gitlab.com/pycqa/flake8)

# Document scanner
Extract text from photos

Tesseract for OCR is required. To install on Ubuntu: `sudo apt-get install tesseract-ocr`

Example usage:
```python
import cv2
from document_scanner import DocumentScanner

img_bgr, img_rgb = DocumentScanner.get_image("assets/img.png")
img_processed = DocumentScanner.get_processed_image(img_bgr)
text = DocumentScanner.get_text(img_processed)

print(text)

cv2.imshow("Image BGR", img_bgr)
cv2.imshow("Image RGB", img_rgb)
cv2.imshow("Processed img", img_processed)
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
```
