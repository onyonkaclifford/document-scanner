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
