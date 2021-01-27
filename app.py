"""
"""
import io

import yaml
import cv2
import numpy as np
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from starlette.requests import Request
from PIL import Image

from backend import InvoiceDetector, InvoiceOCR

app = FastAPI()


with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

detector = InvoiceDetector()
method = config.get('method')
if method not in ('tesseract', 'easyocr'):
    method = 'tesseract'
recognizer = InvoiceOCR(method=method)


def read_text(img):
    """
    """
    # Detect invoice contour
    invoice_roi = detector.detect(img)
    if invoice_roi is not None:
        # Recognize
        results = recognizer.recognize(invoice_roi)
        return results


@app.get("/")
def index():
    """Test server status"""
    return 'It works'


@app.post('/read-text')
async def process(file: UploadFile = File(...)):
    """Process image and returns extracted data.

    Upload command:

    ```bash
    curl -X POST "http://localhost:8000/read-text" \\
        -H "accept: application/json" \\
        -H "Content-Type: multipart/form-data" \\
        -F "file=@path/to/image"
    ```
    
    Respone format:

    ```
    {
        "success": True/False,
        "results": [
            {"x": 1, "y": 1, "w": 50, "h": 50, "text": "TEXT"},
            ...
        ]
    }
    ```
    """
    data = {'sucess': False, 'results': []}

    contents = await file.read()
    arr = np.fromstring(contents, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # Recognize image
    results = read_text(image)
    for box, text in results:
        x, y, w, h = box
        box_data = {
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'text': text
        }
        data['results'].append(box_data)
    return data