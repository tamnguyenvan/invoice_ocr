"""
"""
import os
import io
import json

import yaml
import cv2
import numpy as np
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import PlainTextResponse
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

# Create storage directory
storage_dir = config.get('storage', 'images')
if not os.path.exists(storage_dir):
    os.makedirs(storage_dir, exist_ok=True)


def read_text(img, filename, save_image=True):
    """
    """
    # Detect invoice contour
    invoice_roi, (x_offset, y_offset) = detector.detect(img)
    if save_image:
        out_path = os.path.join(storage_dir, filename)
        cv2.imwrite(out_path, invoice_roi)

    results = []
    if invoice_roi is not None:
        # Recognize
        raw_results = recognizer.recognize(invoice_roi)
        for box, text in raw_results:
            x, y, w, h = box
            x += x_offset
            y += y_offset
            results.append(((x, y, w, h), text))
        return results


@app.get("/")
def index():
    """Test server status"""
    return 'It works'


@app.post('/read-text')
async def process(file: UploadFile = File(...), response_class=PlainTextResponse):
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
    try:
        contents = await file.read()
    except:
        return 'Read image failed'
    
    try:
        arr = np.fromstring(contents, np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except:
        return 'Decode image failed'

    # Recognize image
    filename = file.filename
    results = read_text(image, filename)
    try:
        data = ''
        if results:
            num_texts = 0
            for box, text in results:
                num_texts += 1
                x, y, w, h = box
                box_data = '{} {} {} {} {}'.format(x, y, w, h, text)
                data += box_data + '\n'
    except:
        return 'Text extraction failed'
    
    return data