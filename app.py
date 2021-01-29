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
        ext = os.path.splitext(filename)[1]
        if not ext:
            filename = filename + '.png'
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


def format_result(results, cell_gap=30):
    """
    """
    transform_data = []
    for box, text in results:
        x, y, w, h = box
        left = x
        bottom = y + h
        top = y
        transform_data.append((left, top, bottom, x, y, w, h, text))
    
    # Sort by y-axis
    row_data = sorted(transform_data, key=lambda x: x[1])

    # Gather texts are in the same line
    table_data = []
    row = []
    prev_bottom = None
    max_x = 0
    for cell in row_data:
        x_index = cell[3] // cell_gap
        if x_index > max_x:
            max_x = x_index
        
        if prev_bottom is None:
            row = [cell[3:]]
            prev_bottom = cell[2]
        else:
            if cell[1] > prev_bottom:
                # New row
                table_data.append(sorted(row, key=lambda x: x[0]))
                row = [cell[3:]]
                prev_bottom = cell[2]
            else:
                # Same row
                row.append(cell[3:])
    
    grid = [[' ' for _ in range(max_x+1)] for _ in range(len(table_data)+1)]
    for i, row in enumerate(table_data):
        for cell in row:
            x_index = cell[0] // cell_gap
            text = cell[-1]
            if 0 <= x_index <= max_x:
                if grid[i][x_index] != ' ':
                    grid[i][x_index] += ' ' + text
                else:
                    grid[i][x_index] = text
    
    result_str = ''
    for row in grid:
        result_str += ''.join([text for text in row]) + '\n'
    return result_str


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
    response_type = config.get('response_type', 'text')
    message = ''
    if response_type == 'text':
        message = ''
    else:
        ret = {'status': False, 'data': None, 'message': None}

    try:
        contents = await file.read()
    except:
        message = 'Read image failed'
        if response_type == 'text':
            return message
        else:
            ret['message'] = message
            return ret
    
    try:
        arr = np.fromstring(contents, np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except:
        message = 'Decode image failed'
        if response_type == 'text':
            return message
        else:
            ret['message'] = message
            return ret

    # Recognize image
    filename = file.filename
    results = read_text(image, filename)
    try:
        if results:
            if response_type == 'text':
                message = format_result(results, config.get('cell_gap', 50))
                return message
            else:
                ret['data'] = []
                ret['message'] = 'succesed'
                ret['status'] = True
                for box, text in results:
                    x, y, w, h = box
                    ret['data'].append({
                        'x': x,
                        'y': y,
                        'w': w,
                        'h': h,
                        'text': text
                    })
                return ret
    except:
        message = 'Text extraction failed'
        if response_type == 'text':
            return message
        else:
            ret['message'] = message
            return ret
    
    return ret