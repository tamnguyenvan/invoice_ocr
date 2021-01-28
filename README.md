# Invoice OCR
Invoice OCR project

# Installation
Install all dependencies in `requirements.txt`:
```
pip install -r requirements.txt
```

# Run server
Use this command to run server: `uvicorn app:app --port 8080`.

# Quick start
Please go to `http://localhost:80000/docs` to read the details. Sending an image:
```
curl -X POST "http://localhost:8000/read-text" \
    -H "accept: application/json" \
    -H "Content-Type: multipart/form-data" \
    -F "file=@path/to/image" \
    awk '{gsub(/\\n/,"\n")}1'
```
*Please replace the path with your own image path*