# invoice_text_analytics

## Tesseract OCR
For Tesseract OCR to work on windows: 
- Download tesseract from https://digi.bib.uni-mannheim.de/tesseract/
- Don't forget to also install the additional languages you need
- Add Tesseract OCR to your PATH
- Open command line and write tesseract -v to check if it was installed correctly

## main.py
This copies the invoices into a ./data folder, scans them with ocr and combines it with some structured data in a csv file

## api.py
Starts the Flask application on localhost:3000