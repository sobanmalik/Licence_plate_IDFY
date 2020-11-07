# Licence_plate_IDFY
Licence Plate OCR detection model

Used Google Tesseract with custom configuration for OCR detection. Didn't train any data as Tesseract models are well trained already and there wasn't enough data for custom training. Calculated individual accuracy of each licence plate rather than accuracy as a whole and saved it in 'inference.csv'.

Requirements: OpenCV, Tesseract-OCR, Pandas

Set your Tesseract folder location path in the beginning of script.py.
