import PyPDF2
import sys
import os

pdf_path = "[붙임1] 2026년도 범부처재생의료기술개발사업 연구개발계획서(26.01.02 이상훈-수정-수정).pdf"
output_path = "pdf_content.txt"

try:
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at {pdf_path}")
        sys.exit(1)

    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        with open(output_path, 'w', encoding='utf-8') as text_file:
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                text_file.write(f"--- Page {page_num + 1} ---\n")
                text_file.write(text)
                text_file.write("\n\n")
    print(f"Successfully extracted text to {output_path}")
except Exception as e:
    print(f"Error: {e}")
