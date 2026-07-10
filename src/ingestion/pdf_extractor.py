import os
# pyrefly: ignore [missing-import]
import fitz
import re
import sys

# Ensure Python can find the src module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.config import RAW_PDF_DIR, PROCESSED_TEXT_DIR

def clean_text(text: str) -> str:
    """
    Cleans extracted PDF text by removing excessive newlines and spaces,
    making it easier for the embedding model to parse.
    """
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n+', '\n', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def extract_text_from_pdfs():
    """
    Iterates over all PDFs in the raw_pdfs directory, extracts their text,
    cleans it, and saves it to the processed_text directory.
    """
    print(f"Scanning directory: {RAW_PDF_DIR}")
    
    pdf_files = [f for f in os.listdir(RAW_PDF_DIR) if f.endswith('.pdf')]
    if not pdf_files:
        print("❌ No PDF files found in the raw_pdfs directory.")
        return

    for pdf_file in pdf_files:
        pdf_path = os.path.join(RAW_PDF_DIR, pdf_file)
        print(f"📄 Processing: {pdf_file}...")
        
        try:
            # Open PDF using PyMuPDF
            doc = fitz.open(pdf_path)
            full_text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                full_text += page.get_text("text") + "\n"
            
            cleaned_text = clean_text(full_text)
            
            # Save to processed_text
            txt_filename = pdf_file.replace('.pdf', '.txt')
            txt_path = os.path.join(PROCESSED_TEXT_DIR, txt_filename)
            
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
                
            print(f"✅ Saved extracted text to: {txt_filename}")
            
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")

if __name__ == "__main__":
    extract_text_from_pdfs()