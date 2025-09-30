# app/utils/file_io.py
import base64
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
from fpdf import FPDF
from pdf2image import convert_from_bytes

from app.core.config import settings
from .text_processing import make_farsi_text_for_pdf

def pdf_to_images(pdf_bytes: bytes, dpi: int = 300) -> List[np.ndarray]:
    """Converts a PDF byte stream into a list of NumPy array images."""
    pil_images = convert_from_bytes(pdf_bytes, dpi=dpi)
    images = [cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR) for pil_img in pil_images]
    return images

def create_searchable_pdf(texts: List[str], output_path: str):
    """Creates a searchable PDF file from a list of text strings."""
    pdf = FPDF()
    pdf.add_font('DejaVu', '', settings.SEARCHABLE_PDF_FONT_PATH, uni=True)
    pdf.set_font('DejaVu', '', 11)
    
    for text in texts:
        pdf.add_page()
        bidi_text = make_farsi_text_for_pdf(text)
        pdf.multi_cell(0, 10, bidi_text, align='R')
    pdf.output(output_path)

def save_as_text(text_data: List[str], output_path: str) -> str:
    """Saves extracted text content to a .txt file."""
    txt_output_path = Path(output_path).with_suffix('.txt')
    with open(txt_output_path, 'w', encoding='utf-8') as f:
        for i, page_text in enumerate(text_data):
            if i > 0: f.write('\n\n' + '='*20 + '\n\n')
            f.write(page_text.strip())
    return str(txt_output_path)

def read_text_file(file_path: str) -> str:
    """Reads the content of a text file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def get_file_as_base64(file_path: str) -> str:
    """Reads a file and returns its base64 encoded content."""
    with open(file_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')
