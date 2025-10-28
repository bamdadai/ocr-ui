# app/utils/file_io.py
import base64
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
from fpdf import FPDF
from pdf2image import convert_from_bytes
import structlog

from app.core.config import settings
from .text_processing import make_farsi_text_for_pdf

logger = structlog.get_logger(__name__)

def pdf_to_images(pdf_bytes: bytes, dpi: int = 300) -> List[np.ndarray]:
    """
    Converts a PDF byte stream into a list of NumPy array images.
    
    Implements adaptive DPI fallback:
    - Tries initial DPI (300)
    - Falls back to 200 if initial fails (large/corrupted PDFs)
    - Falls back to 150 if 200 fails (very large PDFs)
    - Raises error if all attempts fail
    
    This prevents OOM crashes on large PDFs and handles corrupted files gracefully.
    """
    dpi_attempts = [dpi, 200, 150]  # Progressive DPI fallback
    last_error = None
    
    for attempt_dpi in dpi_attempts:
        try:
            logger.debug("pdf_to_images.attempt", dpi=attempt_dpi)
            pil_images = convert_from_bytes(pdf_bytes, dpi=attempt_dpi)
            
            if not pil_images:
                logger.warning("pdf_to_images.empty_result", dpi=attempt_dpi)
                continue
                
            images = [cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR) for pil_img in pil_images]
            
            if attempt_dpi != dpi:
                logger.info("pdf_to_images.fallback_succeeded", original_dpi=dpi, fallback_dpi=attempt_dpi, page_count=len(images))
            
            return images
            
        except Exception as e:
            last_error = e
            logger.warning("pdf_to_images.conversion_failed", dpi=attempt_dpi, error=str(e), error_type=type(e).__name__)
            continue
    
    # All DPI attempts failed
    logger.error("pdf_to_images.all_attempts_failed", attempted_dpis=dpi_attempts, final_error=str(last_error))
    raise ValueError(f"Failed to convert PDF. Tried DPIs {dpi_attempts}. Last error: {str(last_error)}")

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
