# app/utils/text_processing.py
import arabic_reshaper
from bidi.algorithm import get_display

def make_farsi_text_for_display(text: str) -> str:
    """Prepares Farsi text for display in libraries like Matplotlib."""
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)

def make_farsi_text_for_pdf(text: str) -> str:
    """Prepares Farsi text for libraries like FPDF that need reshaping."""
    return arabic_reshaper.reshape(text)

