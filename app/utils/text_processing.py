# app/utils/text_processing.py
import re
import arabic_reshaper
from bidi.algorithm import get_display

def make_farsi_text_for_display(text: str) -> str:
    """Prepares Farsi text for display in libraries like Matplotlib."""
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)

def make_farsi_text_for_pdf(text: str) -> str:
    """Prepares Farsi text for libraries like FPDF that need reshaping."""
    return arabic_reshaper.reshape(text)

def fix_persian_parentheses(text: str) -> str:
    """
    Replaces Persian/Arabic parentheses with English parentheses in correct order.
    
    Handles various Persian/Arabic parenthesis patterns:
    - )text( → (text)  (reversed parentheses around RTL text)
    - )( → ()  (adjacent reversed parentheses)
    - Unicode Persian parentheses → English parentheses
    
    Args:
        text: Input text string
        
    Returns:
        Text with Persian parentheses fixed to English form
    """
    if not text:
        return text
    
    # Replace Unicode Persian parentheses with English parentheses
    persian_open_paren = '\uFD3E'  # ﴾
    persian_close_paren = '\uFD3F'  # ﴿
    text = text.replace(persian_open_paren, '(')
    text = text.replace(persian_close_paren, ')')
    
    # Pattern to match reversed parentheses around text: )text(
    # This handles cases like )س( → (س)
    # The pattern matches: ) + any characters (especially Persian/Arabic) + (
    def fix_reversed_parentheses(match):
        content = match.group(1)  # Text between parentheses
        return f'({content})'  # Swap to correct order
    
    # Match pattern: ) followed by any characters (with Persian/Arabic preferred) followed by (
    # This will match )س( and convert to (س)
    text = re.sub(r'\)([^)]*?)\(', fix_reversed_parentheses, text)
    
    # Also handle simple adjacent reversed parentheses: )( → ()
    text = text.replace(')(', '()')
    
    return text

