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

def join_spaced_numbers(text: str) -> str:
    """Removes spaces between numbers and joins them together (supports both Arabic and Persian digits)."""
    # Pattern to match sequences of digits separated by spaces
    # This handles cases like "1 2 3" -> "123" or "۱ ۲ ۳" -> "۱۲۳"
    def replace_number_sequence(match):
        # Extract the matched sequence and remove spaces between digits
        sequence = match.group(0)
        # Replace spaces between digits with nothing (both Arabic and Persian)
        return re.sub(r'(?<=[۰-۹0-9])\s+(?=[۰-۹0-9])', '', sequence)

    # Use regex to find sequences that contain digits and spaces
    # This pattern finds sequences that start and end with digits and contain spaces
    # Supports both Arabic (0-9) and Persian (۰-۹) digits
    number_sequence_pattern = re.compile(r'[۰-۹0-9]+(?:\s+[۰-۹0-9]+)+')
    return number_sequence_pattern.sub(replace_number_sequence, text)

def fix_dash_positioning(text: str) -> str:
    """Moves dashes from before numbers to after numbers (supports both Arabic and Persian digits)."""
    # Pattern to match dash followed by optional spaces and then digits
    # This handles cases like "-123" -> "123-" or "- ۱ ۲ ۳" -> "۱۲۳-"
    def move_dash_after_number(match):
        dash = match.group(1)  # The dash
        spaces = match.group(2)  # Optional spaces after dash
        number_part = match.group(3)  # The number part
        
        # Remove spaces from the number part and put dash after
        clean_number = re.sub(r'\s+', '', number_part)
        return clean_number + dash
    
    # Pattern: dash + optional spaces + digits (Arabic 0-9 or Persian ۰-۹) and any following digits/spaces
    # Persian digits: ۰۱۲۳۴۵۶۷۸۹
    dash_before_number_pattern = re.compile(r'(-)(\s*)([۰-۹0-9]+(?:\s*[۰-۹0-9]+)*)')
    return dash_before_number_pattern.sub(move_dash_after_number, text)

def fix_dash_comma_spacing(text: str) -> str:
    """Removes spaces around dashes and Persian commas, and handles spaced patterns like - - - or ، ، ،."""
    # Pattern to match spaces around dashes and Persian commas
    # This handles cases like "word - word" -> "word-word" or "word ، word" -> "word،word"
    
    # Handle spaced dash patterns: "- - -" or "- -" -> "-"
    text = re.sub(r'(\s*-\s*)+', '-', text)
    
    # Handle spaced Persian comma patterns: "، ، ،" or "، ،" -> "،"
    text = re.sub(r'(\s*،\s*)+', '،', text)
    
    # Remove multiple consecutive dashes (without spaces): "--" or "---" -> "-"
    text = re.sub(r'-+', '-', text)
    
    # Remove multiple consecutive Persian commas (without spaces): "،،" or "،،،" -> "،"
    text = re.sub(r'،+', '،', text)
    
    return text

def fix_mixed_text_order(text: str) -> str:
    """Corrects display order for strings with mixed RTL and LTR text."""
    persian_pattern = re.compile(r'[\u0600-\u06FF]+')
    tokens = re.findall(r'\S+|\s+', text)
    segments, temp_segment, is_persian = [], [], None
    for token in tokens:
        current_is_persian = bool(persian_pattern.search(token))
        if is_persian is None: is_persian = current_is_persian
        if current_is_persian == is_persian:
            temp_segment.append(token)
        else:
            segments.append((is_persian, temp_segment))
            temp_segment = [token]
            is_persian = current_is_persian
    if temp_segment: segments.append((is_persian, temp_segment))

    fixed_segments = []
    for is_persian, segment in segments:
        if not is_persian: segment.reverse()
        fixed_segments.append(''.join(segment))
    return ''.join(fixed_segments)