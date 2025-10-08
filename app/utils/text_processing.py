# app/utils/text_processing.py
import re
import json
import os
from pathlib import Path
import arabic_reshaper
from bidi.algorithm import get_display

PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ENGLISH_DIGITS = "0123456789"
_EN_TO_FA_TRANSLATION = str.maketrans(ENGLISH_DIGITS, PERSIAN_DIGITS)

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


def fix_period_positioning(text: str) -> str:
    """Moves periods from before Persian numbers to after Persian numbers."""
    # Pattern to match period followed by optional spaces and then Persian digits
    # This handles cases like ".۱۲۳" -> "۱۲۳." or ". ۱ ۲ ۳" -> "۱۲۳."
    def move_period_after_number(match):
        period = match.group(1)  # The period
        spaces = match.group(2)  # Optional spaces after period
        number_part = match.group(3)  # The number part
        
        # Remove spaces from the number part and put period after
        clean_number = re.sub(r'\s+', '', number_part)
        return clean_number + period
    
    # Pattern: period + optional spaces + Persian digits (۰-۹) and any following digits/spaces
    # Persian digits: ۰۱۲۳۴۵۶۷۸۹
    period_before_number_pattern = re.compile(r'(\.)(\s*)([۰-۹]+(?:\s*[۰-۹]+)*)')
    return period_before_number_pattern.sub(move_period_after_number, text)

def fix_dash_comma_spacing(text: str) -> str:
    """Removes spaces around dashes and Persian commas, and handles spaced patterns like - - - or ، ، ،."""
    # Pattern to match spaces around dashes and Persian commas
    # This handles cases like "word - word" -> "word-word" or "word ، word" -> "word،word"
    
    text = re.sub(r'(-+ -+)+', '-', text)
    
    # Handle spaced Persian comma patterns: "، ، ،" or "، ،" -> "،"
    text = re.sub(r'(،+ ،+)+', '،', text)
    
    
    
    return text

def convert_mixed_digit_sequences(text: str) -> str:
    """Converts English digits to Persian when they appear alongside Persian digits."""
    digit_pattern = re.compile(r'[0-9\u06F0-\u06F9]+')

    def replace_mixed_digits(match: re.Match) -> str:
        sequence = match.group(0)
        has_persian = any('\u06F0' <= ch <= '\u06F9' for ch in sequence)
        has_english = any('0' <= ch <= '9' for ch in sequence)
        if has_persian and has_english:
            return sequence.translate(_EN_TO_FA_TRANSLATION)
        return sequence

    return digit_pattern.sub(replace_mixed_digits, text)

def load_replacements_from_json(json_file_path: str) -> dict:
    """Loads replacement patterns from a JSON file.
    
    Args:
        json_file_path: Path to the JSON file containing replacement patterns
        
    Returns:
        Dictionary of replacement patterns
        
    Raises:
        FileNotFoundError: If the JSON file doesn't exist
        json.JSONDecodeError: If the JSON file is malformed
    """
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"Replacement file not found: {json_file_path}")
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def apply_custom_replacements(text: str, replacement_dict: dict = None, json_file_path: str = None) -> str:
    """Applies custom pattern replacements defined in a dictionary or JSON file.
    
    Args:
        text: The input text to process
        replacement_dict: Dictionary where keys are patterns to find and values are replacements
                         Keys can be either strings (exact matches) or regex patterns
        json_file_path: Path to JSON file containing replacement patterns (alternative to replacement_dict)
    
    Returns:
        Text with custom replacements applied
    """
    # Load replacements from JSON file if provided
    if json_file_path:
        replacement_dict = load_replacements_from_json(json_file_path)
    
    if not replacement_dict:
        return text
    
    # Flatten nested dictionaries if they exist
    flat_replacements = {}
    for key, value in replacement_dict.items():
        if isinstance(value, dict):
            # If it's a nested dictionary, flatten it
            flat_replacements.update(value)
        else:
            # If it's a direct pattern-replacement pair
            flat_replacements[key] = value
    
    for pattern, replacement in flat_replacements.items():
        # If the pattern is a string, treat it as a literal replacement
        if isinstance(pattern, str):
            text = text.replace(pattern, replacement)
        else:
            # If it's a compiled regex or string that should be treated as regex
            text = re.sub(pattern, replacement, text)
    
    return text

def remove_parentheses(text: str) -> str:
    """Removes parentheses ( and ) from text and replaces them with spaces."""
    # Replace both opening and closing parentheses with spaces
    text = text.replace('(', ' ')
    text = text.replace(')', ' ')
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
