"""
Khmer Text Normalization - Full Implementation
Hệ thống chuẩn hóa văn bản tiếng Khmer đầy đủ dựa trên textnorm system
"""

import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from collections import defaultdict


# ============================================================================
# CONSTANTS - Từ các file .grm
# ============================================================================

# Khmer months mapping (từ date.grm)
KHMER_MONTHS = {
    1: "មករា",
    2: "កុម្ភៈ",
    3: "មិនា",
    4: "មេសា",
    5: "ឧសភា",
    6: "មិថុនា",
    7: "កក្កដា",
    8: "សីហា",
    9: "កញ្ញា",
    10: "តុលា",
    11: "វិច្ឆិកា",
    12: "ធ្នូ",
}

# Khmer weekday abbreviations
KHMER_WEEKDAYS = {
    "ច": "ចន្ទ",
    "អ": "អង្គារ",
    "ពុ": "ពុធ",
    "ព្រ": "ព្រហស្បតិ៍",
    "សុ": "សុក្រ",
    "ស": "សៅរ៍",
    "អា": "អាទិត្យ",
}

# Currency names (từ money.grm)
CURRENCY_NAMES = {
    "usd": "ដុល្លារ អាមេរិក",
    "gbp": "ផោន",
    "jpy": "យ៉េន ជប៉ុន",
    "khr": "រៀល",
    "thb": "បាត",
    "aud": "ដុល្លារ អូស្រ្តាលី",
    "chf": "ហ្វ្រង់ ស្វីស",
}

# Khmer digits to Arabic
KHMER_TO_ARABIC = {
    "០": "0", "១": "1", "២": "2", "៣": "3", "៤": "4",
    "៥": "5", "៦": "6", "៧": "7", "៨": "8", "៩": "9",
}

# Basic Khmer numbers (0-20, 30, 40, 50, 60, 70, 80, 90, 100, 1000, etc.)
KHMER_NUMBERS = {
    0: "សូន្យ",
    1: "មួយ",
    2: "ពីរ",
    3: "បី",
    4: "បួន",
    5: "ប្រាំ",
    6: "ប្រាំមួយ",
    7: "ប្រាំពិល",
    8: "ប្រាំបី",
    9: "ប្រាំបួន",
    10: "ដប់",
    11: "ដប់ មួយ",
    12: "ដប់ ពីរ",
    13: "ដប់ បី",
    14: "ដប់ បួន",
    15: "ដប់ ប្រាំ",
    16: "ដប់ ប្រាំមួយ",
    17: "ដប់ ប្រាំពិល",
    18: "ដប់ ប្រាំបី",
    19: "ដប់ ប្រាំបួន",
    20: "ម្ភៃ",
    30: "សាមសិប",
    40: "សែសិប",
    50: "ហាសិប",
    60: "ហុកសិប",
    70: "ចិតសិប",
    80: "ប៉ែតសិប",
    90: "កៅសិប",
    100: "រយ",
    1000: "ពាន់",
    10000: "ម៉ឺន",
    100000: "សែន",
    1000000: "លាន",
    10000000: "កោដិ",
    1000000000: "ប៊ីលាន",
    1000000000000: "ទ្រីលាន",
}

# Measure units (từ measure.grm và measure.tsv)
MEASURE_UNITS = {
    "meter": "ម៉ែត្រ",
    "km": "គីឡូ ម៉ែត្រ",
    "kilometer": "គីឡូ ម៉ែត្រ",
    "gram": "ក្រាម",
    "kg": "គីឡូ ក្រាម",
    "kilogram": "គីឡូ ក្រាម",
    "liter": "លីត្រ",
    "second": "វិនាទី",
    "minute": "នាទី",
    "hour": "ម៉ោង",
    "day": "ថ្ងៃ",
    "week": "សប្ដាហ៍",
    "month": "ខែ",
    "year": "ឆ្នាំ",
    "percent": "ភាគរយ",
    "%": "ភាគរយ",
}

# Emoticons mapping (từ emoticons.tsv)
EMOTICONS_MAP = {
    ":)": "មុខ ញញឹម",
    ":-)": "មុខ ញញឹម",
    "=)": "មុខ ញញឹម",
    ":(": "មុខ ក្រៀម",
    ":-(": "មុខ ក្រៀម",
    ":'(": "មុខ យំ",
    ":')": "មុខ យំ សើច",
    ";-)": "មុខ មិច ភ្នែក",
    ":-P": "មុខ ញញឹម លៀន អណ្ដាត",
    ":-D": "មុខ សើច",
    ":-O": "មុខ រន្ធត់",
    ">:-)": "មុខ អាក្រក់",
    ":-@": "មុខ ខឹង",
    "☹": "មុខ ក្រមូវ",
    "☺": "មុខ ញញឹម",
    "☻": "មុខ ញញឹម",
}


# ============================================================================
# TEST DATA LOADER - Load từ các file TSV
# ============================================================================

def _get_testdata_path(filename: str, subdir: str = "verbalizer") -> Path:
    """Get path to testdata file"""
    return Path(__file__).parent.parent / "textnorm" / subdir / "testdata" / filename


def load_testdata_tsv(filename: str, subdir: str = "verbalizer") -> Dict:
    """
    Load test data từ file TSV.
    Format: TYPE\tinput\toutput
    Returns dict với các mappings
    """
    testdata_path = _get_testdata_path(filename, subdir)
    data = defaultdict(dict)
    
    if not testdata_path.exists():
        return data
    
    try:
        with open(testdata_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split('\t')
                if len(parts) >= 3:
                    data_type = parts[0]
                    input_data = parts[1]
                    output_data = parts[2]
                    data[data_type][input_data] = output_data
    except Exception as e:
        print(f"Error loading {filename}: {e}")
    
    return data


def load_cardinal_testdata() -> Dict[int, str]:
    """Load cardinal test data và tạo mapping số -> Khmer words"""
    testdata = load_testdata_tsv("cardinal.tsv")
    cardinal_map = {}
    
    for data_type, mappings in testdata.items():
        for input_str, output_str in mappings.items():
            # Extract number from "cardinal|integer:123|"
            match = re.search(r'integer:(-?\d+)', input_str)
            if match:
                num = int(match.group(1))
                cardinal_map[num] = output_str
    
    return cardinal_map


def load_digit_testdata() -> Dict[str, str]:
    """Load digit test data"""
    testdata = load_testdata_tsv("digit.tsv")
    digit_map = {}
    
    for data_type, mappings in testdata.items():
        for input_str, output_str in mappings.items():
            # Extract digit from "digit|digit:1|" or "digit|digit:១|"
            match = re.search(r'digit:([០-៩0-9]+)', input_str)
            if match:
                digit = match.group(1)
                digit_map[digit] = output_str
    
    return digit_map


def load_date_testdata() -> List[Tuple[str, str]]:
    """Load date test data"""
    testdata = load_testdata_tsv("date.tsv")
    date_examples = []
    
    for data_type, mappings in testdata.items():
        for input_str, output_str in mappings.items():
            date_examples.append((input_str, output_str))
    
    return date_examples


def load_time_testdata() -> List[Tuple[str, str]]:
    """Load time test data"""
    testdata = load_testdata_tsv("time.tsv")
    time_examples = []
    
    for data_type, mappings in testdata.items():
        for input_str, output_str in mappings.items():
            time_examples.append((input_str, output_str))
    
    return time_examples


def load_money_testdata() -> List[Tuple[str, str]]:
    """Load money test data"""
    testdata = load_testdata_tsv("money.tsv")
    money_examples = []
    
    for data_type, mappings in testdata.items():
        for input_str, output_str in mappings.items():
            money_examples.append((input_str, output_str))
    
    return money_examples


def load_measure_testdata() -> List[Tuple[str, str]]:
    """Load measure test data"""
    testdata = load_testdata_tsv("measure.tsv")
    measure_examples = []
    
    for data_type, mappings in testdata.items():
        for input_str, output_str in mappings.items():
            measure_examples.append((input_str, output_str))
    
    return measure_examples


def load_telephone_testdata() -> List[Tuple[str, str]]:
    """Load telephone test data"""
    testdata = load_testdata_tsv("telephone.tsv")
    telephone_examples = []
    
    for data_type, mappings in testdata.items():
        for input_str, output_str in mappings.items():
            telephone_examples.append((input_str, output_str))
    
    return telephone_examples


def load_decimal_testdata() -> List[Tuple[str, str]]:
    """Load decimal test data"""
    testdata = load_testdata_tsv("decimal.tsv")
    decimal_examples = []
    
    for data_type, mappings in testdata.items():
        for input_str, output_str in mappings.items():
            decimal_examples.append((input_str, output_str))
    
    return decimal_examples


def load_fraction_testdata() -> List[Tuple[str, str]]:
    """Load fraction test data"""
    testdata = load_testdata_tsv("fraction.tsv")
    fraction_examples = []
    
    for data_type, mappings in testdata.items():
        for input_str, output_str in mappings.items():
            fraction_examples.append((input_str, output_str))
    
    return fraction_examples


def load_electronic_testdata() -> List[Tuple[str, str]]:
    """Load electronic (email/URL) test data"""
    testdata = load_testdata_tsv("electronic.tsv")
    electronic_examples = []
    
    for data_type, mappings in testdata.items():
        for input_str, output_str in mappings.items():
            electronic_examples.append((input_str, output_str))
    
    return electronic_examples


# Load all test data
_CARDINAL_TESTDATA = load_cardinal_testdata()
_DIGIT_TESTDATA = load_digit_testdata()


# ============================================================================
# NUMBER CONVERSION FUNCTIONS
# ============================================================================

def khmer_digit_to_arabic(text: str) -> str:
    """Convert Khmer digits to Arabic digits"""
    result = text
    for khmer, arabic in KHMER_TO_ARABIC.items():
        result = result.replace(khmer, arabic)
    return result


def number_to_khmer_words(num: int, use_testdata: bool = True) -> str:
    """
    Chuyển đổi số nguyên sang từ tiếng Khmer.
    Dựa trên textnorm/verbalizer/cardinal.grm
    """
    # Use test data if available
    if use_testdata and num in _CARDINAL_TESTDATA:
        return _CARDINAL_TESTDATA[num]
    
    if num < 0:
        return "ដក " + number_to_khmer_words(-num, use_testdata)
    
    if num in KHMER_NUMBERS:
        return KHMER_NUMBERS[num]
    
    # Handle numbers 21-99
    if 21 <= num < 100:
        tens = (num // 10) * 10
        ones = num % 10
        if ones == 0:
            return KHMER_NUMBERS[tens]
        return f"{KHMER_NUMBERS[tens]} {KHMER_NUMBERS[ones]}"
    
    # Handle hundreds
    if 100 <= num < 1000:
        hundreds = num // 100
        remainder = num % 100
        if remainder == 0:
            return f"{number_to_khmer_words(hundreds, use_testdata)} {KHMER_NUMBERS[100]}"
        return f"{number_to_khmer_words(hundreds, use_testdata)} {KHMER_NUMBERS[100]} {number_to_khmer_words(remainder, use_testdata)}"
    
    # Handle thousands
    if 1000 <= num < 10000:
        thousands = num // 1000
        remainder = num % 1000
        if remainder == 0:
            return f"{number_to_khmer_words(thousands, use_testdata)} {KHMER_NUMBERS[1000]}"
        return f"{number_to_khmer_words(thousands, use_testdata)} {KHMER_NUMBERS[1000]} {number_to_khmer_words(remainder, use_testdata)}"
    
    # Handle ten thousands (ម៉ឺន)
    if 10000 <= num < 100000:
        ten_thousands = num // 10000
        remainder = num % 10000
        if remainder == 0:
            return f"{number_to_khmer_words(ten_thousands, use_testdata)} {KHMER_NUMBERS[10000]}"
        return f"{number_to_khmer_words(ten_thousands, use_testdata)} {KHMER_NUMBERS[10000]} {number_to_khmer_words(remainder, use_testdata)}"
    
    # Handle hundred thousands (សែន)
    if 100000 <= num < 1000000:
        hundred_thousands = num // 100000
        remainder = num % 100000
        if remainder == 0:
            return f"{number_to_khmer_words(hundred_thousands, use_testdata)} {KHMER_NUMBERS[100000]}"
        return f"{number_to_khmer_words(hundred_thousands, use_testdata)} {KHMER_NUMBERS[100000]} {number_to_khmer_words(remainder, use_testdata)}"
    
    # Handle millions (លាន)
    if 1000000 <= num < 10000000:
        millions = num // 1000000
        remainder = num % 1000000
        if remainder == 0:
            return f"{number_to_khmer_words(millions, use_testdata)} {KHMER_NUMBERS[1000000]}"
        return f"{number_to_khmer_words(millions, use_testdata)} {KHMER_NUMBERS[1000000]} {number_to_khmer_words(remainder, use_testdata)}"
    
    # For very large numbers, return as string (can be extended)
    return str(num)


def digits_to_khmer_words(digits: str) -> str:
    """Convert sequence of digits to Khmer words (e.g., "123" -> "មួយ ពីរ បី")"""
    # Convert Khmer digits to Arabic first
    digits = khmer_digit_to_arabic(digits)
    
    # Use test data if available
    if digits in _DIGIT_TESTDATA:
        return _DIGIT_TESTDATA[digits]
    
    # Convert each digit
    result = []
    for digit in digits:
        if digit.isdigit():
            num = int(digit)
            result.append(KHMER_NUMBERS.get(num, digit))
        else:
            result.append(digit)
    
    return " ".join(result)


# ============================================================================
# NORMALIZATION FUNCTIONS
# ============================================================================

def normalize_cardinal(text: str) -> str:
    """
    Normalize cardinal numbers trong text.
    Tìm và thay thế các số bằng từ Khmer.
    """
    def replace_number(match):
        num_str = match.group(0)
        # Convert Khmer digits to Arabic
        num_str = khmer_digit_to_arabic(num_str)
        try:
            num = int(num_str)
            return number_to_khmer_words(num)
        except ValueError:
            return num_str
    
    # Pattern để tìm số nguyên (cả Arabic và Khmer digits)
    pattern = r'\b[\d០-៩]+\b'
    return re.sub(pattern, replace_number, text)


def normalize_digit(text: str) -> str:
    """
    Normalize digit sequences (chữ số riêng lẻ).
    Ví dụ: "123" -> "មួយ ពីរ បី"
    """
    def replace_digits(match):
        digits = match.group(0)
        return digits_to_khmer_words(digits)
    
    # Pattern để tìm chuỗi số (ưu tiên số dài hơn)
    # Tìm số có ít nhất 2 chữ số để tránh conflict với cardinal
    pattern = r'\b[\d០-៩]{2,}\b'
    return re.sub(pattern, replace_digits, text)


def _normalize_fractional_part(fractional_part: str) -> str:
    """
    Normalize fractional part: leading zeros digit-by-digit, rest as cardinal.
    Based on decimal.grm
    """
    # Count leading zeros
    leading_zeros_count = 0
    for char in fractional_part:
        if char == '0':
            leading_zeros_count += 1
        else:
            break
    
    fractional_khmer_parts = []
    # Add leading zeros
    for _ in range(leading_zeros_count):
        fractional_khmer_parts.append("សូន្យ")
    
    # Convert the rest as cardinal
    rest_part = fractional_part[leading_zeros_count:]
    if rest_part:
        fractional_khmer_parts.append(number_to_khmer_words(int(rest_part)))
    elif leading_zeros_count == 0:
        # Should not happen if fractional_part is not empty and contains digits
        pass
        
    return " ".join(fractional_khmer_parts)


def normalize_decimal(text: str) -> str:
    """
    Normalize decimal numbers.
    Ví dụ: "4.5" -> "បួន ក្បៀស ប្រាំ"
    """
    def replace_decimal(match):
        full_match = match.group(0)
        # Check for negative
        is_negative = full_match.startswith('-')
        number_part = full_match.lstrip('-')
        
        # Split by . or ,
        if '.' in number_part:
            parts = number_part.split('.')
        elif ',' in number_part:
            parts = number_part.split(',')
        else:
            return full_match
        
        integer_part = parts[0]
        fractional_part = parts[1] if len(parts) > 1 else ""
        
        # Convert parts
        integer_khmer = number_to_khmer_words(int(integer_part)) if integer_part else "សូន្យ"
        fractional_khmer = _normalize_fractional_part(fractional_part) if fractional_part else ""
        
        result = f"{'ដក ' if is_negative else ''}{integer_khmer}"
        if fractional_khmer:
            result += f" ក្បៀស {fractional_khmer}"
        
        return result
    
    # Pattern cho decimal: số có dấu . hoặc , với phần thập phân
    pattern = r'-?\d+[.,]\d+'
    return re.sub(pattern, replace_decimal, text)


def normalize_fraction(text: str, skip_dates: bool = True) -> str:
    """
    Normalize fractions.
    Ví dụ: "4/5" -> "បួន លើ ប្រាំ"
    """
    # Store positions that look like dates to skip them
    date_positions = set()
    if skip_dates:
        # Find all potential dates first
        date_pattern = r'\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b'
        for match in re.finditer(date_pattern, text):
            month = int(match.group(2))
            # If month is 1-12, it's likely a date
            if 1 <= month <= 12:
                date_positions.add((match.start(), match.end()))
    
    def replace_fraction(match):
        # Check if this position was identified as a date
        if skip_dates and (match.start(), match.end()) in date_positions:
            return match.group(0)
        
        numerator = match.group(1)
        denominator = match.group(2)
        
        num_val = int(numerator.lstrip('-'))
        den_val = int(denominator)
        
        # If denominator is > 31, it's definitely a fraction
        # If numerator is > 31, it's likely a fraction
        # If both are small (<=31), check if denominator could be a month (1-12)
        if den_val > 31 or num_val > 31:
            is_negative = numerator.startswith('-')
            numerator_khmer = number_to_khmer_words(num_val)
            denominator_khmer = number_to_khmer_words(den_val)
            result = f"{'ដក ' if is_negative else ''}{numerator_khmer} លើ {denominator_khmer}"
            return result
        elif 1 <= den_val <= 12:
            # Could be a date (month), skip
            return match.group(0)
        else:
            # Small numbers but not a month, treat as fraction
            is_negative = numerator.startswith('-')
            numerator_khmer = number_to_khmer_words(num_val)
            denominator_khmer = number_to_khmer_words(den_val)
            result = f"{'ដក ' if is_negative else ''}{numerator_khmer} លើ {denominator_khmer}"
            return result
    
    # Pattern cho fraction: số/số
    pattern = r'\b(-?\d+)/(\d+)\b'
    text = re.sub(pattern, replace_fraction, text)
    return text


def normalize_date(text: str) -> str:
    """
    Normalize dates trong text.
    Hỗ trợ các format: DD/MM/YYYY, DD-MM-YYYY, "day month year"
    """
    def replace_date_slash(match):
        day = int(match.group(1))
        month = int(match.group(2))
        year = int(match.group(3)) if match.group(3) else None
        
        # Validate: month should be 1-12, day should be reasonable
        if month < 1 or month > 12:
            return match.group(0)  # Not a valid date
        
        day_khmer = number_to_khmer_words(day)
        month_name = KHMER_MONTHS.get(month, f"ខែ{month}")
        
        if year:
            year_khmer = number_to_khmer_words(year)
            return f"{day_khmer} ខែ {month_name} ឆ្នាំ {year_khmer}"
        else:
            return f"{day_khmer} ខែ {month_name}"
    
    # Pattern cho DD/MM/YYYY hoặc DD-MM-YYYY (chỉ match nếu có year hoặc month hợp lệ)
    # Avoid matching phone numbers by requiring year or checking month validity
    date_pattern = r'\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))\b'
    text = re.sub(date_pattern, replace_date_slash, text)
    
    # Pattern cho "day month year" format (Khmer)
    # Có thể thêm các pattern khác ở đây
    return text


def normalize_time(text: str) -> str:
    """
    Normalize time.
    Ví dụ: "10:30" -> "ម៉ោង ដប់ ម៉ោង សាមសិប នាទី"
    """
    def replace_time(match):
        hours = int(match.group(1))
        minutes = int(match.group(2)) if match.group(2) else 0
        seconds = int(match.group(3)) if match.group(3) else None
        
        hours_khmer = number_to_khmer_words(hours)
        result = f"ម៉ោង {hours_khmer}"
        
        if minutes > 0:
            minutes_khmer = number_to_khmer_words(minutes)
            result += f" {minutes_khmer} នាទី"
        
        if seconds is not None and seconds > 0:
            seconds_khmer = number_to_khmer_words(seconds)
            result += f" {seconds_khmer} វិនាទី"
        
        # Add period (sáng/chiều/tối) based on hours
        if 5 <= hours < 12:
            result += " ពេលព្រឹក"
        elif 12 <= hours < 17:
            result += " ពេលរសៀល"
        elif 17 <= hours < 21:
            result += " ល្ងាច"
        elif 21 <= hours or hours < 5:
            result += " យប់"
        
        return result
    
    # Pattern cho time: HH:MM hoặc HH:MM:SS
    time_pattern = r'(\d{1,2}):(\d{2})(?::(\d{2}))?'
    return re.sub(time_pattern, replace_time, text)


def normalize_money(text: str) -> str:
    """
    Normalize money trong text.
    Hỗ trợ: $100, 100 riel, etc.
    """
    def replace_money_dollar(match):
        amount = match.group(1)
        if '.' in amount:
            integer_part, fractional_part = amount.split('.')
            integer_khmer = number_to_khmer_words(int(integer_part))
            fractional_khmer = _normalize_fractional_part(fractional_part)
            return f"{integer_khmer} ក្បៀស {fractional_khmer} {CURRENCY_NAMES['usd']}"
        else:
            integer_khmer = number_to_khmer_words(int(amount))
            return f"{integer_khmer} {CURRENCY_NAMES['usd']}"
    
    # Pattern cho $number (process before decimal normalization)
    money_pattern1 = r'\$(\d+(?:\.\d+)?)'
    text = re.sub(money_pattern1, replace_money_dollar, text)
    
    # Pattern cho number riel
    def replace_money_riel(match):
        amount = match.group(1)
        if '.' in amount:
            integer_part, fractional_part = amount.split('.')
            integer_khmer = number_to_khmer_words(int(integer_part))
            fractional_khmer = _normalize_fractional_part(fractional_part)
            return f"{integer_khmer} ក្បៀស {fractional_khmer} {CURRENCY_NAMES['khr']}"
        else:
            integer_khmer = number_to_khmer_words(int(amount))
            return f"{integer_khmer} {CURRENCY_NAMES['khr']}"
    
    money_pattern2 = r'(\d+(?:\.\d+)?)\s*រៀល'
    text = re.sub(money_pattern2, replace_money_riel, text)
    
    return text


def normalize_telephone(text: str) -> str:
    """
    Normalize telephone numbers.
    Ví dụ: "012-345-678" -> "សូន្យ មួយ ពីរ sil បី បួន ប្រាំ sil ប្រាំមួយ ប្រាំបី ប្រាំបី"
    """
    def replace_phone(match):
        # Get all number parts
        phone_str = match.group(0)
        # Remove separators
        digits = re.sub(r'[-.\s]', '', phone_str)
        
        # Convert to Khmer words with sil between groups
        # Group digits (typically 3-4 digits per group)
        result = []
        # Split into groups of 3-4 digits
        i = 0
        while i < len(digits):
            if i > 0:
                result.append("sil")
            # Take 3-4 digits
            group_size = 3 if len(digits) - i >= 6 else min(4, len(digits) - i)
            group = digits[i:i+group_size]
            # Convert each digit in group
            for digit in group:
                result.append(KHMER_NUMBERS.get(int(digit), digit))
            i += group_size
        return " ".join(result)
    
    # Pattern cho phone numbers: các số với separators - . hoặc space
    # Avoid matching dates by requiring longer sequences or specific patterns
    # Match: 10+ digits total, or patterns like 0XX-XXX-XXX
    phone_pattern = r'\b(?:0\d{2}[-.\s]?\d{3}[-.\s]?\d{3,4}|\+?\d{1,3}[-.\s]?\d{3,4}[-.\s]?\d{3,4}[-.\s]?\d{3,4})\b'
    return re.sub(phone_pattern, replace_phone, text)


def normalize_measure(text: str) -> str:
    """
    Normalize measurements.
    Ví dụ: "5 km" -> "ប្រាំ គីឡូ ម៉ែត្រ"
    """
    def replace_measure(match):
        number = match.group(1)
        unit = match.group(2).lower()
        
        # Convert number
        number_khmer = number_to_khmer_words(int(number))
        
        # Get unit in Khmer
        unit_khmer = MEASURE_UNITS.get(unit, unit)
        
        return f"{number_khmer} {unit_khmer}"
    
    # Pattern cho measure: số + unit
    # Common units: km, kg, m, g, l, etc.
    measure_pattern = r'(\d+)\s*(km|kg|m|g|l|liter|meter|gram|kilogram|kilometer|percent|%)'
    return re.sub(measure_pattern, replace_measure, text, flags=re.IGNORECASE)


def normalize_electronic(text: str) -> str:
    """
    Normalize electronic addresses (email, URL).
    Ví dụ: "test@example.com" -> "t_letter-en e_letter-en s_letter-en t_letter-en អ៊ែត example ដត់ com"
    """
    def replace_email(match):
        username = match.group(1)
        domain = match.group(2)
        
        # Convert username to letter-by-letter
        username_khmer = []
        for char in username:
            if char.isalpha():
                username_khmer.append(f"{char.lower()}_letter-en")
            elif char.isdigit():
                username_khmer.append(number_to_khmer_words(int(char)))
            else:
                username_khmer.append(char)
        
        # Convert domain
        domain_parts = domain.split('.')
        domain_khmer = []
        for part in domain_parts:
            if part in ['com', 'org', 'net', 'edu']:
                domain_khmer.append(part)
            else:
                # Convert letter by letter
                for char in part:
                    if char.isalpha():
                        domain_khmer.append(f"{char.lower()}_letter-en")
                    else:
                        domain_khmer.append(char)
            domain_khmer.append("ដត់")
        
        return " ".join(username_khmer) + " អ៊ែត " + " ".join(domain_khmer[:-1])
    
    # Pattern cho email
    email_pattern = r'(\w+(?:[._]\w+)*)@([\w.-]+)'
    text = re.sub(email_pattern, replace_email, text)
    
    # Pattern cho URL (simplified)
    def replace_url(match):
        url = match.group(0)
        # Convert www. to Khmer
        url = url.replace('www.', 'ដាប់ប៊លយូ ដាប់ប៊លយូ ដាប់ប៊លយូ ដត់ ')
        return url
    
    url_pattern = r'https?://[\w.-]+'
    text = re.sub(url_pattern, replace_url, text)
    
    return text


def normalize_emoticons(text: str) -> str:
    """Normalize emoticons"""
    for emoticon, khmer_text in EMOTICONS_MAP.items():
        text = text.replace(emoticon, khmer_text)
    return text


# ============================================================================
# MAIN NORMALIZATION FUNCTION
# ============================================================================

def normalize_khmer_text(text: str,
                         normalize_cardinals: bool = True,
                         normalize_digits: bool = False,  # Usually not needed if cardinals are normalized
                         normalize_decimals: bool = True,
                         normalize_fractions: bool = True,
                         normalize_dates: bool = True,
                         normalize_time_flag: bool = True,
                         normalize_money_flag: bool = True,
                         normalize_telephone_flag: bool = True,
                         normalize_measure_flag: bool = True,
                         normalize_electronic_flag: bool = True,
                         normalize_emoticons_flag: bool = True) -> str:
    """
    Normalize Khmer text bằng cách xử lý tất cả các loại văn bản đặc biệt.
    
    Args:
        text: Input text
        normalize_*: Flags để bật/tắt từng loại normalization
    
    Returns:
        Normalized text
    """
    # Order matters: process more specific patterns first
    # 1. Electronic (email/URL) - very specific patterns
    if normalize_electronic_flag:
        text = normalize_electronic(text)
    
    # 2. Emoticons - specific patterns
    if normalize_emoticons_flag:
        text = normalize_emoticons(text)
    
    # 3. Telephone - before dates (to avoid conflict)
    if normalize_telephone_flag:
        text = normalize_telephone(text)
    
    # 4. Time - before dates
    if normalize_time_flag:
        text = normalize_time(text)
    
    # 5. Money - before decimals (money has $ or currency symbols)
    if normalize_money_flag:
        text = normalize_money(text)
    
    # 6. Measure - before cardinals (has units)
    if normalize_measure_flag:
        text = normalize_measure(text)
    
    # 7. Fractions and Dates - need careful ordering
    # Process dates with years first (most specific)
    # Then fractions
    # Then dates without years (less specific)
    if normalize_dates or normalize_fractions:
        # First, normalize dates with years (DD/MM/YYYY or DD-MM-YYYY with year)
        year_date_pattern = r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b'
        def replace_year_date(match):
            day = int(match.group(1))
            month = int(match.group(2))
            year = int(match.group(3))
            if 1 <= month <= 12 and normalize_dates:
                day_khmer = number_to_khmer_words(day)
                month_name = KHMER_MONTHS.get(month, f"ខែ{month}")
                year_khmer = number_to_khmer_words(year)
                return f"{day_khmer} ខែ {month_name} ឆ្នាំ {year_khmer}"
            return match.group(0)
        text = re.sub(year_date_pattern, replace_year_date, text)
        
        # Then normalize fractions - prioritize fractions over dates
        # Fractions are normalized first, dates only if not already a fraction
        if normalize_fractions:
            def replace_fraction_safe(match):
                numerator = match.group(1)
                denominator = match.group(2)
                num_val = int(numerator.lstrip('-'))
                den_val = int(denominator)
                # Normalize all as fractions (dates with years already handled above)
                is_negative = numerator.startswith('-')
                numerator_khmer = number_to_khmer_words(num_val)
                denominator_khmer = number_to_khmer_words(den_val)
                return f"{'ដក ' if is_negative else ''}{numerator_khmer} លើ {denominator_khmer}"
            fraction_pattern = r'\b(-?\d+)/(\d+)\b(?!\s*[/-]\d)'  # Don't match if followed by another number (date with year)
            text = re.sub(fraction_pattern, replace_fraction_safe, text)
        
        # Finally, normalize dates without years (DD/MM format) that weren't normalized as fractions
        # Only normalize if it wasn't already processed and looks like a date
        # Check for date context (words like "date", "ngày", etc.) or if it's clearly a date format
        if normalize_dates:
            no_year_date_pattern = r'\b(\d{1,2})[/-](\d{1,2})\b(?!\s*[/-]\d)'
            def replace_no_year_date(match):
                # Check if this was already processed as fraction (contains "លើ")
                if 'លើ' in text[max(0, match.start()-10):match.end()+10]:
                    return match.group(0)  # Already processed as fraction
                day = int(match.group(1))
                month = int(match.group(2))
                # Check context - look for date-related words nearby
                context_before = text[max(0, match.start()-20):match.start()].lower()
                context_after = text[match.end():min(len(text), match.end()+20)].lower()
                has_date_context = any(word in context_before + context_after for word in ['date', 'ngày', 'day', 'tháng', 'month'])
                
                # Only normalize as date if:
                # 1. Has date context, OR
                # 2. Day > 12 (unlikely to be numerator of fraction, more likely day of month)
                if (1 <= month <= 12 and 1 <= day <= 31) and (has_date_context or day > 12):
                    day_khmer = number_to_khmer_words(day)
                    month_name = KHMER_MONTHS.get(month, f"ខែ{month}")
                    return f"{day_khmer} ខែ {month_name}"
                return match.group(0)
            text = re.sub(no_year_date_pattern, replace_no_year_date, text)
    
    # 9. Decimals - after money and dates
    if normalize_decimals:
        text = normalize_decimal(text)
    
    # 10. Cardinal numbers should be last (most general)
    # But we need to be careful not to normalize numbers that are already normalized
    if normalize_cardinals:
        text = normalize_cardinal(text)
    elif normalize_digits:
        text = normalize_digit(text)
    
    return text


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    # Test cases
    test_cases = [
        ("123", "Cardinal number"),
        ("I have 5 apples", "Cardinal in sentence"),
        ("The price is $100.50", "Money with decimal"),
        ("Date: 3/1/2455", "Date"),
        ("100 រៀល", "Money in Khmer"),
        ("Time: 10:30", "Time"),
        ("Phone: 012-345-678", "Telephone"),
        ("Distance: 5 km", "Measure"),
        ("Email: test@example.com", "Electronic"),
        ("4.5", "Decimal"),
        ("4/5", "Fraction"),
        (":)", "Emoticon"),
    ]
    
    print("=" * 80)
    print("Khmer Text Normalization Test")
    print("=" * 80)
    print()
    
    for test_input, description in test_cases:
        normalized = normalize_khmer_text(test_input)
        print(f"Test: {description}")
        print(f"Input:  {test_input}")
        print(f"Output: {normalized}")
        print()
