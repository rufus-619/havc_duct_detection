import re
from typing import List, Optional, Tuple
from app.models.domain import ParsedText, PressureClass

# Dimension patterns - more flexible for various HVAC drawing formats
# Matches: 12x8, 12X8, 12×8, 12"x8", 12"×8", 12 by 8, etc.
_DIM_RECT = re.compile(
    r'(\d{1,3})\s*["\u2033\']?\s*[xX\u00d7\*]\s*(\d{1,3})\s*["\u2033\']?|'  # 12x8, 12"x8", etc.
    r'(\d{1,3})\s*inch(?:es)?\s*[xX\u00d7]\s*(\d{1,3})\s*inch(?:es)?',  # 12 inches x 8 inches
    re.I
)

# Round duct patterns: 14Ø, 14D, 14"D, 14" dia, 14 inches dia, etc.
_DIM_ROUND = re.compile(
    r'(\d{1,3})\s*["\u2033\']?\s*[\u00d8\u2300Dd](?:ia)?|'  # 14Ø, 14D, 14"D
    r'(\d{1,3})\s*(?:inch(?:es)?)?\s*(?:dia|diameter)|'  # 14 dia, 14 inches diameter
    r'[\u00d8\u2300](\d{1,3})',  # Ø14 (diameter symbol before number)
    re.I
)

_PRESSURE = re.compile(r'\b(LP|MP|HP|Low\s+Pressure|Med(?:ium)?\s+Pressure|High\s+Pressure)\b', re.I)
_CFM = re.compile(r'(\d+)\s*CFM', re.I)

def parse_dimension(text: str) -> Optional[Tuple[float, float, float]]:
    """Returns (width_in, height_in, diameter_in). Unused fields are 0.0."""
    # Try rectangular patterns first
    rect_match = _DIM_RECT.search(text)
    if rect_match:
        # Check which groups matched (pattern has alternatives)
        if rect_match.group(1) and rect_match.group(2):
            return (float(rect_match.group(1)), float(rect_match.group(2)), 0.0)
        elif rect_match.group(3) and rect_match.group(4):
            return (float(rect_match.group(3)), float(rect_match.group(4)), 0.0)
    
    # Try round duct patterns
    round_match = _DIM_ROUND.search(text)
    if round_match:
        # Find the first non-None group which contains the diameter
        for group in round_match.groups():
            if group:
                return (0.0, 0.0, float(group))
        
    return None

def parse_pressure_class(text: str) -> PressureClass:
    """Returns PressureClass enum. Defaults to PressureClass.UNKNOWN."""
    match = _PRESSURE.search(text)
    if not match:
        return PressureClass.UNKNOWN
        
    val = match.group(1).lower()
    if val in ['lp', 'low pressure']:
        return PressureClass.LOW
    elif val in ['mp', 'med pressure', 'medium pressure']:
        return PressureClass.MEDIUM
    elif val in ['hp', 'high pressure']:
        return PressureClass.HIGH
        
    return PressureClass.UNKNOWN

def filter_domain_text(parsed: List[ParsedText]) -> List[ParsedText]:
    """Keeps only rows matching _DIM_RECT or _DIM_ROUND. Discards room numbers,
    electrical notes, grid labels, general notes, title block text."""
    filtered = []
    for item in parsed:
        if _DIM_RECT.search(item.text) or _DIM_ROUND.search(item.text):
            filtered.append(item)
    return filtered
