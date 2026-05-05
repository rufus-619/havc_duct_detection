import math
import re
import numpy as np
from app.models.domain import DrawingScale, ParsedText
from app.services.ocr_service import OCRAdapter

def extract_scale(image: np.ndarray, ocr_adapter: OCRAdapter) -> DrawingScale:
    # 1. Crop bottom-right 25% of image
    h, w = image.shape[:2]
    crop = image[int(h*0.75):h, int(w*0.75):w]
    
    parsed = ocr_adapter.extract(crop)
    
    # 2. Regex: r'1\s*/\s*(\d+)\s*"?\s*=\s*1\s*\'-?\s*0\s*"'
    # e.g., 1/4" = 1'-0" -> denominator 4
    scale_regex = re.compile(r'1\s*/\s*(\d+)\s*"?\s*=\s*1\s*\'?-?\s*0\s*"?', re.I)
    
    for item in parsed:
        match = scale_regex.search(item.text)
        if match:
            denominator = int(match.group(1))
            px_per_foot = (300 * 1) / (denominator * 12)
            return DrawingScale(ratio=denominator, px_per_foot=px_per_foot, source="title_block")
            
    # Default to 1/4" = 1'-0" if not found
    return DrawingScale(ratio=4, px_per_foot=6.25, source="default")

def measure_length(pt1: tuple, pt2: tuple, scale: DrawingScale) -> float:
    pixel_length = math.dist(pt1, pt2)
    raw_ft = pixel_length / scale.px_per_foot
    return round(raw_ft * 2) / 2
