import cv2
import numpy as np
import logging
from typing import Optional, Tuple, List
from app.models.domain import ParsedText, BoundingBox, DuctType, LineSegment
from app.core.config import settings

logger = logging.getLogger("hvac_analyzer")


def _get_line_angle(x1, y1, x2, y2):
    """Calculate line angle in degrees (0-180)."""
    return np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1))) % 180


def _lines_are_parallel(angle1, angle2, tolerance=10):
    """Check if two line angles are parallel within tolerance."""
    diff = abs(angle1 - angle2)
    return diff < tolerance or diff > (180 - tolerance)


def _merge_parallel_lines(lines, angle_tolerance=10, distance_threshold=20):
    """
    Group parallel lines that are close together (likely duct walls).
    Returns list of (angle, [lines]) groups.
    """
    if not lines:
        return []
    
    # Calculate angles for all lines
    line_data = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = _get_line_angle(x1, y1, x2, y2)
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        line_data.append({
            'line': (x1, y1, x2, y2),
            'angle': angle,
            'length': length,
            'center': (cx, cy)
        })
    
    # Group by similar angle
    angle_groups = []
    for ld in line_data:
        found_group = False
        for group in angle_groups:
            if _lines_are_parallel(ld['angle'], group['angle'], angle_tolerance):
                group['lines'].append(ld)
                found_group = True
                break
        if not found_group:
            angle_groups.append({
                'angle': ld['angle'],
                'lines': [ld]
            })
    
    # Keep only groups with multiple lines (parallel pairs)
    return [g for g in angle_groups if len(g['lines']) >= 2]


def _extract_duct_segments_from_lines(line_groups, x_min, y_min):
    """
    Extract duct line segments from parallel line groups.
    Returns list of LineSegment objects mapped to full image coordinates.
    """
    segments = []
    
    for group in line_groups:
        # Sort lines by their center position perpendicular to angle
        lines = group['lines']
        if len(lines) < 2:
            continue
            
        # For horizontal-ish lines (angle near 0 or 180), sort by Y
        # For vertical-ish lines (angle near 90), sort by X
        angle = group['angle']
        
        if angle < 45 or angle > 135:  # Horizontal-ish
            lines.sort(key=lambda ld: ld['center'][1])
        else:  # Vertical-ish
            lines.sort(key=lambda ld: ld['center'][0])
        
        # Take the outermost pair as duct walls
        wall1 = lines[0]
        wall2 = lines[-1]
        
        # Create line segments for both walls
        for wall in [wall1, wall2]:
            x1, y1, x2, y2 = wall['line']
            segments.append(LineSegment(
                x1=int(x1 + x_min),
                y1=int(y1 + y_min),
                x2=int(x2 + x_min),
                y2=int(y2 + y_min)
            ))
    
    return segments


def detect_duct_for_label(
    image: np.ndarray,
    label: ParsedText,
    padding: int = settings.HVAC_PADDING_PX
) -> Tuple[BoundingBox, List[LineSegment], Optional[float], bool]:
    """
    Detect duct geometry by tracing parallel lines around the label.
    
    Returns:
        - duct_geometry: Bounding box around detected duct
        - duct_lines: List of LineSegment objects representing duct walls
        - detection_confidence: Confidence score (0-1)
        - fallback_used: True if line detection failed and label bbox used
    """
    h_img, w_img = image.shape[:2]
    
    # Expand label bounding box by padding
    x_min = max(0, label.geometry.x - padding)
    y_min = max(0, label.geometry.y - padding)
    x_max = min(w_img, label.geometry.x + label.geometry.width + padding)
    y_max = min(h_img, label.geometry.y + label.geometry.height + padding)
    
    crop = image[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return label.geometry, [], None, True
    
    # Grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # Adaptive Threshold for line detection
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Morphological Closing to connect line fragments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Canny Edge Detection
    edges = cv2.Canny(closed, 50, 150)
    
    # Hough Lines with adjusted parameters for duct walls
    lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=25, 
        minLineLength=30, 
        maxLineGap=20
    )
    
    duct_geometry = None
    duct_lines: List[LineSegment] = []
    detection_confidence = None
    fallback_used = True
    
    if lines is not None and len(lines) >= 2:
        # Find parallel line groups (duct walls)
        parallel_groups = _merge_parallel_lines(lines)
        
        if parallel_groups:
            # Extract duct line segments
            duct_lines = _extract_duct_segments_from_lines(parallel_groups, x_min, y_min)
            
            if duct_lines:
                # Calculate bounding box from line segments
                all_x = [p.x for seg in duct_lines for p in [(seg.x1, seg.y1), (seg.x2, seg.y2)] for p in [seg.x1, seg.x2]]
                all_y = [p.y for seg in duct_lines for p in [(seg.x1, seg.y1), (seg.x2, seg.y2)] for p in [seg.y1, seg.y2]]
                all_x = []
                all_y = []
                for seg in duct_lines:
                    all_x.extend([seg.x1, seg.x2])
                    all_y.extend([seg.y1, seg.y2])
                
                min_x, max_x = min(all_x), max(all_x)
                min_y, max_y = min(all_y), max(all_y)
                
                duct_geometry = BoundingBox(
                    x=int(min_x),
                    y=int(min_y),
                    width=int(max_x - min_x),
                    height=int(max_y - min_y)
                )
                
                # Confidence based on number of lines detected
                total_line_length = sum(
                    np.sqrt((seg.x2-seg.x1)**2 + (seg.y2-seg.y1)**2) 
                    for seg in duct_lines
                )
                crop_diag = np.sqrt(crop.shape[0]**2 + crop.shape[1]**2)
                detection_confidence = min(1.0, total_line_length / (crop_diag * 0.5))
                fallback_used = False
    
    if fallback_used:
        # FALLBACK: use label bounding box
        duct_geometry = label.geometry
        detection_confidence = None
        
        logger.warning(
            "duct_detection_fallback",
            extra={
                "extra_data": {
                    "label_text": label.text,
                    "label_bbox": label.geometry.model_dump()
                }
            }
        )
    
    return duct_geometry, duct_lines, detection_confidence, fallback_used

def classify_duct_type(image: np.ndarray, duct_geometry: BoundingBox) -> DuctType:
    # Sample mean BGR color
    x, y, w, h = duct_geometry.x, duct_geometry.y, duct_geometry.width, duct_geometry.height
    crop = image[y:y+h, x:x+w]
    if crop.size == 0:
        return DuctType.UNKNOWN
        
    b_mean = np.mean(crop[:,:,0])
    g_mean = np.mean(crop[:,:,1])
    r_mean = np.mean(crop[:,:,2])
    
    if b_mean > r_mean + g_mean * 0.5:
        return DuctType.SUPPLY
    else:
        return DuctType.RETURN
