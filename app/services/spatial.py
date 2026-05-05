import math
from typing import List
from app.models.domain import DuctSegment, ParsedText, PressureClass
from app.services.extractor import _PRESSURE

def _distance(bb1, bb2) -> float:
    # Calculate center distance between two bounding boxes
    c1 = (bb1.x + bb1.width / 2, bb1.y + bb1.height / 2)
    c2 = (bb2.x + bb2.width / 2, bb2.y + bb2.height / 2)
    return math.dist(c1, c2)

def associate_pressure_class(
    segments: List[DuctSegment],
    all_parsed_text: List[ParsedText],
    max_distance_px: int = 200
) -> List[DuctSegment]:
    
    pressure_labels = [p for p in all_parsed_text if _PRESSURE.search(p.text)]
    
    for segment in segments:
        if segment.pressure_class != PressureClass.UNKNOWN:
            continue
            
        closest_label = None
        min_dist = float('inf')
        
        for label in pressure_labels:
            d = _distance(segment.label_geometry, label.geometry)
            if d < min_dist and d <= max_distance_px:
                min_dist = d
                closest_label = label
                
        if closest_label:
            from app.services.extractor import parse_pressure_class
            segment.pressure_class = parse_pressure_class(closest_label.text)
            
    return segments

def deduplicate_segments(segments: List[DuctSegment]) -> List[DuctSegment]:
    """Remove segments whose duct_geometry overlaps > 80% with another segment.
    Keep the one with higher detection_confidence."""
    
    def _overlap_area(bb1, bb2):
        dx = min(bb1.x + bb1.width, bb2.x + bb2.width) - max(bb1.x, bb2.x)
        dy = min(bb1.y + bb1.height, bb2.y + bb2.height) - max(bb1.y, bb2.y)
        if (dx >= 0) and (dy >= 0):
            return dx * dy
        return 0

    to_remove = set()
    n = len(segments)
    
    for i in range(n):
        if i in to_remove:
            continue
        for j in range(i + 1, n):
            if j in to_remove:
                continue
                
            s1, s2 = segments[i], segments[j]
            area1 = s1.duct_geometry.width * s1.duct_geometry.height
            area2 = s2.duct_geometry.width * s2.duct_geometry.height
            
            overlap = _overlap_area(s1.duct_geometry, s2.duct_geometry)
            
            # Check if overlap is > 80% of the smaller area
            min_area = min(area1, area2)
            if min_area > 0 and overlap / min_area > 0.8:
                # Keep the one with higher confidence
                c1 = s1.detection_confidence or 0.0
                c2 = s2.detection_confidence or 0.0
                
                if c1 >= c2:
                    to_remove.add(j)
                else:
                    to_remove.add(i)
                    break # s1 is removed, no need to check further against s1
                    
    return [s for i, s in enumerate(segments) if i not in to_remove]
