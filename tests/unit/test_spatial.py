from app.models.domain import DuctSegment, ParsedText, BoundingBox, DuctType, PressureClass
from app.services.spatial import associate_pressure_class, deduplicate_segments

def test_nearest_pressure_label_association():
    seg1 = DuctSegment(
        id="s1", duct_type=DuctType.UNKNOWN, pressure_class=PressureClass.UNKNOWN,
        dims_str="12x8", width_in=12, height_in=8, diameter_in=0, length_ft=10,
        label_geometry=BoundingBox(x=100, y=100, width=50, height=20),
        duct_geometry=BoundingBox(x=100, y=100, width=50, height=20), trace_id="1"
    )
    
    labels = [
        ParsedText(text="LP", confidence=0.9, geometry=BoundingBox(x=120, y=130, width=30, height=20))
    ]
    
    res = associate_pressure_class([seg1], labels, max_distance_px=100)
    assert res[0].pressure_class == PressureClass.LOW

def test_orphan_label_gets_unknown_pressure():
    seg1 = DuctSegment(
        id="s1", duct_type=DuctType.UNKNOWN, pressure_class=PressureClass.UNKNOWN,
        dims_str="12x8", width_in=12, height_in=8, diameter_in=0, length_ft=10,
        label_geometry=BoundingBox(x=100, y=100, width=50, height=20),
        duct_geometry=BoundingBox(x=100, y=100, width=50, height=20), trace_id="1"
    )
    
    labels = [
        ParsedText(text="LP", confidence=0.9, geometry=BoundingBox(x=1000, y=1000, width=30, height=20))
    ]
    
    res = associate_pressure_class([seg1], labels, max_distance_px=100)
    assert res[0].pressure_class == PressureClass.UNKNOWN

def test_overlapping_segment_deduplication():
    # s1 and s2 overlap entirely
    seg1 = DuctSegment(
        id="s1", duct_type=DuctType.UNKNOWN, pressure_class=PressureClass.UNKNOWN,
        dims_str="12x8", width_in=12, height_in=8, diameter_in=0, length_ft=10,
        label_geometry=BoundingBox(x=100, y=100, width=50, height=20),
        duct_geometry=BoundingBox(x=100, y=100, width=50, height=20),
        detection_confidence=0.9, trace_id="1"
    )
    seg2 = DuctSegment(
        id="s2", duct_type=DuctType.UNKNOWN, pressure_class=PressureClass.UNKNOWN,
        dims_str="12x8", width_in=12, height_in=8, diameter_in=0, length_ft=10,
        label_geometry=BoundingBox(x=102, y=102, width=50, height=20),
        duct_geometry=BoundingBox(x=102, y=102, width=50, height=20),
        detection_confidence=0.8, trace_id="1"
    )
    
    res = deduplicate_segments([seg1, seg2])
    assert len(res) == 1
    assert res[0].id == "s1"
