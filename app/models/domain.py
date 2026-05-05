from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum

class DuctType(str, Enum):
    SUPPLY  = "supply"
    RETURN  = "return"
    UNKNOWN = "unknown"

class PressureClass(str, Enum):
    LOW     = "Low Pressure"
    MEDIUM  = "Medium Pressure"
    HIGH    = "High Pressure"
    UNKNOWN = "Unknown"

class BoundingBox(BaseModel):
    x: int       = Field(..., description="Top-left X coordinate in pixels")
    y: int       = Field(..., description="Top-left Y coordinate in pixels")
    width: int   = Field(..., gt=0, description="Width in pixels")
    height: int  = Field(..., gt=0, description="Height in pixels")

class LineSegment(BaseModel):
    """Represents a line segment for duct tracing (start and end points)."""
    x1: int = Field(..., description="Start X coordinate")
    y1: int = Field(..., description="Start Y coordinate")
    x2: int = Field(..., description="End X coordinate")
    y2: int = Field(..., description="End Y coordinate")

class ParsedText(BaseModel):
    """Stable output contract of OCRAdapter. Nothing else knows about Tesseract."""
    text:       str
    confidence: float = Field(..., ge=0.0, le=100.0)
    geometry:   BoundingBox

class DrawingScale(BaseModel):
    ratio:       int            # denominator D in "1/D" = 1'-0"
    px_per_foot: float          # 300 DPI * 1 / (D * 12)
    source:      str            # "title_block" | "scale_bar"

class DuctSegment(BaseModel):
    """Final aggregated entity — both geometries always present."""
    id:                   str
    duct_type:            DuctType
    pressure_class:       PressureClass
    dims_str:             str            # e.g. "12×8" or "14Ø"
    width_in:             float
    height_in:            float          # 0.0 if round duct
    diameter_in:          float          # 0.0 if rectangular
    length_ft:            float
    label_geometry:       BoundingBox
    duct_geometry:        BoundingBox
    duct_lines:           List[LineSegment] = []  # Traced duct line segments
    detection_confidence: Optional[float] = None
    fallback_used:        bool = False
    trace_id:             str            # propagated from request middleware
