"""
Annotation rendering module.
Renders translucent overlays and dimension labels onto output image.
"""
import logging
from typing import List, Tuple, Optional
import numpy as np
import cv2

from app.core.config import settings
from app.pipeline.mapping import MappedDuct
from app.pipeline.extraction import DimensionLabel

logger = logging.getLogger("hvac_analyzer.pipeline")


class AnnotationService:
    """Renders annotated output images."""
    
    def __init__(self):
        self.line_thickness = settings.RENDER_DUCT_LINE_THICKNESS
        self.alpha = settings.RENDER_OVERLAY_ALPHA
        self.font_scale = settings.RENDER_FONT_SCALE
    
    def render(
        self,
        original_image: np.ndarray,
        mapped_ducts: List[MappedDuct]
    ) -> np.ndarray:
        """
        Render annotated image with duct highlights and dimension labels.
        
        Args:
            original_image: Original input image
            mapped_ducts: List of ducts with associated dimensions
            
        Returns:
            Annotated image (BGR)
        """
        logger.debug("pipeline.annotation.started", extra={"count": len(mapped_ducts)})
        
        # Create overlay for translucent effects
        overlay = original_image.copy()
        output = original_image.copy()
        
        # Draw each duct
        for mapped in mapped_ducts:
            self._draw_duct(overlay, mapped)
        
        # Apply overlay with alpha blending
        cv2.addWeighted(overlay, self.alpha, output, 1 - self.alpha, 0, output)
        
        # Draw dimension labels and callouts
        for mapped in mapped_ducts:
            self._draw_label(output, mapped)
        
        # Draw legend
        self._draw_legend(output)
        
        logger.debug("pipeline.annotation.completed")
        return output
    
    def _draw_duct(self, overlay: np.ndarray, mapped: MappedDuct) -> None:
        """Draw duct highlight on overlay - blue line through the center of duct."""
        duct = mapped.duct
        
        # Color based on duct type - bright vibrant colors
        if duct.duct_type == "supply":
            color = (255, 0, 0)  # Pure Blue (BGR) - bright blue
        elif duct.duct_type == "return":
            color = (0, 0, 255)  # Pure Red (BGR) - bright red
        else:
            color = (0, 255, 0)  # Pure Green (BGR) - bright green for unknown
        
        # Calculate center line between the two walls
        # Average the endpoints of wall1 and wall2 to get center line
        cx1 = int((duct.wall1.x1 + duct.wall2.x1) / 2)
        cy1 = int((duct.wall1.y1 + duct.wall2.y1) / 2)
        cx2 = int((duct.wall1.x2 + duct.wall2.x2) / 2)
        cy2 = int((duct.wall1.y2 + duct.wall2.y2) / 2)
        
        # Draw center line through the duct
        self._draw_thick_line(
            overlay,
            cx1, cy1,
            cx2, cy2,
            color,
            self.line_thickness
        )
    
    def _draw_thick_line(
        self,
        image: np.ndarray,
        x1: int, y1: int,
        x2: int, y2: int,
        color: Tuple[int, int, int],
        thickness: int
    ) -> None:
        """Draw a thick line using anti-aliased line with outline."""
        cv2.line(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    
    def _draw_label(self, image: np.ndarray, mapped: MappedDuct) -> None:
        """Draw dimension label callout."""
        duct = mapped.duct
        dim = mapped.dimension
        
        # Position label near duct center
        cx = int(duct.center[0])
        cy = int(duct.center[1])
        
        # Build text lines with dimension and pressure class
        lines = []
        
        # Line 1: Dimension
        if dim:
            lines.append(dim.dims_str)
        else:
            lines.append(f"Duct: {duct.duct_type.upper()}")
        
        # Line 2: Length
        lines.append(f"L: {mapped.real_length_ft:.1f} ft")
        
        # Line 3: Pressure Class (important!)
        if mapped.pressure_class:
            lines.append(f"Class: {mapped.pressure_class}")
        else:
            lines.append("Class: TBD")
        
        # Calculate text dimensions
        font = cv2.FONT_HERSHEY_SIMPLEX
        line_height = 25
        max_width = 0
        
        for line in lines:
            (w, h), _ = cv2.getTextSize(line, font, self.font_scale, 2)
            max_width = max(max_width, w)
        
        box_width = max_width + 20
        box_height = len(lines) * line_height + 10
        
        # Adjust position to keep on screen
        x1 = max(10, min(cx - box_width // 2, image.shape[1] - box_width - 10))
        y1 = max(10, min(cy - box_height // 2, image.shape[0] - box_height - 10))
        x2 = x1 + box_width
        y2 = y1 + box_height
        
        # Draw callout box with rounded corners
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), -1)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 2)
        
        # Draw text
        for i, line in enumerate(lines):
            y_pos = y1 + 20 + i * line_height
            cv2.putText(
                image, line,
                (x1 + 10, y_pos),
                font, self.font_scale, (0, 0, 0), 2, cv2.LINE_AA
            )
    
    def _draw_legend(self, image: np.ndarray) -> None:
        """Draw legend in bottom-left corner."""
        # Legend box
        legend_x = 20
        legend_y = image.shape[0] - 120
        legend_w = 250
        legend_h = 100
        
        cv2.rectangle(
            image,
            (legend_x, legend_y),
            (legend_x + legend_w, legend_y + legend_h),
            (255, 255, 255),
            -1
        )
        cv2.rectangle(
            image,
            (legend_x, legend_y),
            (legend_x + legend_w, legend_y + legend_h),
            (0, 0, 0),
            2
        )
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Supply duct - Blue
        self._draw_thick_line(
            image,
            legend_x + 15, legend_y + 25,
            legend_x + 45, legend_y + 25,
            (255, 0, 0),  # Blue
            8  # Thicker line
        )
        cv2.putText(
            image, "Supply (Blue)",
            (legend_x + 55, legend_y + 30),
            font, 0.5, (0, 0, 0), 1, cv2.LINE_AA
        )
        
        # Return duct - Red
        self._draw_thick_line(
            image,
            legend_x + 15, legend_y + 55,
            legend_x + 45, legend_y + 55,
            (0, 0, 255),  # Red
            8  # Thicker line
        )
        cv2.putText(
            image, "Return (Red)",
            (legend_x + 55, legend_y + 60),
            font, 0.5, (0, 0, 0), 1, cv2.LINE_AA
        )
        
        # Unknown - Green
        self._draw_thick_line(
            image,
            legend_x + 15, legend_y + 85,
            legend_x + 45, legend_y + 85,
            (0, 255, 0),  # Green
            8  # Thicker line
        )
        cv2.putText(
            image, "Unknown (Green)",
            (legend_x + 55, legend_y + 90),
            font, 0.5, (0, 0, 0), 1, cv2.LINE_AA
        )
    
    def encode_to_bytes(self, image: np.ndarray, format: str = "png") -> bytes:
        """
        Encode annotated image to bytes with high quality.
        
        Args:
            image: BGR image array
            format: Output format (png, jpg)
            
        Returns:
            Image bytes
        """
        if format.lower() == "png":
            # PNG with best compression quality (0 = no compression, 9 = max)
            # Use 3 for good balance of size and quality
            success, buffer = cv2.imencode(f".{format}", image, 
                [cv2.IMWRITE_PNG_COMPRESSION, 3])
        elif format.lower() in ["jpg", "jpeg"]:
            # JPEG with high quality (95%)
            success, buffer = cv2.imencode(f".{format}", image,
                [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            success, buffer = cv2.imencode(f".{format}", image)
            
        if not success:
            raise RuntimeError(f"Failed to encode image to {format}")
        return buffer.tobytes()
