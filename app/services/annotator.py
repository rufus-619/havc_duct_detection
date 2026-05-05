import io
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple
from app.models.domain import DuctSegment, DuctType, LineSegment

def _draw_thick_line(draw, x1: int, y1: int, x2: int, y2: int, 
                     color: Tuple[int, int, int, int], thickness: int = 8):
    """Draw a thick line segment using PIL."""
    # Calculate perpendicular offset for thickness
    dx = x2 - x1
    dy = y2 - y1
    length = np.sqrt(dx*dx + dy*dy)
    if length == 0:
        return
    
    # Normalize
    ux = -dy / length
    uy = dx / length
    
    # Create polygon for thick line
    half_thick = thickness / 2
    poly = [
        (x1 + ux * half_thick, y1 + uy * half_thick),
        (x2 + ux * half_thick, y2 + uy * half_thick),
        (x2 - ux * half_thick, y2 - uy * half_thick),
        (x1 - ux * half_thick, y1 - uy * half_thick),
    ]
    draw.polygon(poly, fill=color)


def _get_line_endpoints(seg: DuctSegment) -> List[Tuple[int, int, int, int]]:
    """Get line endpoints for drawing. Uses duct_lines if available, else creates from bbox."""
    if seg.duct_lines:
        return [(ln.x1, ln.y1, ln.x2, ln.y2) for ln in seg.duct_lines]
    
    # Fallback: create horizontal line through bbox center
    x0, y0 = seg.duct_geometry.x, seg.duct_geometry.y
    w, h = seg.duct_geometry.width, seg.duct_geometry.height
    cy = y0 + h // 2
    return [(x0, cy, x0 + w, cy)]


def render_annotated_image(image: np.ndarray, segments: List[DuctSegment]) -> bytes:
    """
    Render annotated image with duct lines highlighted.
    
    Args:
        image: Input BGR image from OpenCV
        segments: List of detected duct segments with line geometry
        
    Returns:
        PNG image bytes with annotations
    """
    # Convert BGR to RGB PIL Image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    base = Image.fromarray(image_rgb).convert("RGBA")
    
    # Create transparent overlay for annotations
    overlay = Image.new("RGBA", base.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    
    has_fallback = any(seg.fallback_used for seg in segments)
    
    # Load fonts
    try:
        font = ImageFont.truetype("arial.ttf", 20)
        small_font = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    for seg in segments:
        # Color scheme: Blue for supply, Gray for return/unknown
        if seg.duct_type == DuctType.SUPPLY:
            line_color = (30, 100, 220, 180)  # Blue semi-transparent
            border_color = (20, 80, 180, 255)
        else:
            line_color = (120, 120, 120, 180)  # Gray semi-transparent
            border_color = (80, 80, 80, 255)
        
        # Draw thick lines along duct path
        line_endpoints = _get_line_endpoints(seg)
        for x1, y1, x2, y2 in line_endpoints:
            _draw_thick_line(draw, x1, y1, x2, y2, line_color, thickness=10)
            # Add subtle border
            _draw_thick_line(draw, x1, y1, x2, y2, border_color, thickness=12)
        
        # Draw label geometry outline if fallback was used
        if seg.fallback_used:
            lx0, ly0 = seg.label_geometry.x, seg.label_geometry.y
            lx1, ly1 = lx0 + seg.label_geometry.width, ly0 + seg.label_geometry.height
            # Dashed outline effect using multiple small rectangles
            dash_len = 8
            gap_len = 4
            # Top edge
            for x in range(lx0, lx1, dash_len + gap_len):
                draw.rectangle([x, ly0, min(x + dash_len, lx1), ly0 + 2], fill=(255, 191, 0, 255))
            # Bottom edge
            for x in range(lx0, lx1, dash_len + gap_len):
                draw.rectangle([x, ly1 - 2, min(x + dash_len, lx1), ly1], fill=(255, 191, 0, 255))
            # Left edge
            for y in range(ly0, ly1, dash_len + gap_len):
                draw.rectangle([lx0, y, lx0 + 2, min(y + dash_len, ly1)], fill=(255, 191, 0, 255))
            # Right edge
            for y in range(ly0, ly1, dash_len + gap_len):
                draw.rectangle([lx1 - 2, y, lx1, min(y + dash_len, ly1)], fill=(255, 191, 0, 255))
        
        # Draw callout box with dimensions
        # Position near first line endpoint or bbox center
        if line_endpoints:
            x1, y1, x2, y2 = line_endpoints[0]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
        else:
            cx = seg.duct_geometry.x + seg.duct_geometry.width // 2
            cy = seg.duct_geometry.y + seg.duct_geometry.height // 2
        
        # Build callout text
        lines = [seg.dims_str, f"{seg.length_ft:.1f} ft"]
        if seg.pressure_class and seg.pressure_class != "Unknown":
            lines.append(seg.pressure_class)
        if seg.fallback_used:
            lines.append("⚠ fallback")
        
        # Calculate text dimensions
        line_heights = [20] * len(lines)
        box_h = sum(line_heights) + 10
        max_text_w = max([draw.textbbox((0, 0), line, font=font if i == 0 else small_font)[2] 
                         for i, line in enumerate(lines)]) if lines else 80
        box_w = max(max_text_w + 20, 100)
        
        # Draw callout box with rounded corners
        box_x = cx - box_w // 2
        box_y = cy - box_h // 2
        draw.rounded_rectangle(
            [box_x, box_y, box_x + box_w, box_y + box_h],
            radius=6,
            fill=(255, 255, 255, 230),
            outline=(0, 0, 0, 255),
            width=2
        )
        
        # Draw text lines
        text_y = box_y + 5
        for i, line in enumerate(lines):
            fill = (255, 140, 0, 255) if "⚠" in line else (0, 0, 0, 255)
            text_font = font if i == 0 else small_font
            draw.text((box_x + 10, text_y), line, fill=fill, font=text_font)
            text_y += line_heights[i]
    
    # Draw legend
    legend_x, legend_y = 50, base.size[1] - 180
    legend_w, legend_h = 320, 130
    draw.rectangle(
        [legend_x, legend_y, legend_x + legend_w, legend_y + legend_h],
        fill=(255, 255, 255, 220),
        outline=(0, 0, 0, 255),
        width=2
    )
    
    # Legend items
    legend_item_y = legend_y + 15
    # Supply duct indicator
    _draw_thick_line(draw, legend_x + 15, legend_item_y + 10, legend_x + 45, legend_item_y + 10, 
                     (30, 100, 220, 180), thickness=8)
    draw.text((legend_x + 55, legend_item_y), "Supply duct", fill=(0, 0, 0, 255), font=font)
    
    # Return duct indicator
    legend_item_y += 30
    _draw_thick_line(draw, legend_x + 15, legend_item_y + 10, legend_x + 45, legend_item_y + 10,
                     (120, 120, 120, 180), thickness=8)
    draw.text((legend_x + 55, legend_item_y), "Return duct", fill=(0, 0, 0, 255), font=font)
    
    # Fallback indicator
    if has_fallback:
        legend_item_y += 30
        # Dashed line indicator
        for x in range(legend_x + 15, legend_x + 45, 10):
            draw.rectangle([x, legend_item_y + 8, min(x + 6, legend_x + 45), legend_item_y + 12], 
                          fill=(255, 191, 0, 255))
        draw.text((legend_x + 55, legend_item_y), "⚠ Fallback (label only)", 
                 fill=(255, 140, 0, 255), font=small_font)
    
    # Composite overlay onto base image
    out = Image.alpha_composite(base, overlay)
    out_rgb = out.convert("RGB")
    
    # Save to bytes
    buf = io.BytesIO()
    out_rgb.save(buf, format="PNG", dpi=(300, 300))
    return buf.getvalue()
