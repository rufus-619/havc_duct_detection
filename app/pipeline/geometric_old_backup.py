"""
Geometric detection module.
Uses HoughLinesP to detect line segments and algorithmic heuristics
to group parallel lines into duct segments.
"""
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import cv2

from app.core.config import settings

logger = logging.getLogger("hvac_analyzer.pipeline")


@dataclass
class LineSegment:
    """Represents a detected line segment."""
    x1: int
    y1: int
    x2: int
    y2: int
    angle: float  # Degrees, 0-180
    length: float
    midpoint: Tuple[float, float]


@dataclass
class DuctSegment:
    """Represents a detected duct (pair of parallel walls)."""
    wall1: LineSegment
    wall2: LineSegment
    width_px: float
    length_px: float
    angle: float
    center: Tuple[float, float]
    duct_type: str = "unknown"  # 'supply', 'return', or 'unknown'


class GeometricDetectionService:
    """Detects duct geometry using Hough line detection."""
    
    def __init__(self):
        # Hough parameters
        self.rho_res = settings.CV_HOUGH_RHO_RESOLUTION
        self.theta_res = np.pi / 180.0 * settings.CV_HOUGH_THETA_RESOLUTION
        self.threshold = settings.CV_HOUGH_THRESHOLD
        self.min_line_length = settings.CV_HOUGH_MIN_LINE_LENGTH
        self.max_line_gap = settings.CV_HOUGH_MAX_LINE_GAP
        
        # Parallel line grouping parameters
        self.angle_tolerance = settings.CV_PARALLEL_ANGLE_TOLERANCE
        self.min_duct_width = settings.CV_DUCT_MIN_WIDTH_PX
        self.max_duct_width = settings.CV_DUCT_MAX_WIDTH_PX
        self.distance_tolerance = settings.CV_DUCT_WALL_DISTANCE_TOLERANCE
        self.min_line_thickness = settings.CV_DUCT_MIN_LINE_THICKNESS
    
    def _measure_line_thickness(
        self,
        grayscale: np.ndarray,
        seg: LineSegment,
        num_samples: int = 5
    ) -> float:
        """
        Measure average pixel thickness of a line in the original grayscale image.

        For each sample point along the line, scan perpendicular to the line
        direction and count consecutive dark pixels (value < 128).
        Returns the average thickness across all sample points.
        """
        h, w = grayscale.shape
        angle_rad = np.radians(seg.angle)
        # Perpendicular direction
        perp_dx = -np.sin(angle_rad)
        perp_dy =  np.cos(angle_rad)

        # Evenly spaced sample positions along the line (avoid end 10%)
        ts = np.linspace(0.1, 0.9, num_samples)
        thicknesses = []

        for t in ts:
            cx = int(seg.x1 + t * (seg.x2 - seg.x1))
            cy = int(seg.y1 + t * (seg.y2 - seg.y1))

            # Scan perpendicular, up to 30px each side
            thickness = 0
            for sign in (1, -1):
                for d in range(1, 31):
                    px = int(cx + sign * d * perp_dx)
                    py = int(cy + sign * d * perp_dy)
                    if 0 <= px < w and 0 <= py < h:
                        if grayscale[py, px] < 128:  # dark pixel
                            thickness += 1
                        else:
                            break
                    else:
                        break

            thicknesses.append(thickness)

        return float(np.mean(thicknesses)) if thicknesses else 0.0

    def _filter_thin_lines(
        self,
        segments: List[LineSegment],
        grayscale: np.ndarray
    ) -> List[LineSegment]:
        """
        Keep only line segments whose pixel thickness in the grayscale image
        is >= self.min_line_thickness.
        Logs how many lines were discarded.
        Saves debug image showing kept (green) vs discarded (red) lines.
        """
        thick = []
        thin_segs = []
        for seg in segments:
            t = self._measure_line_thickness(grayscale, seg)
            if t >= self.min_line_thickness:
                thick.append(seg)
            else:
                thin_segs.append(seg)

        logger.info("pipeline.geometric.thickness_filter", extra={
            "before": len(segments),
            "after": len(thick),
            "discarded_thin": len(thin_segs),
            "min_thickness_px": self.min_line_thickness
        })

        # Save debug image: green = kept thick lines, red = discarded thin lines
        try:
            from pathlib import Path
            debug_dir = Path(__file__).resolve().parents[2] / "debug_images"
            debug_dir.mkdir(exist_ok=True)
            debug = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)
            for seg in thin_segs:
                cv2.line(debug, (seg.x1, seg.y1), (seg.x2, seg.y2), (0, 0, 255), 2)  # Red = thin
            for seg in thick:
                cv2.line(debug, (seg.x1, seg.y1), (seg.x2, seg.y2), (0, 255, 0), 3)  # Green = thick
            cv2.imwrite(str(debug_dir / "04b_thickness_filter.png"), debug)
        except Exception:
            pass

        return thick

    def _extract_line_segments(self, lines: np.ndarray) -> List[LineSegment]:
        """Convert raw Hough lines to LineSegment objects."""
        segments = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate angle (0-180 degrees)
            dx = x2 - x1
            dy = y2 - y1
            angle = np.degrees(np.arctan2(abs(dy), abs(dx))) % 180
            
            # Calculate length
            length = np.sqrt(dx**2 + dy**2)
            
            # Calculate midpoint
            midpoint = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            segments.append(LineSegment(
                x1=int(x1), y1=int(y1),
                x2=int(x2), y2=int(y2),
                angle=angle,
                length=length,
                midpoint=midpoint
            ))
        
        return segments
    
    def _group_parallel_lines(self, segments: List[LineSegment]) -> List[List[LineSegment]]:
        """
        Group lines by similar angle (parallel lines).
        
        Returns:
            List of groups, each containing parallel lines
        """
        if not segments:
            return []
        
        # Sort by angle for easier grouping
        sorted_segments = sorted(segments, key=lambda s: s.angle)
        
        groups = []
        current_group = [sorted_segments[0]]
        
        for segment in sorted_segments[1:]:
            # Check if angle is similar to first in current group
            angle_diff = abs(segment.angle - current_group[0].angle)
            # Handle wrap-around at 180 degrees
            angle_diff = min(angle_diff, 180 - angle_diff)
            
            if angle_diff <= self.angle_tolerance:
                current_group.append(segment)
            else:
                if len(current_group) >= 2:  # Need at least 2 for duct walls
                    groups.append(current_group)
                current_group = [segment]
        
        # Add last group if valid
        if len(current_group) >= 2:
            groups.append(current_group)
        
        return groups
    
    def _form_duct_segments(
        self, 
        parallel_groups: List[List[LineSegment]],
        color_mask: Optional[np.ndarray] = None
    ) -> List[DuctSegment]:
        """
        Form duct segments by finding parallel line pairs at duct-width distance.
        
        If color_mask is provided, only include lines that overlap with
        colored regions (blue=supply, red=return).
        """
        ducts = []
        
        for group in parallel_groups:
            # Sort by perpendicular offset to center point
            angle_rad = np.radians(group[0].angle)
            
            # For horizontal-ish lines, sort by y-position
            # For vertical-ish lines, sort by x-position
            if group[0].angle < 45 or group[0].angle > 135:
                # Horizontal-ish: sort by y
                sorted_lines = sorted(group, key=lambda l: l.midpoint[1])
                perpendicular_idx = 1  # y-axis
            else:
                # Vertical-ish: sort by x
                sorted_lines = sorted(group, key=lambda l: l.midpoint[0])
                perpendicular_idx = 0  # x-axis
            
            # Find pairs that could be duct walls
            for i in range(len(sorted_lines)):
                for j in range(i + 1, len(sorted_lines)):
                    line1 = sorted_lines[i]
                    line2 = sorted_lines[j]
                    
                    # Calculate perpendicular distance
                    if perpendicular_idx == 1:
                        distance = abs(line2.midpoint[1] - line1.midpoint[1])
                    else:
                        distance = abs(line2.midpoint[0] - line1.midpoint[0])
                    
                    # Check if distance is in valid duct width range
                    if self.min_duct_width <= distance <= self.max_duct_width:
                        # Calculate average length
                        avg_length = (line1.length + line2.length) / 2
                        
                        # FILTER 1: Ducts must be reasonably long (at least 100px)
                        if avg_length < 100:
                            continue
                        
                        # FILTER 2: Aspect ratio - ducts are elongated (length/width > 3)
                        aspect_ratio = avg_length / distance
                        if aspect_ratio < 3.0:
                            continue
                        
                        # Calculate duct center
                        center_x = (line1.midpoint[0] + line2.midpoint[0]) / 2
                        center_y = (line1.midpoint[1] + line2.midpoint[1]) / 2
                        
                        # Check if this segment overlaps with colored duct lines
                        color_score = 1.0  # Default: no color mask = full score
                        if color_mask is not None:
                            # Sample color mask along the duct center line
                            color_score = self._sample_color_mask(
                                color_mask, center_x, center_y, line1.angle, distance
                            )
                            
                            # Require minimum color coverage to be a real duct
                            if color_score < 0.3:  # At least 30% blue/red coverage
                                logger.debug("pipeline.geometric.color_filtered", extra={
                                    "center": (center_x, center_y),
                                    "color_score": round(color_score, 2),
                                    "threshold": 0.3
                                })
                                continue
                            
                            # Determine duct type from color
                            duct_type = "supply" if color_score > 0.5 else "unknown"
                        else:
                            duct_type = "unknown"
                        
                        ducts.append(DuctSegment(
                            wall1=line1,
                            wall2=line2,
                            width_px=distance,
                            length_px=avg_length,
                            angle=line1.angle,
                            center=(center_x, center_y),
                            duct_type=duct_type
                        ))
        
        logger.info("pipeline.geometric.before_overlap_filter", extra={
            "raw_ducts": len(ducts)
        })
        
        # Remove overlapping ducts (keep largest for each region)
        ducts = self._remove_overlapping_ducts(ducts)
        
        # FILTER 3: Sanity check - max 20 ducts (if more, likely noise)
        if len(ducts) > 20:
            logger.warning("pipeline.geometric.too_many_ducts", extra={
                "detected": len(ducts),
                "keeping": 20
            })
            # Keep the 20 largest by area
            ducts = sorted(ducts, key=lambda d: d.width_px * d.length_px, reverse=True)[:20]
        
        logger.info("pipeline.geometric.after_overlap_filter", extra={
            "final_ducts": len(ducts)
        })
        
        return ducts
    
    def _sample_color_mask(
        self, 
        color_mask: np.ndarray,
        center_x: float,
        center_y: float,
        angle: float,
        duct_width: float
    ) -> float:
        """
        Sample color mask along a duct segment.
        Returns ratio of colored pixels (0-1).
        """
        h, w = color_mask.shape
        
        # Create sampling line across duct width
        angle_rad = np.radians(angle + 90)  # Perpendicular to duct
        dx = np.cos(angle_rad) * duct_width / 2
        dy = np.sin(angle_rad) * duct_width / 2
        
        # Sample points along perpendicular
        num_samples = 10
        colored_count = 0
        total_count = 0
        
        for t in np.linspace(-1, 1, num_samples):
            px = int(center_x + dx * t)
            py = int(center_y + dy * t)
            
            if 0 <= px < w and 0 <= py < h:
                total_count += 1
                if color_mask[py, px] > 0:
                    colored_count += 1
        
        return colored_count / total_count if total_count > 0 else 0.0
    
    def _remove_overlapping_ducts(self, ducts: List[DuctSegment]) -> List[DuctSegment]:
        """Remove overlapping duct detections, keeping the most confident ones."""
        if not ducts:
            return ducts
        
        # Sort by area (width * length) - prefer larger ducts
        sorted_ducts = sorted(ducts, key=lambda d: d.width_px * d.length_px, reverse=True)
        
        filtered = []
        for duct in sorted_ducts:
            # Check if this duct overlaps significantly with any already accepted
            overlaps = False
            for existing in filtered:
                distance = np.sqrt(
                    (duct.center[0] - existing.center[0])**2 +
                    (duct.center[1] - existing.center[1])**2
                )
                # Stricter overlap check: centers must be very close (< 20px)
                # AND similar size (within 50% of each other)
                size_similar = (0.5 < duct.width_px / existing.width_px < 2.0)
                if distance < 20 and size_similar:
                    overlaps = True
                    break
            
            if not overlaps:
                filtered.append(duct)
        
        return filtered
    
    def classify_duct_color(
        self,
        duct: DuctSegment,
        blue_mask: np.ndarray,
        red_mask: np.ndarray
    ) -> str:
        """
        Classify duct as supply (blue), return (red), or unknown based on color overlap.
        
        Args:
            duct: Detected duct segment
            blue_mask: Binary mask of blue regions
            red_mask: Binary mask of red regions
            
        Returns:
            'supply' if blue, 'return' if red, 'unknown' if neither or both
        """
        center_x, center_y = duct.center
        
        # Sample both color masks
        blue_score = self._sample_color_mask(
            blue_mask, center_x, center_y, duct.angle, duct.width_px
        )
        red_score = self._sample_color_mask(
            red_mask, center_x, center_y, duct.angle, duct.width_px
        )
        
        # Classify based on which color is stronger
        COLOR_THRESHOLD = 0.2  # At least 20% color coverage to classify
        
        if blue_score > COLOR_THRESHOLD and blue_score > red_score:
            return "supply"
        elif red_score > COLOR_THRESHOLD and red_score > blue_score:
            return "return"
        else:
            return "unknown"
    
    def _filter_by_label_inside(self, ducts: List[DuctSegment], text_blocks: list) -> List[DuctSegment]:
        """
        Keep only ducts that have a dimension label (e.g. 14"ø, 8"ø, 12x8) inside
        the duct bounding box.

        A text block is "inside" if its center falls within the rectangular
        region bounded by the two duct walls + a small margin.
        """
        import re
        # Pattern that matches HVAC dimension labels
        DIM_RE = re.compile(
            r'(\d{1,3}\s*["\u2033\']?\s*[xX×]\s*\d{1,3})|'  # rectangular: 12x8
            r'(\d{1,3}\s*["\u2033\']?\s*[ØDd])|'             # round: 14Ø, 14D
            r'([Ø]\s*\d{1,3})',                               # Ø14
            re.IGNORECASE
        )

        confirmed = []
        unconfirmed = []
        for duct in ducts:
            xs = [duct.wall1.x1, duct.wall1.x2, duct.wall2.x1, duct.wall2.x2]
            ys = [duct.wall1.y1, duct.wall1.y2, duct.wall2.y1, duct.wall2.y2]
            # Generous margin: labels can sit outside or at edge of walls
            margin = max(duct.width_px * 2.0, 100)
            x1 = min(xs) - margin
            x2 = max(xs) + margin
            y1 = min(ys) - margin
            y2 = max(ys) + margin

            found = False
            for tb in text_blocks:
                cx, cy = tb.center
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    if DIM_RE.search(tb.text):
                        found = True
                        break

            if found:
                confirmed.append(duct)
            else:
                unconfirmed.append(duct)

        logger.info("pipeline.geometric.label_inside_filter", extra={
            "before": len(ducts),
            "confirmed": len(confirmed),
            "discarded_no_label": len(unconfirmed)
        })

        # Save two debug images:
        # 04c_label_filter.png  - green=confirmed, orange=discarded duct lines
        # 04d_ocr_vs_ducts.png  - ALL text block centers + duct boxes overlaid on grayscale
        try:
            from pathlib import Path
            debug_dir = Path(__file__).resolve().parents[2] / "debug_images"
            debug_dir.mkdir(exist_ok=True)
            all_xs = [d.wall1.x1 for d in ducts] + [d.wall1.x2 for d in ducts]
            all_ys = [d.wall1.y1 for d in ducts] + [d.wall1.y2 for d in ducts]
            # Infer canvas size from text_blocks if no ducts
            if not all_xs and text_blocks:
                all_xs = [int(tb.x + tb.width) for tb in text_blocks]
                all_ys = [int(tb.y + tb.height) for tb in text_blocks]
            if all_xs and all_ys:
                W = max(all_xs) + 200
                H = max(all_ys) + 200

                # 04c: duct lines coloured by result
                canvas = np.ones((H, W, 3), dtype=np.uint8) * 240
                for d in unconfirmed:
                    cv2.line(canvas, (d.wall1.x1, d.wall1.y1), (d.wall1.x2, d.wall1.y2), (0, 140, 255), 2)
                    cv2.line(canvas, (d.wall2.x1, d.wall2.y1), (d.wall2.x2, d.wall2.y2), (0, 140, 255), 2)
                for d in confirmed:
                    cv2.line(canvas, (d.wall1.x1, d.wall1.y1), (d.wall1.x2, d.wall1.y2), (0, 200, 0), 3)
                    cv2.line(canvas, (d.wall2.x1, d.wall2.y1), (d.wall2.x2, d.wall2.y2), (0, 200, 0), 3)
                cv2.imwrite(str(debug_dir / "04c_label_filter.png"), canvas)

                # 04d: show every text block center (blue dot) + matched dim labels (red dot)
                #      plus duct bounding boxes (yellow rect)
                overlay = np.ones((H, W, 3), dtype=np.uint8) * 240
                # Draw duct search boxes
                for d in ducts:
                    xs2 = [d.wall1.x1, d.wall1.x2, d.wall2.x1, d.wall2.x2]
                    ys2 = [d.wall1.y1, d.wall1.y2, d.wall2.y1, d.wall2.y2]
                    mg = max(d.width_px * 2.0, 100)
                    bx1, bx2 = int(min(xs2) - mg), int(max(xs2) + mg)
                    by1, by2 = int(min(ys2) - mg), int(max(ys2) + mg)
                    bx1, by1 = max(0, bx1), max(0, by1)
                    bx2, by2 = min(W-1, bx2), min(H-1, by2)
                    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (0, 200, 200), 1)
                    cv2.line(overlay, (d.wall1.x1, d.wall1.y1), (d.wall1.x2, d.wall1.y2), (180, 180, 0), 2)
                    cv2.line(overlay, (d.wall2.x1, d.wall2.y1), (d.wall2.x2, d.wall2.y2), (180, 180, 0), 2)
                # Plot every text block
                for tb in text_blocks:
                    cx2, cy2 = int(tb.center[0]), int(tb.center[1])
                    if 0 <= cx2 < W and 0 <= cy2 < H:
                        is_dim = bool(DIM_RE.search(tb.text))
                        color = (0, 0, 220) if is_dim else (200, 200, 200)
                        radius = 8 if is_dim else 4
                        cv2.circle(overlay, (cx2, cy2), radius, color, -1)
                        if is_dim:
                            cv2.putText(overlay, tb.text[:12], (cx2 + 5, cy2 - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 180), 1)
                cv2.imwrite(str(debug_dir / "04d_ocr_vs_ducts.png"), overlay)
        except Exception as e:
            import traceback
            print(f"[DEBUG IMAGE ERROR] {e}")
            traceback.print_exc()

        return confirmed

    def detect(
        self,
        binary_image: np.ndarray,
        grayscale: Optional[np.ndarray] = None,
        color_mask: Optional[np.ndarray] = None,
        text_blocks: Optional[list] = None
    ) -> List[DuctSegment]:
        """
        Detect duct segments from binary image.
        
        Args:
            binary_image: Binary image with duct lines highlighted
            grayscale: Original grayscale image used to measure line thickness
            color_mask: Optional mask of colored duct regions (blue/red)
            text_blocks: OCR text blocks used to confirm ducts by label-inside check
            
        Returns:
            List of detected duct segments
        """
        mask_info = {}
        if color_mask is not None:
            mask_pixels = cv2.countNonZero(color_mask)
            mask_info = {
                "color_mask_pixels": mask_pixels,
                "has_color_mask": True
            }
        else:
            mask_info = {"has_color_mask": False}
        
        logger.info("pipeline.geometric.started", extra={
            "image_shape": binary_image.shape,
            "hough_threshold": self.threshold,
            "min_line_length": self.min_line_length,
            "max_line_gap": self.max_line_gap,
            "thickness_filter": grayscale is not None,
            **mask_info
        })
        
        # Detect all lines using HoughLinesP
        lines = cv2.HoughLinesP(
            binary_image,
            rho=self.rho_res,
            theta=self.theta_res,
            threshold=self.threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        if lines is None:
            nonzero = cv2.countNonZero(binary_image)
            logger.warning("pipeline.geometric.no_lines_detected", extra={
                "nonzero_pixels": nonzero,
                "total_pixels": binary_image.shape[0] * binary_image.shape[1],
                **mask_info
            })
            return []
        
        # Convert to LineSegment objects with computed properties
        line_segments = self._extract_line_segments(lines)
        logger.info("pipeline.geometric.lines_extracted", extra={
            "count": len(line_segments),
            **mask_info
        })
        
        # Filter by line thickness in the original grayscale image
        # This removes thin noise lines (grid, dimensions, text) keeping only bold duct walls
        if grayscale is not None:
            line_segments = self._filter_thin_lines(line_segments, grayscale)
        
        # Group parallel lines into potential duct walls
        parallel_groups = self._group_parallel_lines(line_segments)
        logger.info("pipeline.geometric.parallel_groups", extra={
            "count": len(parallel_groups),
            "total_lines": len(line_segments),
            **mask_info
        })
        
        # Form duct segments from parallel line pairs, with color filtering
        ducts = self._form_duct_segments(parallel_groups, color_mask)
        
        # Filter by dimension label inside duct bounding box
        # Ducts always have labels like 14"ø, 12x8 between their walls - noise doesn't
        print(f"[LABEL FILTER] ducts={len(ducts)}, text_blocks={len(text_blocks) if text_blocks else 0}")
        if text_blocks:
            ducts = self._filter_by_label_inside(ducts, text_blocks)
        else:
            print("[LABEL FILTER] Skipped - no text_blocks provided")
        
        logger.info("pipeline.geometric.completed", extra={
            "duct_count": len(ducts),
            "parallel_groups": len(parallel_groups),
            **mask_info
        })
        
        return ducts
