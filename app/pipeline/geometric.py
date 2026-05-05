"""
Geometric detection module — Morphological Fusion approach.

Instead of fragile O(N²) line pairing, we:
1. Separately extract horizontal and vertical line blobs via morphological OPEN
2. Bridge the gap between parallel walls with morphological CLOSE
3. Use findContours to get each duct as a solid blob
4. Filter by aspect ratio and area
5. Validate by OCR label proximity
"""
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import cv2

from app.core.config import settings

logger = logging.getLogger("hvac_analyzer.pipeline")

DEBUG_DIR = Path(__file__).resolve().parents[2] / "debug_images"
DIM_RE = re.compile(
    r'(\d{1,3}\s*["\u2033\']?\s*[xX\u00d7]\s*\d{1,3})|'
    r'(\d{1,3}\s*["\u2033\']?\s*[\u00d8Dd])|'
    r'([\u00d8]\s*\d{1,3})',
    re.IGNORECASE
)


@dataclass
class LineSegment:
    """Represents a synthetic wall line derived from a contour bounding rect."""
    x1: int
    y1: int
    x2: int
    y2: int
    angle: float
    length: float
    midpoint: Tuple[float, float]


@dataclass
class DuctSegment:
    """Represents a detected duct."""
    wall1: LineSegment
    wall2: LineSegment
    width_px: float
    length_px: float
    angle: float
    center: Tuple[float, float]
    duct_type: str = "unknown"


def _save_debug(name: str, image: np.ndarray) -> None:
    try:
        DEBUG_DIR.mkdir(exist_ok=True)
        cv2.imwrite(str(DEBUG_DIR / f"{name}.png"), image)
    except Exception:
        pass


def _make_duct_from_rect(x: int, y: int, w: int, h: int, is_horizontal: bool) -> DuctSegment:
    """Synthesise a DuctSegment from a bounding rectangle."""
    if is_horizontal:
        # Top and bottom walls
        wall1 = LineSegment(x, y, x + w, y, 0.0, float(w), (x + w / 2, float(y)))
        wall2 = LineSegment(x, y + h, x + w, y + h, 0.0, float(w), (x + w / 2, float(y + h)))
        angle = 0.0
        width_px = float(h)
        length_px = float(w)
    else:
        # Left and right walls
        wall1 = LineSegment(x, y, x, y + h, 90.0, float(h), (float(x), y + h / 2))
        wall2 = LineSegment(x + w, y, x + w, y + h, 90.0, float(h), (float(x + w), y + h / 2))
        angle = 90.0
        width_px = float(w)
        length_px = float(h)

    return DuctSegment(
        wall1=wall1,
        wall2=wall2,
        width_px=width_px,
        length_px=length_px,
        angle=angle,
        center=(x + w / 2, y + h / 2),
        duct_type="unknown"
    )


class GeometricDetectionService:
    """Detects duct geometry using morphological fusion + contour extraction."""

    def __init__(self):
        self.min_duct_width = settings.CV_DUCT_MIN_WIDTH_PX
        self.max_duct_width = settings.CV_DUCT_MAX_WIDTH_PX
        self.h_kernel_len = settings.CV_MORPH_H_KERNEL_LENGTH
        self.v_kernel_len = settings.CV_MORPH_V_KERNEL_LENGTH
        self.min_aspect = settings.CV_DUCT_MIN_ASPECT_RATIO
        self.min_area = settings.CV_DUCT_MIN_AREA_PX
        # Legacy — kept so colour classification callers don't break
        self.angle_tolerance = settings.CV_PARALLEL_ANGLE_TOLERANCE
        self.distance_tolerance = settings.CV_DUCT_WALL_DISTANCE_TOLERANCE

    # ------------------------------------------------------------------
    # Morphological line extraction
    # ------------------------------------------------------------------

    def _extract_orientation(
        self, binary: np.ndarray, horizontal: bool
    ) -> np.ndarray:
        """
        Extract only horizontal (or vertical) line blobs from a binary image.

        Uses a long, thin morphological OPEN to kill everything that isn't
        a long straight line in the desired orientation.
        """
        klen = self.h_kernel_len if horizontal else self.v_kernel_len
        if horizontal:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (klen, 1))
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, klen))
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    def _fuse_walls(self, line_mask: np.ndarray, horizontal: bool) -> np.ndarray:
        """
        Bridge the gap between two parallel duct walls so they form a solid blob.

        Uses a CLOSE kernel perpendicular to the duct direction, sized to be
        slightly larger than max duct width.
        """
        bridge = self.max_duct_width + 20
        if horizontal:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, bridge))
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (bridge, 1))
        return cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    def _contours_to_ducts(
        self, fused: np.ndarray, horizontal: bool
    ) -> List[DuctSegment]:
        """Extract duct candidates from a fused binary mask via findContours."""
        contours, _ = cv2.findContours(fused, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ducts = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area < self.min_area:
                continue
            long_side = max(w, h)
            short_side = min(w, h)
            if short_side == 0:
                continue
            aspect = long_side / short_side
            if aspect < self.min_aspect:
                continue
            # Width of the duct (perpendicular dimension) must be in valid range
            duct_width = h if horizontal else w
            if not (self.min_duct_width <= duct_width <= self.max_duct_width):
                continue
            ducts.append(_make_duct_from_rect(x, y, w, h, horizontal))
        return ducts

    # ------------------------------------------------------------------
    # Label proximity validation (replaces strict inside-box)
    # ------------------------------------------------------------------

    def _validate_by_proximity(
        self, ducts: List[DuctSegment], text_blocks: list
    ) -> List[DuctSegment]:
        """
        Keep ducts that have a dimension label within SPATIAL_MAX_DISTANCE_PX
        of the duct centre.
        """
        max_dist = settings.SPATIAL_MAX_DISTANCE_PX

        # Pre-filter: only text blocks that match dimension patterns
        dim_blocks = [tb for tb in text_blocks if DIM_RE.search(tb.text)]

        confirmed, unconfirmed = [], []
        for duct in ducts:
            cx, cy = duct.center
            found = False
            for tb in dim_blocks:
                tx, ty = tb.center
                dist = ((tx - cx) ** 2 + (ty - cy) ** 2) ** 0.5
                if dist <= max_dist:
                    found = True
                    break
            (confirmed if found else unconfirmed).append(duct)

        logger.info("pipeline.geometric.proximity_filter", extra={
            "before": len(ducts),
            "confirmed": len(confirmed),
            "discarded": len(unconfirmed),
            "dim_labels_available": len(dim_blocks)
        })

        # Debug image
        try:
            DEBUG_DIR.mkdir(exist_ok=True)
            all_xs = [int(d.center[0]) for d in ducts]
            all_ys = [int(d.center[1]) for d in ducts]
            tb_xs = [int(tb.center[0]) for tb in text_blocks]
            tb_ys = [int(tb.center[1]) for tb in text_blocks]
            W = max(all_xs + tb_xs, default=800) + 200
            H = max(all_ys + tb_ys, default=600) + 200
            canvas = np.ones((H, W, 3), dtype=np.uint8) * 240

            for d in unconfirmed:
                x, y, w, h = (int(d.wall1.x1), int(d.wall1.y1),
                               int(abs(d.wall2.x1 - d.wall1.x1) or d.length_px),
                               int(d.width_px))
                cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 140, 255), 2)
            for d in confirmed:
                x, y, w, h = (int(d.wall1.x1), int(d.wall1.y1),
                               int(abs(d.wall2.x1 - d.wall1.x1) or d.length_px),
                               int(d.width_px))
                cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 200, 0), 3)
            for tb in text_blocks:
                tx, ty = int(tb.center[0]), int(tb.center[1])
                is_dim = bool(DIM_RE.search(tb.text))
                color = (0, 0, 200) if is_dim else (180, 180, 180)
                cv2.circle(canvas, (tx, ty), 8 if is_dim else 3, color, -1)
                if is_dim and 0 <= tx < W and 0 <= ty < H:
                    cv2.putText(canvas, tb.text[:14], (tx + 4, ty - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 160), 1)
            cv2.imwrite(str(DEBUG_DIR / "04d_proximity_filter.png"), canvas)
        except Exception as e:
            print(f"[DEBUG IMAGE ERROR] {e}")

        return confirmed

    # ------------------------------------------------------------------
    # Colour classification (unchanged API)
    # ------------------------------------------------------------------

    def _sample_color_mask(
        self,
        color_mask: np.ndarray,
        center_x: float,
        center_y: float,
        angle: float,
        width: float,
        num_samples: int = 10
    ) -> float:
        h, w = color_mask.shape[:2]
        angle_rad = np.radians(angle)
        dx = np.cos(angle_rad)
        dy = np.sin(angle_rad)
        half_len = width * 2
        hits = 0
        for i in range(num_samples):
            t = (i / (num_samples - 1) - 0.5) * 2 * half_len
            px = int(center_x + t * dx)
            py = int(center_y + t * dy)
            if 0 <= px < w and 0 <= py < h:
                if color_mask[py, px] > 0:
                    hits += 1
        return hits / num_samples

    def classify_duct_color(
        self,
        duct: DuctSegment,
        blue_mask: np.ndarray,
        red_mask: np.ndarray
    ) -> str:
        center_x, center_y = duct.center
        blue_score = self._sample_color_mask(blue_mask, center_x, center_y, duct.angle, duct.width_px)
        red_score = self._sample_color_mask(red_mask, center_x, center_y, duct.angle, duct.width_px)
        COLOR_THRESHOLD = 0.2
        if blue_score > COLOR_THRESHOLD and blue_score > red_score:
            return "supply"
        elif red_score > COLOR_THRESHOLD and red_score > blue_score:
            return "return"
        return "unknown"

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def detect(
        self,
        binary_image: np.ndarray,
        grayscale: Optional[np.ndarray] = None,
        color_mask: Optional[np.ndarray] = None,
        text_blocks: Optional[list] = None
    ) -> List[DuctSegment]:
        """
        Detect duct segments using morphological fusion.

        1. Extract horizontal lines  → fuse walls → find contours
        2. Extract vertical lines    → fuse walls → find contours
        3. Merge, de-duplicate
        4. Validate by OCR proximity if text_blocks supplied
        """
        logger.info("pipeline.geometric.started", extra={
            "image_shape": binary_image.shape,
            "h_kernel": self.h_kernel_len,
            "v_kernel": self.v_kernel_len,
            "min_area": self.min_area,
            "min_aspect": self.min_aspect,
        })

        # Pass A — horizontal ducts
        h_lines = self._extract_orientation(binary_image, horizontal=True)
        h_fused = self._fuse_walls(h_lines, horizontal=True)
        h_ducts = self._contours_to_ducts(h_fused, horizontal=True)
        _save_debug("04e_h_lines", h_lines)
        _save_debug("04f_h_fused", h_fused)

        # Pass B — vertical ducts
        v_lines = self._extract_orientation(binary_image, horizontal=False)
        v_fused = self._fuse_walls(v_lines, horizontal=False)
        v_ducts = self._contours_to_ducts(v_fused, horizontal=False)
        _save_debug("04g_v_lines", v_lines)
        _save_debug("04h_v_fused", v_fused)

        ducts = h_ducts + v_ducts

        logger.info("pipeline.geometric.contours_found", extra={
            "horizontal": len(h_ducts),
            "vertical": len(v_ducts),
            "total": len(ducts)
        })

        print(f"[LABEL FILTER] ducts={len(ducts)}, text_blocks={len(text_blocks) if text_blocks else 0}")

        # Validate by proximity to dimension labels
        if text_blocks:
            ducts = self._validate_by_proximity(ducts, text_blocks)
        else:
            print("[LABEL FILTER] Skipped - no text_blocks provided")

        logger.info("pipeline.geometric.completed", extra={"duct_count": len(ducts)})
        return ducts
