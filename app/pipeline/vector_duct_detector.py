"""
Vector-based duct detection.

Input:  VectorPage (clean PDF line primitives, exact float coords).
Output: List[DuctSegment] (same type as the raster pipeline, so downstream
        mapping/annotation code is unchanged).

Algorithm (parallel wall pairing):

    1. Filter lines by length (drop short ticks, hatch strokes)
    2. Bucket by orientation: horizontal / vertical / diagonal
    3. Within each bucket, sort by perpendicular offset and pair lines that are:
         - parallel (already guaranteed by bucket)
         - separated by a duct-sized gap
         - overlapping along their shared direction
    4. Merge collinear duct segments that touch
    5. Convert to DuctSegment (pixel-space) for downstream code.

This only works because the input is vector — the same pairing logic was
hopeless on raster because Hough produced thousands of broken fragments.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from app.core.config import settings
from app.pipeline.geometric import DuctSegment, LineSegment
from app.pipeline.vector_extraction import VectorLine, VectorPage

logger = logging.getLogger("hvac_analyzer.pipeline")
DEBUG_DIR = Path(__file__).resolve().parents[2] / "debug_images"


# ---------------------------------------------------------------------------
# Orientation bucketing
# ---------------------------------------------------------------------------

_HORIZ_TOL = 5.0   # degrees
_VERT_TOL = 5.0    # degrees


def _is_horizontal(angle: float) -> bool:
    return angle < _HORIZ_TOL or angle > (180.0 - _HORIZ_TOL)


def _is_vertical(angle: float) -> bool:
    return abs(angle - 90.0) < _VERT_TOL


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _horiz_overlap(a: VectorLine, b: VectorLine) -> Tuple[float, float, float]:
    """Return (overlap_len, min_len, ratio) for two horizontal lines on x-axis."""
    a_min, a_max = sorted((a.x1, a.x2))
    b_min, b_max = sorted((b.x1, b.x2))
    overlap = max(0.0, min(a_max, b_max) - max(a_min, b_min))
    min_len = min(a_max - a_min, b_max - b_min)
    ratio = overlap / min_len if min_len > 0 else 0.0
    return overlap, min_len, ratio


def _vert_overlap(a: VectorLine, b: VectorLine) -> Tuple[float, float, float]:
    """Return (overlap_len, min_len, ratio) for two vertical lines on y-axis."""
    a_min, a_max = sorted((a.y1, a.y2))
    b_min, b_max = sorted((b.y1, b.y2))
    overlap = max(0.0, min(a_max, b_max) - max(a_min, b_min))
    min_len = min(a_max - a_min, b_max - b_min)
    ratio = overlap / min_len if min_len > 0 else 0.0
    return overlap, min_len, ratio


# ---------------------------------------------------------------------------
# Pair finding
# ---------------------------------------------------------------------------

def _length_similar(a: VectorLine, b: VectorLine, max_ratio: float) -> bool:
    la, lb = a.length, b.length
    if la == 0 or lb == 0:
        return False
    return max(la, lb) / min(la, lb) <= max_ratio


def _stroke_match(a: VectorLine, b: VectorLine, tol: float) -> bool:
    """Property #10: duct walls share stroke width."""
    return abs(a.width - b.width) <= tol


def _ends_aligned_horiz(a: VectorLine, b: VectorLine, tol: float) -> bool:
    """For horizontal lines: x-extents must roughly coincide on both ends."""
    ax_min, ax_max = sorted((a.x1, a.x2))
    bx_min, bx_max = sorted((b.x1, b.x2))
    return abs(ax_min - bx_min) <= tol and abs(ax_max - bx_max) <= tol


def _ends_aligned_vert(a: VectorLine, b: VectorLine, tol: float) -> bool:
    ay_min, ay_max = sorted((a.y1, a.y2))
    by_min, by_max = sorted((b.y1, b.y2))
    return abs(ay_min - by_min) <= tol and abs(ay_max - by_max) <= tol


def _pair_horizontal_lines(
    lines: List[VectorLine],
    min_gap: float,
    max_gap: float,
    min_overlap_ratio: float,
    max_length_ratio: float,
    end_offset_tol: float,
    stroke_tol: float,
) -> List[Tuple[VectorLine, VectorLine]]:
    """
    Sort by Y, window-scan for pairs that look like duct walls:
      - separation in [min_gap, max_gap]
      - high projection overlap
      - similar lengths
      - aligned endpoints
    Each line is used in at most one pair (greedy by best score).
    """
    lines = sorted(lines, key=lambda l: (l.y1 + l.y2) / 2.0)
    pairs: List[Tuple[VectorLine, VectorLine]] = []
    used = [False] * len(lines)
    n = len(lines)

    for i in range(n):
        if used[i]:
            continue
        yi = (lines[i].y1 + lines[i].y2) / 2.0
        best_j = -1
        best_score = 0.0
        for j in range(i + 1, n):
            if used[j]:
                continue
            yj = (lines[j].y1 + lines[j].y2) / 2.0
            gap = yj - yi
            if gap < min_gap:
                continue
            if gap > max_gap:
                break  # sorted → no further candidates
            if not _stroke_match(lines[i], lines[j], stroke_tol):
                continue
            if not _length_similar(lines[i], lines[j], max_length_ratio):
                continue
            if not _ends_aligned_horiz(lines[i], lines[j], end_offset_tol):
                continue
            _, _, ratio = _horiz_overlap(lines[i], lines[j])
            if ratio < min_overlap_ratio:
                continue
            score = ratio * min(lines[i].length, lines[j].length)
            if score > best_score:
                best_j = j
                best_score = score
        if best_j >= 0:
            pairs.append((lines[i], lines[best_j]))
            used[i] = True
            used[best_j] = True
    return pairs


def _pair_vertical_lines(
    lines: List[VectorLine],
    min_gap: float,
    max_gap: float,
    min_overlap_ratio: float,
    max_length_ratio: float,
    end_offset_tol: float,
    stroke_tol: float,
) -> List[Tuple[VectorLine, VectorLine]]:
    """Same as horizontal pairing, sorting by X."""
    lines = sorted(lines, key=lambda l: (l.x1 + l.x2) / 2.0)
    pairs: List[Tuple[VectorLine, VectorLine]] = []
    used = [False] * len(lines)
    n = len(lines)

    for i in range(n):
        if used[i]:
            continue
        xi = (lines[i].x1 + lines[i].x2) / 2.0
        best_j = -1
        best_score = 0.0
        for j in range(i + 1, n):
            if used[j]:
                continue
            xj = (lines[j].x1 + lines[j].x2) / 2.0
            gap = xj - xi
            if gap < min_gap:
                continue
            if gap > max_gap:
                break
            if not _stroke_match(lines[i], lines[j], stroke_tol):
                continue
            if not _length_similar(lines[i], lines[j], max_length_ratio):
                continue
            if not _ends_aligned_vert(lines[i], lines[j], end_offset_tol):
                continue
            _, _, ratio = _vert_overlap(lines[i], lines[j])
            if ratio < min_overlap_ratio:
                continue
            score = ratio * min(lines[i].length, lines[j].length)
            if score > best_score:
                best_j = j
                best_score = score
        if best_j >= 0:
            pairs.append((lines[i], lines[best_j]))
            used[i] = True
            used[best_j] = True
    return pairs


# ---------------------------------------------------------------------------
# Convert pair → DuctSegment (in pixel space)
# ---------------------------------------------------------------------------

def _pair_to_duct_segment(
    a: VectorLine,
    b: VectorLine,
    scale: float,
    horizontal: bool,
) -> DuctSegment:
    """Build a DuctSegment in pixel coordinates from a parallel wall pair."""
    # Pixel-space walls
    def to_px(ln: VectorLine) -> LineSegment:
        x1, y1 = ln.x1 * scale, ln.y1 * scale
        x2, y2 = ln.x2 * scale, ln.y2 * scale
        length = math.hypot(x2 - x1, y2 - y1)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
        return LineSegment(
            x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
            angle=angle, length=length,
            midpoint=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
        )

    w1, w2 = to_px(a), to_px(b)

    if horizontal:
        ya = (w1.y1 + w1.y2) / 2.0
        yb = (w2.y1 + w2.y2) / 2.0
        width_px = abs(yb - ya)
        x_lo = max(min(w1.x1, w1.x2), min(w2.x1, w2.x2))
        x_hi = min(max(w1.x1, w1.x2), max(w2.x1, w2.x2))
        length_px = max(0, x_hi - x_lo)
        center = ((x_lo + x_hi) / 2.0, (ya + yb) / 2.0)
        angle = 0.0
    else:
        xa = (w1.x1 + w1.x2) / 2.0
        xb = (w2.x1 + w2.x2) / 2.0
        width_px = abs(xb - xa)
        y_lo = max(min(w1.y1, w1.y2), min(w2.y1, w2.y2))
        y_hi = min(max(w1.y1, w1.y2), max(w2.y1, w2.y2))
        length_px = max(0, y_hi - y_lo)
        center = ((xa + xb) / 2.0, (y_lo + y_hi) / 2.0)
        angle = 90.0

    return DuctSegment(
        wall1=w1, wall2=w2,
        width_px=float(width_px),
        length_px=float(length_px),
        angle=angle,
        center=center,
        duct_type="unknown",
    )


# ---------------------------------------------------------------------------
# Collinear merge (Property #15 — spatial continuity)
# ---------------------------------------------------------------------------

def _duct_extent(d: DuctSegment) -> Tuple[float, float, float, float]:
    """Bounding rectangle of the duct (in pixel space)."""
    xs = [d.wall1.x1, d.wall1.x2, d.wall2.x1, d.wall2.x2]
    ys = [d.wall1.y1, d.wall1.y2, d.wall2.y1, d.wall2.y2]
    return min(xs), min(ys), max(xs), max(ys)


def _merge_collinear_ducts(
    ducts: List[DuctSegment],
    tol_px: float,
) -> List[DuctSegment]:
    """
    Greedy merge: two ducts merge if they share orientation, have similar
    centreline offset and width, and touch / overlap along the run direction.

    The merged duct uses the union bbox; wall1/wall2 are synthesized from
    that bbox (so downstream code that only reads center/length/width works).
    """
    if not ducts:
        return []

    # Two passes: horizontal ducts together, vertical ducts together
    horiz = [d for d in ducts if d.angle == 0.0]
    vert = [d for d in ducts if d.angle == 90.0]
    other = [d for d in ducts if d.angle not in (0.0, 90.0)]

    def merge_axis(group: List[DuctSegment], horizontal: bool) -> List[DuctSegment]:
        if not group:
            return []
        # Bucket by (centreline-offset, width) so only candidates that *could*
        # be the same duct are compared.
        items = []
        for d in group:
            x0, y0, x1, y1 = _duct_extent(d)
            cx, cy = d.center
            run_lo = x0 if horizontal else y0
            run_hi = x1 if horizontal else y1
            cross = cy if horizontal else cx
            items.append({
                "duct": d, "run_lo": run_lo, "run_hi": run_hi,
                "cross": cross, "width": d.width_px,
                "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            })
        # Sort by (cross, width, run_lo) so neighbours in the run are adjacent
        items.sort(key=lambda it: (it["cross"], it["width"], it["run_lo"]))

        merged_items: List[dict] = []
        for it in items:
            absorbed = False
            for m in merged_items:
                if abs(m["cross"] - it["cross"]) > tol_px:
                    continue
                if abs(m["width"] - it["width"]) > tol_px:
                    continue
                # Touch or overlap along run direction (allow small gap)
                if it["run_lo"] > m["run_hi"] + tol_px:
                    continue
                if m["run_lo"] > it["run_hi"] + tol_px:
                    continue
                # Merge (extend bbox)
                m["x0"] = min(m["x0"], it["x0"])
                m["y0"] = min(m["y0"], it["y0"])
                m["x1"] = max(m["x1"], it["x1"])
                m["y1"] = max(m["y1"], it["y1"])
                m["run_lo"] = min(m["run_lo"], it["run_lo"])
                m["run_hi"] = max(m["run_hi"], it["run_hi"])
                absorbed = True
                break
            if not absorbed:
                merged_items.append(dict(it))

        # Rebuild DuctSegments from merged bounding boxes
        out: List[DuctSegment] = []
        for m in merged_items:
            x0, y0, x1, y1 = m["x0"], m["y0"], m["x1"], m["y1"]
            if horizontal:
                w1 = LineSegment(int(x0), int(y0), int(x1), int(y0), 0.0, x1 - x0,
                                 ((x0 + x1) / 2, y0))
                w2 = LineSegment(int(x0), int(y1), int(x1), int(y1), 0.0, x1 - x0,
                                 ((x0 + x1) / 2, y1))
                width_px = abs(y1 - y0)
                length_px = abs(x1 - x0)
                angle = 0.0
            else:
                w1 = LineSegment(int(x0), int(y0), int(x0), int(y1), 90.0, y1 - y0,
                                 (x0, (y0 + y1) / 2))
                w2 = LineSegment(int(x1), int(y0), int(x1), int(y1), 90.0, y1 - y0,
                                 (x1, (y0 + y1) / 2))
                width_px = abs(x1 - x0)
                length_px = abs(y1 - y0)
                angle = 90.0
            center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
            out.append(DuctSegment(
                wall1=w1, wall2=w2,
                width_px=float(width_px),
                length_px=float(length_px),
                angle=angle, center=center, duct_type="unknown",
            ))
        return out

    return merge_axis(horiz, True) + merge_axis(vert, False) + other


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class VectorDuctDetectionService:
    """Detect ducts by pairing parallel vector lines."""

    def __init__(self) -> None:
        self.min_line_length = settings.VEC_MIN_LINE_LENGTH_PTS
        self.min_gap = settings.VEC_MIN_DUCT_WIDTH_PTS
        self.max_gap = settings.VEC_MAX_DUCT_WIDTH_PTS
        self.min_overlap = settings.VEC_MIN_OVERLAP_RATIO
        self.max_length_ratio = settings.VEC_MAX_LENGTH_RATIO
        self.end_offset_tol = settings.VEC_MAX_END_OFFSET_PTS
        self.stroke_tol = settings.VEC_STROKE_WIDTH_TOL

    def detect(
        self,
        page: VectorPage,
        scale: float,
        roi_pts: Optional[Tuple[float, float, float, float]] = None,
    ) -> List[DuctSegment]:
        """
        Args:
            page:    extracted vector page (PDF-unit coords).
            scale:   PDF-points → pixels multiplier (dpi / 72).
            roi_pts: optional (x0, y0, x1, y1) bbox in PDF points; lines whose
                     midpoint falls outside are dropped.  Use this to ignore
                     title blocks / notes / borders.

        Returns:
            List of DuctSegment in pixel coordinates.
        """
        # ROI filter (drop title block, notes, drawing border)
        candidate_lines = page.lines
        if roi_pts is not None:
            x0, y0, x1, y1 = roi_pts
            candidate_lines = [
                l for l in candidate_lines
                if x0 <= (l.x1 + l.x2) / 2.0 <= x1
                and y0 <= (l.y1 + l.y2) / 2.0 <= y1
            ]
            logger.info("vector.detect.roi", extra={
                "lines_before": len(page.lines),
                "lines_after_roi": len(candidate_lines),
                "roi_pts": roi_pts,
            })

        # Filter + bucket
        long_lines = [l for l in candidate_lines if l.length >= self.min_line_length]
        horiz = [l for l in long_lines if _is_horizontal(l.angle_deg)]
        vert = [l for l in long_lines if _is_vertical(l.angle_deg)]

        logger.info("vector.detect.buckets", extra={
            "total_lines": len(page.lines),
            "long_lines": len(long_lines),
            "horizontal": len(horiz),
            "vertical": len(vert),
        })

        horiz_pairs = _pair_horizontal_lines(
            horiz, self.min_gap, self.max_gap, self.min_overlap,
            self.max_length_ratio, self.end_offset_tol, self.stroke_tol,
        )
        vert_pairs = _pair_vertical_lines(
            vert, self.min_gap, self.max_gap, self.min_overlap,
            self.max_length_ratio, self.end_offset_tol, self.stroke_tol,
        )

        ducts: List[DuctSegment] = []
        for a, b in horiz_pairs:
            ducts.append(_pair_to_duct_segment(a, b, scale, horizontal=True))
        for a, b in vert_pairs:
            ducts.append(_pair_to_duct_segment(a, b, scale, horizontal=False))

        # Merge collinear/adjacent ducts (Property #15: spatial continuity)
        merge_tol_px = settings.VEC_MERGE_COLLINEAR_TOL_PTS * scale
        ducts = _merge_collinear_ducts(ducts, merge_tol_px)

        logger.info("vector.detect.done", extra={
            "horizontal_pairs": len(horiz_pairs),
            "vertical_pairs": len(vert_pairs),
            "ducts": len(ducts),
        })
        return ducts


# ---------------------------------------------------------------------------
# Debug rendering
# ---------------------------------------------------------------------------

def debug_draw_ducts(
    ducts: List[DuctSegment],
    out_shape: Tuple[int, int, int],
    filename: str,
    background=None,
) -> None:
    """Save a PNG showing all detected ducts as filled rectangles."""
    try:
        import numpy as np
        import cv2
        DEBUG_DIR.mkdir(exist_ok=True)

        if background is not None:
            canvas = background.copy()
        else:
            canvas = np.full(out_shape, 255, dtype=np.uint8)

        for d in ducts:
            w1, w2 = d.wall1, d.wall2
            pts = np.array([
                (w1.x1, w1.y1), (w1.x2, w1.y2),
                (w2.x2, w2.y2), (w2.x1, w2.y1),
            ], dtype=np.int32)
            cv2.fillPoly(canvas, [pts], (0, 200, 0))
            cv2.polylines(canvas, [pts], True, (0, 100, 0), 2)

        cv2.imwrite(str(DEBUG_DIR / filename), canvas)
        logger.info("vector.debug_ducts_saved", extra={"file": filename, "ducts": len(ducts)})
    except Exception as e:
        logger.warning("vector.debug_ducts_failed", extra={"err": str(e)})
