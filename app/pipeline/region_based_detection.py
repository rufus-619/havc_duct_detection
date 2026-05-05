"""
Region-Based Duct Detection Pipeline — LEGACY APPROACH

NOTE: This was Path 1 of our exploration (~30 hours). The raster/OCR-based approach
was abandoned due to fundamental limitations: ducts and structural lines are identical
in pixel space. See README.md "Path 1: Region-Based Detection" for full post-mortem.

Current pipeline uses vector extraction (vector_extraction.py + vector_duct_detector.py).
This file remains as documentation of the problem-solving journey.

Original approach:
Replaces fragile Hough line-pair matching with a connected-region + topology approach.

Pipeline:
  Stage 1 — OCR Masking      : erase text strokes from binary image
  Stage 2 — Region Bridging  : morph-close to fill duct interiors
  Stage 3 — CC Extraction    : connected components → candidate regions
  Stage 4 — Contour Scoring  : filter by elongation / rectangularity
  Stage 5 — Skeletonisation  : thin each region to a 1-px centreline
  Stage 6 — Graph Building   : endpoints, junctions, edge paths

All stages save debug PNGs to debug_images/.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.core.config import settings

logger = logging.getLogger("hvac_analyzer.pipeline")

DEBUG_DIR = Path(__file__).resolve().parents[2] / "debug_images"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(name: str, img: np.ndarray) -> None:
    """Write a debug PNG, never raises."""
    try:
        DEBUG_DIR.mkdir(exist_ok=True)
        cv2.imwrite(str(DEBUG_DIR / f"{name}.png"), img)
    except Exception as exc:
        logger.warning("region.debug_save_failed", extra={"name": name, "error": str(exc)})


def _colorise_labels(label_img: np.ndarray) -> np.ndarray:
    """Turn a connected-components label image into a colour debug image."""
    n = int(label_img.max()) + 1
    colours = np.random.default_rng(0).integers(80, 255, size=(n, 3), dtype=np.uint8)
    colours[0] = (0, 0, 0)  # background = black
    return colours[label_img]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class DuctRegion:
    """A candidate duct extracted as a connected region."""
    label: int
    x: int
    y: int
    w: int
    h: int
    area: int
    aspect_ratio: float   # long / short
    extent: float         # area / bounding-box area
    score: float          # composite confidence [0..1]
    duct_type: str = "unknown"
    center: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))

    def __post_init__(self) -> None:
        self.center = (self.x + self.w / 2, self.y + self.h / 2)

    @property
    def is_horizontal(self) -> bool:
        return self.w >= self.h

    @property
    def long_side(self) -> int:
        return max(self.w, self.h)

    @property
    def short_side(self) -> int:
        return min(self.w, self.h)

    def to_duct_segment(self):
        """
        Convert to a DuctSegment-compatible object so existing annotation
        and mapping code works without changes.
        """
        from app.pipeline.geometric import DuctSegment as DS, LineSegment as LS

        if self.is_horizontal:
            wall1 = LS(self.x, self.y,          self.x + self.w, self.y,          0.0,  float(self.w), (self.x + self.w / 2, float(self.y)))
            wall2 = LS(self.x, self.y + self.h, self.x + self.w, self.y + self.h, 0.0,  float(self.w), (self.x + self.w / 2, float(self.y + self.h)))
            angle = 0.0
        else:
            wall1 = LS(self.x,          self.y, self.x,          self.y + self.h, 90.0, float(self.h), (float(self.x),          self.y + self.h / 2))
            wall2 = LS(self.x + self.w, self.y, self.x + self.w, self.y + self.h, 90.0, float(self.h), (float(self.x + self.w),  self.y + self.h / 2))
            angle = 90.0

        return DS(
            wall1=wall1,
            wall2=wall2,
            width_px=float(self.short_side),
            length_px=float(self.long_side),
            angle=angle,
            center=self.center,
            duct_type=self.duct_type,
        )


@dataclass
class GraphNode:
    """A skeleton node (endpoint or junction)."""
    node_id: int
    x: int
    y: int
    kind: str  # 'endpoint' | 'junction' | 'segment'


@dataclass
class DuctGraph:
    """Topology graph of the duct network."""
    nodes: List[GraphNode] = field(default_factory=list)
    edges: List[Tuple[int, int]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage 1 — OCR Masking
# ---------------------------------------------------------------------------

def mask_text_regions(
    binary: np.ndarray,
    text_blocks: list,
    padding: int = 8,
) -> np.ndarray:
    """
    Erase text-stroke pixels from *binary* by painting each OCR bounding
    box black (background).  Returns a copy; never mutates the input.

    Args:
        binary:      White-on-black binary image.
        text_blocks: List of TextBlock with .x .y .width .height attributes.
        padding:     Extra pixels to expand each box (catches partial strokes).
    """
    if not text_blocks:
        return binary.copy()

    result = binary.copy()
    h_img, w_img = result.shape[:2]
    erased = 0
    for tb in text_blocks:
        x1 = max(0, tb.x - padding)
        y1 = max(0, tb.y - padding)
        x2 = min(w_img - 1, tb.x + tb.width + padding)
        y2 = min(h_img - 1, tb.y + tb.height + padding)
        cv2.rectangle(result, (x1, y1), (x2, y2), 0, -1)
        erased += 1

    logger.info("region.mask_text", extra={
        "boxes_erased": erased,
        "pixels_before": int(cv2.countNonZero(binary)),
        "pixels_after": int(cv2.countNonZero(result)),
    })
    _save("04a_text_masked", result)
    return result


# ---------------------------------------------------------------------------
# Stage 2a — Remove tiny / non-elongated CCs BEFORE bridging
# ---------------------------------------------------------------------------

def remove_small_components(
    masked: np.ndarray,
    min_area: int = 200,
    min_aspect: float = 3.0,
) -> np.ndarray:
    """Remove CCs that are too small or not elongated (text/symbols)."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        masked, connectivity=8, ltype=cv2.CV_32S
    )
    cleaned = np.zeros_like(masked)
    kept, removed = 0, 0
    for lbl in range(1, num_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        w = int(stats[lbl, cv2.CC_STAT_WIDTH])
        h = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        if area < min_area:
            removed += 1; continue
        long_s, short_s = max(w, h), min(w, h)
        if short_s == 0 or long_s / short_s < min_aspect:
            removed += 1; continue
        cleaned[labels == lbl] = 255
        kept += 1
    logger.info("region.remove_small_cc", extra={"kept": kept, "removed": removed})
    _save("04b1_cleaned_ccs", cleaned)
    return cleaned


# ---------------------------------------------------------------------------
# Stage 2b — Hough prefilter: keep only long-line pixels
# ---------------------------------------------------------------------------

def extract_long_lines(
    cleaned: np.ndarray,
    min_line_length: int = 200,
    hough_threshold: int = 30,
) -> np.ndarray:
    """Run HoughLinesP, return mask of long-line pixels only."""
    lines = cv2.HoughLinesP(
        cleaned, rho=1.0, theta=np.pi / 180.0,
        threshold=hough_threshold, minLineLength=min_line_length, maxLineGap=20,
    )
    line_mask = np.zeros_like(cleaned)
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0, :]:
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, thickness=3)
    logger.info("region.hough_prefilter", extra={"lines": len(lines) if lines is not None else 0})
    _save("04b2_line_mask", line_mask)
    return line_mask


# ---------------------------------------------------------------------------
# Stage 2c — Reconstruction-based closing (no spillover)
# ---------------------------------------------------------------------------

def _reconstruct(seed: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Geodesic reconstruction: dilate seed constrained by mask."""
    prev = seed.copy()
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    while True:
        curr = cv2.bitwise_and(cv2.dilate(prev, k3), mask)
        if np.array_equal(curr, prev):
            break
        prev = curr
    return curr


def _reconstruction_close(src: np.ndarray, kernel: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Close via reconstruction — fills gaps without bleeding."""
    result = src.copy()
    for _ in range(iterations):
        result = _reconstruct(result, cv2.dilate(result, kernel))
    return result


# ---------------------------------------------------------------------------
# Stage 2 — Morphological Region Bridging (revised)
# ---------------------------------------------------------------------------

def bridge_duct_regions(
    masked: np.ndarray,
    min_area: int = 200,
    min_aspect: float = 3.0,
    hough_min_len: int = 200,
    hough_thresh: int = 30,
    hough_enabled: bool = True,
    kernel_w: int = 80,
    kernel_h: int = 1,
    iterations: int = 3,
) -> np.ndarray:
    """
    Fill duct interiors by bridging parallel walls into solid regions.

    Revised flow (morphology AFTER structural filtering):
    1. Remove tiny / non-elongated CCs (symbols, text fragments)
    2. Hough prefilter — keep only long-line pixels
    3. Reconstruction-close horizontally → fills horizontal ducts
    4. Reconstruction-close vertically   → fills vertical ducts
    5. AND the two results (both orientations must agree)
    """
    cleaned = remove_small_components(masked, min_area=min_area, min_aspect=min_aspect)
    line_mask = extract_long_lines(cleaned, min_line_length=hough_min_len, hough_threshold=hough_thresh) if hough_enabled else cleaned

    k_horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_h, kernel_w))
    bridged_h = _reconstruction_close(line_mask, k_horiz, iterations=iterations)

    k_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
    bridged_v = _reconstruction_close(line_mask, k_vert, iterations=iterations)

    bridged = cv2.bitwise_and(bridged_h, bridged_v)

    logger.info("region.bridge", extra={
        "kernel_w": kernel_w, "iterations": iterations,
        "pixels_before": int(cv2.countNonZero(masked)),
        "pixels_after": int(cv2.countNonZero(bridged)),
    })
    _save("04b_bridged_regions", bridged)
    return bridged


# ---------------------------------------------------------------------------
# Stage 2d — Diffuser Detection (X-in-box symbols = duct endpoints)
# ---------------------------------------------------------------------------

def detect_diffusers(
    binary: np.ndarray,
    min_size: int = 20,
    max_size: int = 120,
    min_diag_fill: float = 0.3,
) -> List[Tuple[int, int, int, int]]:
    """
    Detect HVAC diffuser/grille symbols: square/rectangle with an X inside.
    Every duct connects to a diffuser — these are duct endpoints.

    Heuristic:
    1. Find contours with near-square aspect ratio and plausible size
    2. Sample both diagonals — X-in-box has high diagonal pixel density

    Returns list of (cx, cy, w, h) for each diffuser.
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    diffusers: List[Tuple[int, int, int, int]] = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if w < min_size or h < min_size or w > max_size or h > max_size:
            continue
        if area < min_size * min_size * 0.3:
            continue
        if max(w, h) / max(min(w, h), 1) > 2.0:
            continue

        roi = binary[y:y + h, x:x + w]
        if roi.size == 0:
            continue
        n = min(w, h)
        # diagonal TL→BR
        d1 = sum(1 for i in range(n) if roi[int(i*(h-1)/max(n-1,1)), int(i*(w-1)/max(n-1,1))] > 0) / n
        # diagonal TR→BL
        d2 = sum(1 for i in range(n) if roi[int(i*(h-1)/max(n-1,1)), int(w-1-i*(w-1)/max(n-1,1))] > 0) / n

        if d1 >= min_diag_fill and d2 >= min_diag_fill:
            diffusers.append((x + w // 2, y + h // 2, w, h))

    logger.info("region.diffusers", extra={"found": len(diffusers)})
    debug = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for cx, cy, w, h in diffusers:
        cv2.rectangle(debug, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), (0, 255, 0), 2)
    _save("04b3_diffusers", debug)
    return diffusers


# ---------------------------------------------------------------------------
# Stage 2e — Trace Ducts from Diffusers
# ---------------------------------------------------------------------------

def _ray_density(
    binary: np.ndarray, cx: int, cy: int,
    dx: int, dy: int, max_dist: int = 400, sample_width: int = 20,
) -> float:
    """Cast a ray and measure white-pixel density along it."""
    h, w = binary.shape[:2]
    hits, total = 0, 0
    perp_x, perp_y = -dy, dx
    for step in range(10, max_dist, 5):
        px, py = cx + step * dx, cy + step * dy
        if not (0 <= px < w and 0 <= py < h):
            break
        for off in range(-sample_width // 2, sample_width // 2 + 1, 3):
            sx, sy = int(px + off * perp_x), int(py + off * perp_y)
            if 0 <= sx < w and 0 <= sy < h:
                total += 1
                if binary[sy, sx] > 0:
                    hits += 1
    return hits / max(total, 1)


def trace_ducts_from_diffusers(
    binary: np.ndarray,
    diffusers: List[Tuple[int, int, int, int]],
    min_ray_density: float = 0.15,
    max_trace_dist: int = 500,
) -> List[DuctRegion]:
    """
    For each diffuser, cast rays in 4 cardinal directions.  Directions
    with high white-pixel density contain duct walls.  Trace along them
    to extract the full duct region.
    """
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    ducts: List[DuctRegion] = []
    h_img, w_img = binary.shape[:2]

    for cx, cy, dw, dh in diffusers:
        for dx, dy in directions:
            density = _ray_density(binary, cx, cy, dx, dy, max_dist=max_trace_dist)
            if density < min_ray_density:
                continue

            # Walk until density drops
            trace_len = 0
            for step in range(max(dw, dh), max_trace_dist, 10):
                px, py = cx + step * dx, cy + step * dy
                if not (0 <= px < w_img and 0 <= py < h_img):
                    break
                local = binary[max(0, py - 8):min(h_img, py + 8),
                               max(0, px - 8):min(w_img, px + 8)]
                if local.size == 0 or cv2.countNonZero(local) / local.size < 0.05:
                    break
                trace_len = step

            if trace_len < 50:
                continue

            # Build DuctRegion
            if dx != 0:
                x1, x2 = min(cx, cx + dx * trace_len), max(cx, cx + dx * trace_len)
                y1, y2 = cy - dw // 2, cy + dw // 2
            else:
                x1, x2 = cx - dw // 2, cx + dw // 2
                y1, y2 = min(cy, cy + dy * trace_len), max(cy, cy + dy * trace_len)

            w_duct, h_duct = x2 - x1, y2 - y1
            area = cv2.countNonZero(binary[y1:y2, x1:x2])
            if area < 500:
                continue

            ducts.append(DuctRegion(
                label=0, x=x1, y=y1, w=w_duct, h=h_duct,
                area=area,
                aspect_ratio=max(w_duct, h_duct) / max(min(w_duct, h_duct), 1),
                extent=area / max(w_duct * h_duct, 1),
                score=0.5,
            ))

    logger.info("region.trace_ducts", extra={"diffusers": len(diffusers), "ducts": len(ducts)})
    return ducts


# ---------------------------------------------------------------------------
# Stage 3 — Connected Component Extraction
# ---------------------------------------------------------------------------

def extract_duct_candidates(
    bridged: np.ndarray,
    min_area: int = 3000,
    min_long_side: int = 200,
    min_aspect: float = 2.5,
    border_margin: int = 15,
) -> Tuple[List[DuctRegion], np.ndarray]:
    """
    Run connectedComponentsWithStats and filter by geometric heuristics.

    Additionally excludes components whose bounding box touches the image
    border — these are drawing frames / title blocks, not ducts.

    Returns:
        candidates: List of DuctRegion objects passing all filters.
        label_img:  Full label image for debug visualisation.
    """
    num_labels, label_img, stats, _ = cv2.connectedComponentsWithStats(
        bridged, connectivity=8, ltype=cv2.CV_32S
    )
    h_img, w_img = bridged.shape[:2]

    candidates: List[DuctRegion] = []
    border_dropped = 0
    for lbl in range(1, num_labels):  # skip background (0)
        x = int(stats[lbl, cv2.CC_STAT_LEFT])
        y = int(stats[lbl, cv2.CC_STAT_TOP])
        w = int(stats[lbl, cv2.CC_STAT_WIDTH])
        h = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        area = int(stats[lbl, cv2.CC_STAT_AREA])

        # Border exclusion — ducts are never at the drawing edge
        if x <= border_margin or y <= border_margin or \
           x + w >= w_img - border_margin or y + h >= h_img - border_margin:
            border_dropped += 1
            continue

        if area < min_area:
            continue

        long_side = max(w, h)
        short_side = min(w, h)
        if long_side < min_long_side:
            continue
        if short_side == 0:
            continue

        aspect = long_side / short_side
        if aspect < min_aspect:
            continue

        bbox_area = w * h
        extent = area / bbox_area if bbox_area > 0 else 0.0

        candidates.append(DuctRegion(
            label=lbl,
            x=x, y=y, w=w, h=h,
            area=area,
            aspect_ratio=aspect,
            extent=extent,
            score=0.0,
        ))

    logger.info("region.cc_extract", extra={
        "total_components": num_labels - 1,
        "border_dropped": border_dropped,
        "after_filters": len(candidates),
        "min_area": min_area,
        "min_aspect": min_aspect,
    })

    # Debug image
    debug = _colorise_labels(label_img)
    for r in candidates:
        cv2.rectangle(debug, (r.x, r.y), (r.x + r.w, r.y + r.h), (0, 255, 0), 2)
    _save("04c_connected_components", debug)

    return candidates, label_img


# ---------------------------------------------------------------------------
# Stage 4 — Contour-Based Scoring & Validation
# ---------------------------------------------------------------------------

def score_component(region: DuctRegion) -> float:
    """
    Return a confidence score in [0, 1] for a DuctRegion.

    Scoring axes:
    - Elongation  : higher aspect ratio → higher score (capped at 15:1)
    - Rectangularity : how much of the bounding box is filled
    - Area bonus  : larger regions are more likely real ducts
    """
    # Elongation score [0..1] — aspect 2.5 → 0, aspect 15 → 1
    elong = min(1.0, (region.aspect_ratio - 2.5) / 12.5)

    # Rectangularity score [0..1] — extent 0.3 → 0, extent 1.0 → 1
    rect = min(1.0, max(0.0, (region.extent - 0.3) / 0.7))

    # Area score [0..1] — 3000 px → 0, 50000 px → 1
    area_score = min(1.0, (region.area - 3000) / 47000)

    return round(0.4 * elong + 0.35 * rect + 0.25 * area_score, 3)


def validate_duct_candidates(
    candidates: List[DuctRegion],
    label_img: np.ndarray,
    score_threshold: float = 0.4,
) -> List[DuctRegion]:
    """
    Score every candidate and keep those above *score_threshold*.

    Draws bounding boxes on a debug image:
      green  = accepted
      orange = rejected
    """
    accepted, rejected = [], []
    for r in candidates:
        r.score = score_component(r)
        (accepted if r.score >= score_threshold else rejected).append(r)

    logger.info("region.validate", extra={
        "before": len(candidates),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "threshold": score_threshold,
    })

    # Debug image
    h_img, w_img = label_img.shape[:2]
    debug = np.ones((h_img, w_img, 3), dtype=np.uint8) * 240
    for r in rejected:
        cv2.rectangle(debug, (r.x, r.y), (r.x + r.w, r.y + r.h), (0, 140, 255), 2)
        cv2.putText(debug, f"{r.score:.2f}", (r.x, r.y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 200), 1)
    for r in accepted:
        cv2.rectangle(debug, (r.x, r.y), (r.x + r.w, r.y + r.h), (0, 190, 0), 3)
        cv2.putText(debug, f"{r.score:.2f}", (r.x, r.y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 0), 1)
    _save("04d_validated_duct_regions", debug)

    return accepted


# ---------------------------------------------------------------------------
# Stage 4b — OCR Label Proximity Validation
# ---------------------------------------------------------------------------

DIM_RE = re.compile(
    r'(\d{1,3}\s*["\u2033\']?\s*[xX\u00d7]\s*\d{1,3})|'
    r'(\d{1,3}\s*["\u2033\']?\s*[\u00d8Dd])|'
    r'([\u00d8]\s*\d{1,3})',
    re.IGNORECASE,
)


def filter_by_label_proximity(
    regions: List[DuctRegion],
    text_blocks: list,
    max_distance: int = 200,
) -> List[DuctRegion]:
    """
    Keep only duct regions that have a dimension label within *max_distance*
    pixels of their centre.  Borders and room boundaries never have
    dimension labels, so this is the strongest single filter.
    """
    if not text_blocks:
        logger.warning("region.label_proximity.no_text_blocks")
        return regions

    dim_blocks = [tb for tb in text_blocks if DIM_RE.search(tb.text)]
    if not dim_blocks:
        logger.warning("region.label_proximity.no_dim_labels")
        return regions

    confirmed, discarded = [], []
    for r in regions:
        cx, cy = r.center
        found = any(
            ((tb.center[0] - cx) ** 2 + (tb.center[1] - cy) ** 2) ** 0.5 <= max_distance
            for tb in dim_blocks
        )
        (confirmed if found else discarded).append(r)

    # Debug image
    try:
        all_x = [int(r.center[0]) for r in regions] + [int(tb.center[0]) for tb in text_blocks]
        all_y = [int(r.center[1]) for r in regions] + [int(tb.center[1]) for tb in text_blocks]
        W = max(all_x, default=800) + 200
        H = max(all_y, default=600) + 200
        canvas = np.ones((H, W, 3), dtype=np.uint8) * 240
        for r in discarded:
            cv2.rectangle(canvas, (r.x, r.y), (r.x + r.w, r.y + r.h), (0, 140, 255), 2)
        for r in confirmed:
            cv2.rectangle(canvas, (r.x, r.y), (r.x + r.w, r.y + r.h), (0, 200, 0), 3)
        for tb in text_blocks:
            tx, ty = int(tb.center[0]), int(tb.center[1])
            is_dim = bool(DIM_RE.search(tb.text))
            color = (0, 0, 200) if is_dim else (180, 180, 180)
            cv2.circle(canvas, (tx, ty), 8 if is_dim else 3, color, -1)
            if is_dim and 0 <= tx < W and 0 <= ty < H:
                cv2.putText(canvas, tb.text[:14], (tx + 4, ty - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 160), 1)
        _save("04d_proximity_filter", canvas)
    except Exception:
        pass

    logger.info("region.label_proximity", extra={
        "before": len(regions),
        "confirmed": len(confirmed),
        "discarded": len(discarded),
        "dim_labels": len(dim_blocks),
    })
    return confirmed


# ---------------------------------------------------------------------------
# Stage 5 — Skeletonisation
# ---------------------------------------------------------------------------

def skeletonize_duct_regions(
    label_img: np.ndarray,
    regions: List[DuctRegion],
    enabled: bool = True,
) -> Optional[np.ndarray]:
    """
    Thin each accepted duct region to a single-pixel centreline.

    Uses skimage.morphology.skeletonize (Lee's algorithm).

    Returns:
        Skeleton binary image (uint8, same size as label_img), or None if disabled.
    """
    if not enabled:
        return None

    try:
        from skimage.morphology import skeletonize as sk_skeletonize
    except ImportError:
        logger.warning("region.skeletonize.skimage_missing")
        return None

    mask = np.zeros(label_img.shape[:2], dtype=np.uint8)
    for r in regions:
        if r.label > 0:
            mask[label_img == r.label] = 255
        else:
            # Diffuser-traced duct — use bounding box as mask
            mask[r.y:r.y + r.h, r.x:r.x + r.w] = 255

    # skimage expects bool
    skel_bool = sk_skeletonize(mask > 0)
    skeleton = (skel_bool * 255).astype(np.uint8)

    logger.info("region.skeletonize", extra={
        "skeleton_pixels": int(skeleton.sum() // 255),
        "regions": len(regions),
    })
    _save("04e_skeleton", skeleton)
    return skeleton


# ---------------------------------------------------------------------------
# Stage 6 — Graph Construction
# ---------------------------------------------------------------------------

def build_duct_graph(skeleton: Optional[np.ndarray]) -> DuctGraph:
    """
    Walk the skeleton and build a topology graph.

    Node classification:
      1 neighbour  → endpoint
      2 neighbours → interior segment pixel (not a node)
      3+ neighbours → junction

    Returns a DuctGraph with nodes and edges.
    """
    graph = DuctGraph()
    if skeleton is None or not cv2.countNonZero(skeleton):
        return graph

    skel = skeleton > 0
    h, w = skel.shape
    node_map: Dict[Tuple[int, int], int] = {}
    node_id = 0

    # 8-connectivity neighbourhood offsets
    offsets = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1),           (0, 1),
               (1, -1),  (1, 0),  (1, 1)]

    # Pass 1: mark endpoints and junctions
    ys, xs = np.where(skel)
    for y, x in zip(ys.tolist(), xs.tolist()):
        neighbours = sum(
            1 for dy, dx in offsets
            if 0 <= y + dy < h and 0 <= x + dx < w and skel[y + dy, x + dx]
        )
        if neighbours == 1 or neighbours >= 3:
            kind = "endpoint" if neighbours == 1 else "junction"
            node_map[(y, x)] = node_id
            graph.nodes.append(GraphNode(node_id=node_id, x=x, y=y, kind=kind))
            node_id += 1

    # Pass 2: trace edges between nodes (DFS walk)
    visited_edges: set = set()
    for start_node in graph.nodes:
        sy, sx = start_node.y, start_node.x
        # Walk each unvisited skeleton branch from this node
        for dy, dx in offsets:
            ny, nx = sy + dy, sx + dx
            if not (0 <= ny < h and 0 <= nx < w and skel[ny, nx]):
                continue
            if (ny, nx) in node_map:
                # Direct neighbour is already a node
                edge = tuple(sorted((start_node.node_id, node_map[(ny, nx)])))
                if edge not in visited_edges:
                    visited_edges.add(edge)
                    graph.edges.append(edge)
                continue
            # Walk until we hit another node or dead end
            path = [(sy, sx), (ny, nx)]
            while True:
                cy, cx = path[-1]
                nexts = [
                    (cy + ddy, cx + ddx)
                    for ddy, ddx in offsets
                    if 0 <= cy + ddy < h and 0 <= cx + ddx < w
                    and skel[cy + ddy, cx + ddx]
                    and (cy + ddy, cx + ddx) != path[-2]
                ]
                if not nexts:
                    break
                nxt = nexts[0]
                if nxt in node_map:
                    edge = tuple(sorted((start_node.node_id, node_map[nxt])))
                    if edge not in visited_edges:
                        visited_edges.add(edge)
                        graph.edges.append(edge)
                    break
                path.append(nxt)

    logger.info("region.graph", extra={
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "endpoints": sum(1 for n in graph.nodes if n.kind == "endpoint"),
        "junctions": sum(1 for n in graph.nodes if n.kind == "junction"),
    })
    return graph


# ---------------------------------------------------------------------------
# Orchestration façade
# ---------------------------------------------------------------------------

class RegionBasedDetectionService:
    """
    Drop-in replacement for GeometricDetectionService that uses the
    region-based pipeline internally.

    The public interface mirrors GeometricDetectionService so the
    orchestrator needs minimal changes.
    """

    def __init__(self) -> None:
        self.text_pad = settings.RB_TEXT_MASK_PADDING
        self.pre_clean_area = settings.RB_PRE_CLEAN_MIN_AREA
        self.pre_clean_asp = settings.RB_PRE_CLEAN_MIN_ASPECT
        self.hough_enabled = settings.RB_HOUGH_PREFILTER_ENABLED
        self.hough_min_len = settings.RB_HOUGH_PREFILTER_MIN_LEN
        self.hough_thresh = settings.RB_HOUGH_PREFILTER_THRESH
        self.bridge_kw = settings.RB_BRIDGE_KERNEL_W
        self.bridge_kh = settings.RB_BRIDGE_KERNEL_H
        self.bridge_iter = settings.RB_BRIDGE_ITERATIONS
        self.cc_min_area = settings.RB_CC_MIN_AREA
        self.cc_min_long = settings.RB_CC_MIN_LONG_SIDE
        self.cc_min_asp = settings.RB_CC_MIN_ASPECT
        self.score_thresh = settings.RB_SCORE_THRESHOLD
        self.skeleton_enabled = settings.RB_SKELETON_ENABLED

    # ------------------------------------------------------------------
    # Colour classification (same API as GeometricDetectionService)
    # ------------------------------------------------------------------

    def _sample_color_mask(
        self,
        color_mask: np.ndarray,
        center_x: float,
        center_y: float,
        angle: float,
        width: float,
        num_samples: int = 10,
    ) -> float:
        h, w = color_mask.shape[:2]
        angle_rad = np.radians(angle)
        dx, dy = np.cos(angle_rad), np.sin(angle_rad)
        half_len = width * 2
        hits = 0
        for i in range(num_samples):
            t = (i / (num_samples - 1) - 0.5) * 2 * half_len
            px, py = int(center_x + t * dx), int(center_y + t * dy)
            if 0 <= px < w and 0 <= py < h and color_mask[py, px] > 0:
                hits += 1
        return hits / num_samples

    def classify_duct_color(
        self,
        duct: "DuctRegion",
        blue_mask: np.ndarray,
        red_mask: np.ndarray,
    ) -> str:
        cx, cy = duct.center
        angle = 0.0 if duct.is_horizontal else 90.0
        blue = self._sample_color_mask(blue_mask, cx, cy, angle, duct.short_side)
        red = self._sample_color_mask(red_mask, cx, cy, angle, duct.short_side)
        thr = 0.2
        if blue > thr and blue > red:
            return "supply"
        if red > thr and red > blue:
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
        text_blocks: Optional[list] = None,
    ) -> List[DuctRegion]:
        """
        Simple rectangle-only duct detection.
        1. Mask text
        2. Find contours → keep only rectangular ones
        3. Filter by aspect ratio + min area
        4. OCR proximity validation
        """
        logger.info("region.detect.started", extra={"shape": binary_image.shape})

        # Stage 1 — OCR masking
        masked = mask_text_regions(
            binary_image,
            text_blocks or [],
            padding=self.text_pad,
        )

        # Stage 2 — Keep only rectangular contours
        contours, _ = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ducts: List[DuctRegion] = []
        h_img, w_img = masked.shape[:2]

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cnt_area = cv2.contourArea(cnt)
            bbox_area = w * h
            if bbox_area == 0:
                continue

            rectangularity = cnt_area / bbox_area

            # Must be very rectangular (straight lines = near 1.0)
            if rectangularity < 0.7:
                continue

            # Border exclusion
            if x <= 15 or y <= 15 or x + w >= w_img - 15 or y + h >= h_img - 15:
                continue

            # Size + aspect filters
            long_s, short_s = max(w, h), min(w, h)
            if cnt_area < self.cc_min_area:
                continue
            if long_s < self.cc_min_long:
                continue
            if short_s == 0 or long_s / short_s < self.cc_min_asp:
                continue

            ducts.append(DuctRegion(
                label=0, x=x, y=y, w=w, h=h,
                area=int(cnt_area),
                aspect_ratio=long_s / short_s,
                extent=rectangularity,
                score=rectangularity,
            ))

        logger.info("region.rect_filter", extra={"contours": len(contours), "kept": len(ducts)})

        # Debug
        debug = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)
        for d in ducts:
            cv2.rectangle(debug, (d.x, d.y), (d.x + d.w, d.y + d.h), (0, 255, 0), 2)
        _save("04b_rectangles_only", debug)

        # Stage 3 — OCR label proximity
        validated = filter_by_label_proximity(
            ducts,
            text_blocks or [],
            max_distance=settings.SPATIAL_MAX_DISTANCE_PX,
        )

        # Stage 4 — Skeletonisation
        label_img = np.zeros_like(masked, dtype=np.int32)
        for i, d in enumerate(validated, 1):
            label_img[d.y:d.y + d.h, d.x:d.x + d.w] = i
            d.label = i
        skeleton = skeletonize_duct_regions(
            label_img, validated, enabled=self.skeleton_enabled,
        )

        # Stage 5 — Graph
        graph = build_duct_graph(skeleton) if skeleton is not None else DuctGraph([], [])

        logger.info("region.detect.completed", extra={
            "ducts": len(validated),
            "graph_nodes": len(graph.nodes),
            "graph_edges": len(graph.edges),
        })
        print(f"[REGION DETECT] ducts={len(validated)}, "
              f"nodes={len(graph.nodes)}, edges={len(graph.edges)}")
        return validated
