"""
Vector extraction from PDF using PyMuPDF.

Instead of rendering the PDF to pixels and recovering geometry via Hough /
morphology, we read the PDF's drawing operators directly.  This gives us:

  - Exact float coordinates for every line, rectangle, and Bezier curve
  - Line stroke widths and colors
  - Text spans with content and positioned bounding boxes
  - No antialiasing, no quantisation, no pixel ambiguity

All coordinates returned by this module are in **PDF points** (page-space).
Use ``pts_to_px_scale()`` if you need to project onto a raster render.
"""
from __future__ import annotations

import io
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import fitz  # pymupdf

logger = logging.getLogger("hvac_analyzer.pipeline")

DEBUG_DIR = Path(__file__).resolve().parents[2] / "debug_images"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class VectorLine:
    """A single straight-line primitive from the PDF."""
    x1: float
    y1: float
    x2: float
    y2: float
    width: float = 0.0
    color: Optional[Tuple[float, float, float]] = None

    @property
    def length(self) -> float:
        return math.hypot(self.x2 - self.x1, self.y2 - self.y1)

    @property
    def angle_deg(self) -> float:
        """Angle in degrees, normalised to [0, 180)."""
        a = math.degrees(math.atan2(self.y2 - self.y1, self.x2 - self.x1))
        a %= 180.0
        return a

    @property
    def midpoint(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


@dataclass
class VectorRect:
    x: float
    y: float
    w: float
    h: float


@dataclass
class VectorText:
    text: str
    x: float
    y: float
    width: float
    height: float
    font_size: float = 0.0

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)


@dataclass
class VectorPage:
    """Everything extracted from one PDF page."""
    width: float
    height: float
    lines: List[VectorLine] = field(default_factory=list)
    rects: List[VectorRect] = field(default_factory=list)
    texts: List[VectorText] = field(default_factory=list)
    page: Optional[fitz.Page] = None  # keep handle for rendering

    @property
    def primitive_count(self) -> int:
        return len(self.lines) + len(self.rects)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

class VectorExtractionService:
    """Pulls clean vector primitives out of a PDF document."""

    def open(self, pdf_bytes: bytes, page_index: int = 0) -> VectorPage:
        """Open PDF and extract primitives from a single page."""
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if page_index >= len(doc):
            raise ValueError(f"PDF only has {len(doc)} pages")
        page = doc[page_index]

        lines = self._extract_lines(page)
        rects = self._extract_rects(page)
        texts = self._extract_texts(page)

        logger.info("vector.extracted", extra={
            "page_width": float(page.rect.width),
            "page_height": float(page.rect.height),
            "lines": len(lines),
            "rects": len(rects),
            "texts": len(texts),
        })

        return VectorPage(
            width=float(page.rect.width),
            height=float(page.rect.height),
            lines=lines,
            rects=rects,
            texts=texts,
            page=page,
        )

    # ------------------------------------------------------------------
    # Primitive extraction
    # ------------------------------------------------------------------

    def _extract_lines(self, page: fitz.Page) -> List[VectorLine]:
        """
        Walk every drawing operator on the page and flatten to line segments.

        PyMuPDF's get_drawings() returns dicts whose ``items`` are
        (op, *coords) tuples.  Relevant ops:

            'l'  -> line segment:   ('l', Point, Point)
            're' -> rectangle:      ('re', Rect, ...)   (expand to 4 lines)
            'qu' -> quad:           skipped (used for fills/hatch)
            'c'  -> bezier curve:   skipped (rare for ducts)
        """
        lines: List[VectorLine] = []
        for d in page.get_drawings():
            width = float(d.get("width") or 0.0)
            color = d.get("color") or d.get("stroke")
            for item in d.get("items", []):
                op = item[0]
                if op == "l":
                    p1, p2 = item[1], item[2]
                    lines.append(VectorLine(
                        x1=float(p1.x), y1=float(p1.y),
                        x2=float(p2.x), y2=float(p2.y),
                        width=width, color=color,
                    ))
                elif op == "re":
                    r = item[1]
                    x0, y0 = float(r.x0), float(r.y0)
                    x1, y1 = float(r.x1), float(r.y1)
                    # 4 edges
                    lines.append(VectorLine(x0, y0, x1, y0, width, color))
                    lines.append(VectorLine(x1, y0, x1, y1, width, color))
                    lines.append(VectorLine(x1, y1, x0, y1, width, color))
                    lines.append(VectorLine(x0, y1, x0, y0, width, color))
        return lines

    def _extract_rects(self, page: fitz.Page) -> List[VectorRect]:
        """Pull just the re operators as rectangles."""
        rects: List[VectorRect] = []
        for d in page.get_drawings():
            for item in d.get("items", []):
                if item[0] == "re":
                    r = item[1]
                    rects.append(VectorRect(
                        x=float(r.x0), y=float(r.y0),
                        w=float(r.x1 - r.x0), h=float(r.y1 - r.y0),
                    ))
        return rects

    def _extract_texts(self, page: fitz.Page) -> List[VectorText]:
        """
        Get every text span on the page with content and bbox.
        Uses page.get_text('dict') which returns blocks → lines → spans.
        """
        texts: List[VectorText] = []
        data = page.get_text("dict")
        for block in data.get("blocks", []):
            if block.get("type") != 0:  # 0 = text block
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    content = (span.get("text") or "").strip()
                    if not content:
                        continue
                    bbox = span.get("bbox") or (0, 0, 0, 0)
                    x0, y0, x1, y1 = bbox
                    texts.append(VectorText(
                        text=content,
                        x=float(x0), y=float(y0),
                        width=float(x1 - x0),
                        height=float(y1 - y0),
                        font_size=float(span.get("size") or 0.0),
                    ))
        return texts

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def render_page_to_bgr(
        self,
        page: fitz.Page,
        dpi: int = 400,
    ) -> Tuple["np.ndarray", float]:
        """
        Render the PDF page to a BGR ndarray.

        Returns:
            (bgr_image, pts_to_px_scale)  where scale = dpi / 72.
        """
        import numpy as np
        import cv2

        scale = dpi / 72.0
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 3:
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif pix.n == 4:
            bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return bgr, scale


# ---------------------------------------------------------------------------
# Debug rendering (optional)
# ---------------------------------------------------------------------------

def debug_draw_lines(
    page: VectorPage,
    scale: float,
    out_size: Tuple[int, int],
    filename: str,
    color_by_angle: bool = True,
) -> None:
    """
    Save a debug PNG showing every extracted line, scaled to pixel space.
    """
    try:
        import numpy as np
        import cv2
        DEBUG_DIR.mkdir(exist_ok=True)

        h, w = out_size
        canvas = np.full((h, w, 3), 255, dtype=np.uint8)

        for ln in page.lines:
            x1, y1 = int(ln.x1 * scale), int(ln.y1 * scale)
            x2, y2 = int(ln.x2 * scale), int(ln.y2 * scale)
            if color_by_angle:
                a = ln.angle_deg
                if a < 5 or a > 175:
                    color = (0, 0, 200)      # horizontal — red
                elif 85 < a < 95:
                    color = (200, 0, 0)      # vertical — blue
                else:
                    color = (120, 120, 120)  # angled — grey
            else:
                color = (60, 60, 60)
            cv2.line(canvas, (x1, y1), (x2, y2), color, 1)

        cv2.imwrite(str(DEBUG_DIR / filename), canvas)
        logger.info("vector.debug_saved", extra={"file": filename, "lines": len(page.lines)})
    except Exception as e:
        logger.warning("vector.debug_failed", extra={"err": str(e)})
