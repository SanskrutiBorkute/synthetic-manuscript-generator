"""
layout_engine.py
================
Computes spatial layout for synthetic manuscript folios:
- Page margins (top, bottom, left, right)
- Decorative manuscript borders (rubricated red borders, dual rules)
- Text block placement, line heights, and baseline positions
- Word wrapping and column alignments typical of Indic manuscripts
"""

from dataclasses import dataclass, field
import random
from typing import List, Tuple, Optional, Dict, Any


@dataclass
class LineLayout:
    """Represents a computed line of text and its intended baseline position."""
    text: str
    line_index: int
    x: int
    y: int
    max_width: int
    height: int
    block_type: str = "main"  # 'main', 'secondary', 'marginal'


@dataclass
class PageLayout:
    """Full geometric layout description for a manuscript folio."""
    canvas_size: Tuple[int, int]
    content_box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    border_box: Optional[Tuple[int, int, int, int]] = None
    lines: List[LineLayout] = field(default_factory=list)
    has_border: bool = True
    border_color: str = "#8B0000"
    layout_type: str = "standard"  # 'standard', 'multi_block', 'marginal_commentary', 'pothi_leaf'
    has_ruling: bool = True
    ruling_lines: List[int] = field(default_factory=list)
    divider_y: Optional[int] = None           # For multi_block
    margin_divider_x: Optional[int] = None    # For marginal_commentary


class LayoutEngine:
    """Calculates coordinates and margins for laying out text on a manuscript canvas."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, seed: Optional[int] = None):
        """
        Args:
            config: Optional configuration dictionary.
            seed: Optional seed for layout randomization.
        """
        self.config = config or {}
        self.rng = random.Random(seed)
        layout_cfg = self.config.get("layout", {})
        self.line_spacing_ratio = layout_cfg.get("line_spacing", 1.38)
        self.draw_borders = layout_cfg.get("draw_borders", True)

    def compute_layout(
        self,
        canvas_size: Tuple[int, int],
        text_lines: List[str],
        font_size: int = 30,
        layout_type: Optional[str] = None,
        secondary_lines: Optional[List[str]] = None,
        is_palm_leaf: bool = False,
        script: str = "devanagari",
    ) -> PageLayout:
        """
        Computes line placements that naturally fill the manuscript writing region.

        Args:
            canvas_size: (width, height) of the canvas.
            text_lines: Raw strings from source file for main text block.
            font_size: Nominal font size in points/pixels.
            layout_type: 'standard', 'multi_block', or 'marginal_commentary'.
            secondary_lines: Optional additional lines from SAME source for secondary/marginal blocks.
            is_palm_leaf: Whether canvas is an elongated palm leaf (Style B).
            script: Target script ('devanagari', 'modi', 'sharada') for width calibration.

        Returns:
            PageLayout containing line coordinates, bounding regions, and ruling lines.
        """
        width, height = canvas_size

        # 1. Determine writing region bounds
        if is_palm_leaf or (width / max(height, 1) > 2.2):
            # Style B: Palm leaf layout
            # Cord hole is at x ~ 144, so left margin starts safely at 230
            c_x1 = 230
            c_x2 = width - 120
            c_y1 = 65
            c_y2 = height - 65
            layout_type = "standard"  # Palm leaf traditionally uses single elongated band
        else:
            # Paper / Parchment layout
            c_x1 = self.rng.randint(115, 140)
            c_x2 = width - self.rng.randint(125, 155)
            c_y1 = self.rng.randint(85, 105)
            c_y2 = height - self.rng.randint(85, 105)

        content_box = (c_x1, c_y1, c_x2, c_y2)
        avail_w = c_x2 - c_x1
        avail_h = c_y2 - c_y1

        # Border box closely framing the writing region with authentic manuscript margin
        has_border = self.draw_borders and (self.rng.random() < 0.70) and not is_palm_leaf
        border_box = None
        if has_border:
            border_box = (c_x1 - 25, c_y1 - 20, c_x2 + 25, c_y2 + 20)

        if layout_type is None:
            if is_palm_leaf:
                layout_type = "standard"
            else:
                layout_type = self.rng.choices(
                    ["standard", "multi_block", "marginal_commentary"],
                    weights=[0.50, 0.25, 0.25]
                )[0]

        computed_lines: List[LineLayout] = []
        ruling_lines: List[int] = []
        line_step = int(font_size * self.line_spacing_ratio)
        divider_y = None
        margin_divider_x = None

        # Build continuous word stream from raw input lines (preserving 100% exact text)
        main_words = self._lines_to_words(text_lines)
        sec_words = self._lines_to_words(secondary_lines) if secondary_lines else []

        # ----------------------------------------------------
        # LAYOUT 1: MULTI_BLOCK (Main verse + Secondary commentary)
        # ----------------------------------------------------
        if layout_type == "multi_block" and not is_palm_leaf and len(main_words) >= 40:
            # Main block occupies top ~48% of height
            main_max_y = c_y1 + int(avail_h * 0.46)
            divider_y = c_y1 + int(avail_h * 0.50)
            sec_start_y = divider_y + int(line_step * 0.8)

            # 1. Main Block lines
            wrapped_main = self._wrap_words(main_words, avail_w, font_size, script=script)
            curr_y = c_y1
            idx = 0
            for lw in wrapped_main:
                if curr_y + font_size > main_max_y:
                    break
                computed_lines.append(
                    LineLayout(
                        text=lw,
                        line_index=idx,
                        x=c_x1,
                        y=curr_y,
                        max_width=avail_w,
                        height=line_step,
                        block_type="main",
                    )
                )
                ruling_lines.append(curr_y + int(font_size * 0.9))
                curr_y += line_step
                idx += 1

            # 2. Secondary Commentary Block lines (slightly smaller script)
            sec_font_size = max(22, int(font_size * 0.84))
            sec_step = int(sec_font_size * 1.34)
            wrapped_sec = self._wrap_words(
                sec_words if sec_words else main_words[idx * 15:],
                avail_w - 40,
                sec_font_size,
                script=script,
            )
            curr_y = sec_start_y
            for lw in wrapped_sec:
                if curr_y + sec_font_size > c_y2:
                    break
                computed_lines.append(
                    LineLayout(
                        text=lw,
                        line_index=idx,
                        x=c_x1 + 20,
                        y=curr_y,
                        max_width=avail_w - 40,
                        height=sec_step,
                        block_type="secondary",
                    )
                )
                ruling_lines.append(curr_y + int(sec_font_size * 0.9))
                curr_y += sec_step
                idx += 1

        # ----------------------------------------------------
        # LAYOUT 2: MARGINAL_COMMENTARY (Central main text + Right margin gloss)
        # ----------------------------------------------------
        elif layout_type == "marginal_commentary" and not is_palm_leaf and len(main_words) >= 40:
            main_w = int(avail_w * 0.67)
            margin_divider_x = c_x1 + main_w + 20
            side_x = margin_divider_x + 22
            side_w = c_x2 - side_x

            # 1. Main Central Block
            wrapped_main = self._wrap_words(main_words, main_w, font_size, script=script)
            curr_y = c_y1
            idx = 0
            for lw in wrapped_main:
                if curr_y + font_size > c_y2:
                    break
                computed_lines.append(
                    LineLayout(
                        text=lw,
                        line_index=idx,
                        x=c_x1,
                        y=curr_y,
                        max_width=main_w,
                        height=line_step,
                        block_type="main",
                    )
                )
                ruling_lines.append(curr_y + int(font_size * 0.9))
                curr_y += line_step
                idx += 1

            # 2. Side Marginal Commentary (smaller script running top to bottom)
            marg_font_size = max(21, int(font_size * 0.74))
            marg_step = int(marg_font_size * 1.34)
            wrapped_marg = self._wrap_words(
                sec_words if sec_words else main_words[idx * 15:],
                side_w,
                marg_font_size,
                script=script,
            )
            marg_y = c_y1 + 5
            for lw in wrapped_marg:
                if marg_y + marg_font_size > c_y2:
                    break
                computed_lines.append(
                    LineLayout(
                        text=lw,
                        line_index=idx,
                        x=side_x,
                        y=marg_y,
                        max_width=side_w,
                        height=marg_step,
                        block_type="marginal",
                    )
                )
                marg_y += marg_step
                idx += 1

        # ----------------------------------------------------
        # LAYOUT 3: STANDARD (Classical continuous central manuscript folio)
        # ----------------------------------------------------
        else:
            layout_type = "standard"
            wrapped_lines = self._wrap_words(main_words, avail_w, font_size, script=script)
            curr_y = c_y1
            idx = 0
            for line_text in wrapped_lines:
                if curr_y + font_size > c_y2:
                    break
                computed_lines.append(
                    LineLayout(
                        text=line_text,
                        line_index=idx,
                        x=c_x1,
                        y=curr_y,
                        max_width=avail_w,
                        height=line_step,
                        block_type="main",
                    )
                )
                ruling_lines.append(curr_y + int(font_size * 0.9))
                curr_y += line_step
                idx += 1

        return PageLayout(
            canvas_size=canvas_size,
            content_box=content_box,
            border_box=border_box,
            lines=computed_lines,
            has_border=has_border,
            border_color="#8B0000" if (self.rng.random() < 0.65) else "#2B1D0C",
            layout_type=layout_type,
            has_ruling=True,
            ruling_lines=ruling_lines,
            divider_y=divider_y,
            margin_divider_x=margin_divider_x,
        )

    @staticmethod
    def _lines_to_words(lines: List[str]) -> List[str]:
        """Flattens a list of raw text lines into ordered non-empty words."""
        words: List[str] = []
        for line in lines:
            line_str = line.strip()
            if line_str:
                for w in line_str.split():
                    if w:
                        words.append(w)
        return words

    @staticmethod
    def _wrap_words(words: List[str], max_width: int, font_size: int, script: str = "devanagari") -> List[str]:
        """
        Wraps a continuous stream of words into lines fitting max_width.
        Ensures lines span naturally across the column width.
        """
        lines: List[str] = []
        current: List[str] = []

        # Script-calibrated glyph advance ratios (accounting for matras & combining marks)
        script_ratios = {
            "devanagari": 0.40,
            "modi": 0.42,
            "sharada": 0.42,
        }
        char_w = font_size * script_ratios.get(script.lower(), 0.40)

        for word in words:
            candidate = " ".join(current + [word])
            est_w = len(candidate) * char_w
            if est_w <= max_width or not current:
                current.append(word)
            else:
                lines.append(" ".join(current))
                current = [word]

        if current:
            lines.append(" ".join(current))

        return lines

