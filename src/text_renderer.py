"""
text_renderer.py
================
Renders historical Indic manuscript text (Devanagari, Modi, Sharada)
with authentic handwritten calligraphy and physical ink-substrate dynamics:

1. Authentic Handwriting Fonts:
   - Uses genuine calligraphic/handwriting fonts (e.g. Kalam reed-pen font for Devanagari)
   - Preserves 100% correct Unicode script shaping, conjuncts, and vowel signs.

2. Line-Level & Word-Level Organic Variation:
   - Non-linear baseline undulation (sine wave + slope drift)
   - 2D smooth elastic displacement field (simulating finger/wrist movement)
   - Organic left-margin offsets and variable word spacing
   - Per-word micro-rotations and subtle vertical displacement

3. Calligraphic Reed-Pen (Kalam / Borū) Dynamics:
   - 45-degree angled flat reed-nib stroke profile (thick downward/diagonal, thin horizontal)
   - Smooth 2D pen pressure field modulating stroke thickness across each line

4. Realistic Ink & Paper Interaction:
   - Pen dipping cycles (ink darkness varying from fresh saturation to faded sepia)
   - Substrate grain interaction (ink settles into paper/parchment valleys and fibers)
   - Capillary edge bleeding (fiber feathering)
   - Subtle incidental scribe smudges near text without obscuring letters

5. Exact Ground-Truth Synchronization:
   - Bounding boxes extracted directly from final non-zero ink pixels of each line.
"""

import math
import os
import random
import unicodedata
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from .layout_engine import PageLayout, LineLayout


class TextRenderer:
    """Renders authentic handwritten manuscript calligraphy with synchronized ground-truth annotations."""

    # Historical ink palettes: (dense_soot, diluted_tone)
    INK_PALETTES = {
        "devanagari": {
            "dense": np.array([18, 14, 11], dtype=np.float32),      # Deep carbon soot black
            "diluted": np.array([72, 54, 40], dtype=np.float32),    # Warm aged sepia-charcoal
        },
        "modi": {
            "dense": np.array([32, 20, 10], dtype=np.float32),      # Dark iron-gall brown
            "diluted": np.array([88, 64, 44], dtype=np.float32),    # Faded bistre / walnut ink
        },
        "sharada": {
            "dense": np.array([16, 14, 12], dtype=np.float32),      # Deep Kashmiri lampblack
            "diluted": np.array([64, 50, 38], dtype=np.float32),    # Weathered birch-bark soot tone
        },
    }

    # Viramas (halants) and joiners across Indic scripts for lossless grapheme cluster segmentation
    VIRAMAS = {'\u094D', '\U0001163F', '\U000111C0'}
    JOINERS = {'\u200C', '\u200D'}

    def __init__(self, fonts_config: Optional[Dict[str, List[str]]] = None, seed: Optional[int] = None):
        """
        Args:
            fonts_config: Mapping from script names to font file paths.
            seed: Optional seed for reproducible calligraphic effects.
        """
        self.fonts_config = fonts_config or {}
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self._font_cache: Dict[Tuple[str, int], ImageFont.FreeTypeFont] = {}

    @classmethod
    def get_grapheme_clusters(cls, word: str) -> List[str]:
        """
        Segments an Indic word into orthographic syllables / grapheme clusters.
        Guarantees that:
        - Base consonants and attached matras (vowel signs) are NEVER split
        - Consonant conjuncts (joined by halant / virama) are NEVER split
        - Combining marks, nuktas, anusvaras, and visargas remain intact
        """
        clusters: List[str] = []
        curr: List[str] = []
        i = 0
        n = len(word)
        while i < n:
            curr.append(word[i])
            i += 1
            while i < n and (word[i] in cls.JOINERS or unicodedata.category(word[i]) in ('Mn', 'Mc')):
                mark = word[i]
                curr.append(mark)
                i += 1
                # If the combining mark is a virama/halant, the conjunct continues into the following consonant
                if mark in cls.VIRAMAS and i < n:
                    curr.append(word[i])
                    i += 1
            clusters.append(''.join(curr))
            curr = []
        if curr:
            clusters.append(''.join(curr))
        return clusters

    def get_font(self, script_name: str, font_size: int = 32) -> Tuple[ImageFont.ImageFont, str]:
        """
        Retrieves a suitable calligraphic or manuscript font for the specified script.
        Ensures a single scribe font is used for the entire folio.

        Args:
            script_name: Script identifier ('devanagari', 'modi', 'sharada').
            font_size: Target size in points.

        Returns:
            Tuple of (PIL ImageFont instance, font_file_path).
        """
        font_paths = self.fonts_config.get(script_name.lower(), [])
        valid_paths = [p for p in font_paths if os.path.exists(p)]

        if not valid_paths:
            return ImageFont.load_default(), "default"

        # Select one scribe font per page (e.g. JainiPurva or Tillana for Devanagari)
        selected_path = self.rng.choice(valid_paths)

        cache_key = (selected_path, font_size)
        if cache_key in self._font_cache:
            return self._font_cache[cache_key], selected_path

        try:
            font = ImageFont.truetype(selected_path, font_size)
            self._font_cache[cache_key] = font
            return font, selected_path
        except Exception:
            return ImageFont.load_default(), "default"

    def render_page(
        self,
        base_image: Image.Image,
        page_layout: PageLayout,
        script_name: str,
        font_size: int = 32,
        ink_color: Optional[Tuple[int, int, int]] = None,
    ) -> Tuple[Image.Image, List[Dict[str, Any]], str]:
        """
        Renders handwritten manuscript text onto the canvas with authentic calligraphic ink dynamics.

        Args:
            base_image: Background PIL Image (parchment, aged paper, palm leaf).
            page_layout: Computed layout structure with line targets.
            script_name: Script identifier ('devanagari', 'modi', 'sharada').
            font_size: Base font size.
            ink_color: Optional base RGB tuple override.

        Returns:
            Tuple of:
            - Final rendered PIL Image with calligraphic ink and authentic aging
            - List of synchronized line annotations with exact bounding boxes
            - Font identifier used
        """
        w, h = base_image.size
        img = base_image.copy()
        draw = ImageDraw.Draw(img)
        font, font_used = self.get_font(script_name, font_size)
        script_key = script_name.lower()

        content_box = page_layout.content_box
        c_x1, c_y1, c_x2, c_y2 = content_box

        # 1. Render decorative manuscript borders if configured
        if page_layout.has_border and page_layout.border_box:
            bx1, by1, bx2, by2 = page_layout.border_box
            border_rgb = (139, 0, 0) if page_layout.border_color == "#8B0000" else (45, 30, 20)
            draw.rectangle([bx1, by1, bx2, by2], outline=border_rgb, width=2)
            draw.rectangle([bx1 + 4, by1 + 4, bx2 - 4, by2 - 4], outline=(55, 42, 28), width=1)

        # 2. Render horizontal divider for multi-block layouts
        if getattr(page_layout, "divider_y", None) is not None:
            dy = page_layout.divider_y
            draw.line([(c_x1, dy - 2), (c_x2, dy - 2)], fill=(139, 0, 0), width=2)
            draw.line([(c_x1 + 10, dy + 2), (c_x2 - 10, dy + 2)], fill=(55, 42, 28), width=1)

        # 3. Render vertical margin divider for marginal commentary layouts
        if getattr(page_layout, "margin_divider_x", None) is not None:
            mx = page_layout.margin_divider_x
            draw.line([(mx, c_y1), (mx, c_y2)], fill=(139, 0, 0), width=2)
            draw.line([(mx + 3, c_y1 + 10), (mx + 3, c_y2 - 10)], fill=(55, 42, 28), width=1)

        # 4. Render faint manuscript ruling guides (lead-point / dry-point lines)
        if getattr(page_layout, "has_ruling", True) and getattr(page_layout, "ruling_lines", None):
            rule_overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            r_draw = ImageDraw.Draw(rule_overlay)
            for ry in page_layout.ruling_lines:
                if 0 <= ry < h:
                    r_alpha = self.rng.randint(24, 45)
                    r_col = (130, 110, 90, r_alpha)
                    r_draw.line([(c_x1 - 15, ry), (c_x2 + 15, ry)], fill=r_col, width=1)
            rule_overlay = rule_overlay.filter(ImageFilter.GaussianBlur(radius=0.6))
            img = Image.alpha_composite(img.convert("RGBA"), rule_overlay).convert("RGB")

        # 3. Extract underlying substrate lightness texture for physical paper/ink interaction
        bg_np = np.array(base_image)
        bg_gray = cv2.cvtColor(bg_np, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        # Get space width for variable inter-word spacing
        try:
            space_bbox = font.getbbox(" ")
            base_space_w = max(8, space_bbox[2] - space_bbox[0])
        except Exception:
            base_space_w = int(font_size * 0.3)

        # 4. Generate smooth 2D elastic displacement field for organic handwriting waviness & rhythm
        sigma_warp = 40.0
        alpha_y = 2.6  # Vertical muscular drift amplitude (natural, readable, ~2.6px)
        alpha_x = 1.4  # Horizontal character rhythm variation (px)

        noise_x = self.np_rng.normal(0, 1, (h, w)).astype(np.float32)
        noise_y = self.np_rng.normal(0, 1, (h, w)).astype(np.float32)
        dx_field = cv2.GaussianBlur(noise_x, (0, 0), sigma_warp)
        dy_field = cv2.GaussianBlur(noise_y, (0, 0), sigma_warp)
        dx_field = (dx_field / (np.max(np.abs(dx_field)) + 1e-5)) * alpha_x
        dy_field = (dy_field / (np.max(np.abs(dy_field)) + 1e-5)) * alpha_y

        # Scribe hand slant (natural calligraphic forward cursive slant)
        slant_shear = self.rng.uniform(-0.008, 0.020)
        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        map_x = grid_x + dx_field + (grid_y - (h / 2.0)) * slant_shear
        map_y = grid_y + dy_field

        # 5. Generate smooth 2D pen pressure & dipping field
        macro_raw = cv2.GaussianBlur(
            self.np_rng.uniform(0.44, 0.95, (h // 25, w // 25)).astype(np.float32),
            (0, 0),
            2.2,
        )
        micro_raw = cv2.GaussianBlur(
            self.np_rng.uniform(-0.12, 0.12, (h // 8, w // 8)).astype(np.float32),
            (0, 0),
            1.5,
        )
        macro_map = cv2.resize(macro_raw, (w, h), interpolation=cv2.INTER_CUBIC)
        micro_map = cv2.resize(micro_raw, (w, h), interpolation=cv2.INTER_CUBIC)
        pressure_map = np.clip(macro_map + micro_map, 0.36, 1.0)

        # Directional reed pen (kalam / borū) nib: 45-degree angled flat bevel
        kalam_nib = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        ], dtype=np.uint8)
        thin_kernel = np.ones((2, 2), dtype=np.uint8)

        full_ink_alpha = np.zeros((h, w), dtype=np.uint8)
        full_rubric_alpha = np.zeros((h, w), dtype=np.uint8)
        full_core_density = np.zeros((h, w), dtype=np.float32)
        line_annotations: List[Dict[str, Any]] = []

        PUNCT_CHARS = {'\u0964', '\u0965', '\U00011641', '\U00011642', '\U000111C5', '\U000111C6', '\U000111C7', '\U000111C8'}

        for line in page_layout.lines:
            raw_text = line.text.strip()
            if not raw_text:
                continue

            # Natural irregular line-start margin drift (human margin alignment)
            margin_drift = self.rng.randint(-6, 6)
            x_cursor = max(c_x1, line.x + margin_drift)
            base_y = line.y

            # Baseline wave parameters specific to this line (gentle organic drift)
            line_wave_period = self.rng.uniform(600, 900)
            line_wave_amp = self.rng.uniform(1.1, 2.0)
            line_wave_phase = self.rng.uniform(0, math.pi * 2)
            line_slope = self.rng.uniform(-0.0012, 0.0012)

            # Scribe hand pressure variation between lines (different ink loads over time)
            line_pressure_bias = self.rng.uniform(-0.14, 0.16)
            line_pressure = np.clip(pressure_map + line_pressure_bias, 0.34, 1.0)

            # Continuous line canvases for assembling shaped grapheme clusters and rubrication
            line_canvas = np.zeros((h, w), dtype=np.uint8)
            rubric_canvas = np.zeros((h, w), dtype=np.uint8)
            pad = 18

            words = raw_text.split(" ")
            for w_idx, word in enumerate(words):
                if not word:
                    continue

                # Check if word is punctuation or section marker for rubrication (sindūra red)
                is_rubric = any(ch in PUNCT_CHARS for ch in word) and (self.rng.random() < 0.85)

                try:
                    w_adv = font.getlength(word)
                except Exception:
                    bb = font.getbbox(word)
                    w_adv = bb[2] - bb[0]

                bb = font.getbbox(word)
                ww = max(int(w_adv), bb[2] - bb[0])
                wh = bb[3] - bb[1]
                pw = max(ww + pad * 2, 48)
                ph = max(wh + pad * 2, int(font_size * 2.5))

                # 1. Rasterize word with FreeType font engine (preserving intact shirorekha and grapheme clusters)
                w_patch = Image.new("L", (pw, ph), 0)
                w_draw = ImageDraw.Draw(w_patch)
                w_draw.text((pad, pad), word, font=font, fill=255)

                # 2. Word-level organic scribe variation (subtle scale, micro-rotation, sub-pixel displacement)
                sx = self.rng.uniform(0.97, 1.03)
                sy = self.rng.uniform(0.97, 1.03)
                rot = self.rng.uniform(-0.6, 0.6)
                dx = self.rng.uniform(-0.8, 0.8)
                dy = self.rng.uniform(-0.6, 0.6)

                center = (pad + ww / 2.0, pad + font_size * 0.40)
                rad = math.radians(rot)
                cos_a = math.cos(rad)
                sin_a = math.sin(rad)
                M = np.array([
                    [sx * cos_a, -sy * sin_a, dx + center[0] * (1 - sx * cos_a) + center[1] * sy * sin_a],
                    [sx * sin_a,  sy * cos_a, dy - center[0] * sx * sin_a + center[1] * (1 - sy * cos_a)],
                ], dtype=np.float32)

                warped_w = cv2.warpAffine(
                    np.array(w_patch),
                    M,
                    (pw, ph),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_CONSTANT,
                )

                # 3. Pen pressure & stroke thickness variation (reed-pen nib dynamics)
                py_idx = min(h - 1, max(0, int(base_y)))
                px_idx = min(w - 1, max(0, int(x_cursor)))
                c_press = line_pressure[py_idx, px_idx]

                if c_press > 0.55:
                    thick_w = np.clip((c_press - 0.55) * 1.5, 0.0, 0.35)
                    w_dilated = cv2.dilate(warped_w, kalam_nib, iterations=1)
                    warped_w = cv2.addWeighted(warped_w, 1.0 - thick_w, w_dilated, thick_w, 0)
                elif c_press < 0.48:
                    thin_w = np.clip((0.48 - c_press) * 1.4, 0.0, 0.25)
                    w_eroded = cv2.erode(warped_w, thin_kernel, iterations=1)
                    warped_w = cv2.addWeighted(warped_w, 1.0 - thin_w, w_eroded, thin_w, 0)

                # 4. Subtle stroke edge irregularity (tiny microscopic nib friction)
                edge_mask = cv2.morphologyEx(warped_w, cv2.MORPH_GRADIENT, np.ones((2, 2), np.uint8))
                if np.any(edge_mask > 0):
                    edge_noise = self.np_rng.normal(0, 8, warped_w.shape).astype(np.float32)
                    edge_factor = edge_mask.astype(np.float32) / 255.0
                    warped_w = np.clip(
                        warped_w.astype(np.float32) + edge_noise * edge_factor,
                        0,
                        255,
                    ).astype(np.uint8)

                # 5. Place word along organic baseline with continuous shirorekha connectivity
                dx_line = x_cursor - c_x1
                wave_y = (
                    math.sin(dx_line * (2 * math.pi / line_wave_period) + line_wave_phase) * line_wave_amp
                    + dx_line * line_slope
                )
                paste_x = int(x_cursor - pad)
                paste_y = int(base_y + wave_y - pad)

                # ROI intersection clipping onto line canvas: ensure word stays strictly inside safe margins
                if x_cursor + int(w_adv * sx) > c_x2:
                    break

                x1_l, y1_l = max(0, paste_x), max(0, paste_y)
                x2_l, y2_l = min(w, paste_x + pw), min(h, paste_y + ph)
                if x1_l >= x2_l or y1_l >= y2_l:
                    continue

                x1_p, y1_p = x1_l - paste_x, y1_l - paste_y
                x2_p, y2_p = x1_p + (x2_l - x1_l), y1_p + (y2_l - y1_l)

                if is_rubric:
                    rubric_canvas[y1_l:y2_l, x1_l:x2_l] = np.maximum(
                        rubric_canvas[y1_l:y2_l, x1_l:x2_l],
                        warped_w[y1_p:y2_p, x1_p:x2_p],
                    )
                else:
                    line_canvas[y1_l:y2_l, x1_l:x2_l] = np.maximum(
                        line_canvas[y1_l:y2_l, x1_l:x2_l],
                        warped_w[y1_p:y2_p, x1_p:x2_p],
                    )

                # Advance cursor with natural rhythm and variable word spacing
                space_factor = self.rng.uniform(0.70, 1.15)
                x_cursor += int(w_adv * sx) + max(6, int(base_space_w * space_factor))

            # Warp entire line with smooth continuous 2D muscular displacement field
            warped_line = cv2.remap(
                line_canvas,
                map_x,
                map_y,
                interpolation=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
            )
            warped_rubric = cv2.remap(
                rubric_canvas,
                map_x,
                map_y,
                interpolation=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
            )

            # 6. Natural Ink Deposition: Stroke core density via distance transform
            combined_warped = np.maximum(warped_line, warped_rubric)
            bin_mask = (combined_warped > 20).astype(np.uint8)
            if np.any(bin_mask > 0):
                dist_map = cv2.distanceTransform(bin_mask, cv2.DIST_L2, 3)
                max_d = np.percentile(dist_map[dist_map > 0], 85)
                line_core = np.clip(dist_map / max(max_d, 1.0), 0.0, 1.0)
            else:
                line_core = np.zeros_like(warped_line, dtype=np.float32)

            # Paper substrate interaction: ink settles slightly into paper valleys
            substrate_factor = 0.88 + 0.12 * bg_gray

            # Capillary edge feathering
            feather = cv2.GaussianBlur(warped_line, (3, 3), 0.65)
            edge_factor = np.clip(1.0 - line_core * 2.0, 0.0, 1.0)

            line_alpha = np.clip(
                warped_line.astype(np.float32)
                * substrate_factor
                * (0.78 + 0.22 * line_core)
                * line_pressure
                + feather.astype(np.float32) * 0.10 * edge_factor,
                0,
                255,
            ).astype(np.uint8)

            rubric_alpha = np.clip(
                warped_rubric.astype(np.float32)
                * substrate_factor
                * (0.80 + 0.20 * line_core)
                * line_pressure,
                0,
                255,
            ).astype(np.uint8)

            # Accumulate into master masks
            full_ink_alpha = np.maximum(full_ink_alpha, line_alpha)
            full_rubric_alpha = np.maximum(full_rubric_alpha, rubric_alpha)
            full_core_density = np.maximum(full_core_density, line_core)

            # 7. Exact Ground-Truth Bounding Box Synchronization from all final non-zero ink pixels
            total_rendered_line = np.maximum(line_alpha, rubric_alpha)
            coords = cv2.findNonZero(total_rendered_line)
            if coords is not None:
                bx, by, bw, bh = cv2.boundingRect(coords)
                exact_bbox = [
                    max(0, int(bx) - 1),
                    max(0, int(by) - 1),
                    min(w, int(bx + bw) + 1),
                    min(h, int(by + bh) + 1),
                ]
            else:
                exact_bbox = [c_x1, base_y, c_x1 + 10, base_y + font_size]

            line_annotations.append({
                "line_index": line.line_index,
                "text": raw_text,
                "bbox": exact_bbox,
                "script": script_name,
                "block_type": getattr(line, "block_type", "main"),
            })

        # 8. Scribe Micro-Smudges (Occasional subtle smudge near text without obscuring characters)
        if self.rng.random() < 0.45:
            text_pts = np.argwhere(full_ink_alpha > 200)
            if len(text_pts) > 0:
                pt = text_pts[self.rng.randint(0, len(text_pts) - 1)]
                sy, sx = pt[0], pt[1]
                sy1, sy2 = max(0, sy - 6), min(h, sy + 16)
                sx1, sx2 = max(0, sx + 4), min(w, sx + 50)  # Trailing to the right
                smudge_roi = full_ink_alpha[sy1:sy2, sx1:sx2]
                if smudge_roi.size > 0 and np.sum(smudge_roi) > 200:
                    smudge_k = np.ones((1, 9), dtype=np.float32) / 9.0
                    smudged_patch = cv2.filter2D(smudge_roi, -1, smudge_k)
                    full_ink_alpha[sy1:sy2, sx1:sx2] = cv2.addWeighted(
                        smudge_roi, 0.80, smudged_patch, 0.20, 0
                    )

        # 9. Multi-Tonal Ink Deposition (Carbon / Iron-Gall Soot to Diluted Sepia)
        palette = self.INK_PALETTES.get(script_key, self.INK_PALETTES["devanagari"])
        dense_col = palette["dense"]
        diluted_col = palette["diluted"]

        if ink_color is not None:
            dense_col = np.array(ink_color, dtype=np.float32)
            diluted_col = np.clip(dense_col * 1.8, 0, 255).astype(np.float32)

        # Tone interpolation: dark carbon soot core transitioning to warm diluted wash at edges
        tone_factor = np.clip(0.50 * pressure_map + 0.50 * full_core_density, 0.0, 1.0)
        ink_rgb_layer = np.zeros((h, w, 3), dtype=np.uint8)
        rubric_rgb_layer = np.zeros((h, w, 3), dtype=np.uint8)

        # Rubrication cinnabar palette
        rubric_dense = np.array([155, 34, 28], dtype=np.float32)
        rubric_diluted = np.array([185, 75, 55], dtype=np.float32)

        for c in range(3):
            channel_col = dense_col[c] * tone_factor + diluted_col[c] * (1.0 - tone_factor)
            ink_rgb_layer[:, :, c] = np.clip(channel_col, 0, 255).astype(np.uint8)

            rubric_col = rubric_dense[c] * tone_factor + rubric_diluted[c] * (1.0 - tone_factor)
            rubric_rgb_layer[:, :, c] = np.clip(rubric_col, 0, 255).astype(np.uint8)

        # 10. Composite ink onto manuscript canvas using calligraphic feathered alpha masks
        ink_pil = Image.fromarray(ink_rgb_layer, mode="RGB")
        mask_pil = Image.fromarray(full_ink_alpha, mode="L")
        img.paste(ink_pil, (0, 0), mask_pil)

        if np.any(full_rubric_alpha > 0):
            rubric_pil = Image.fromarray(rubric_rgb_layer, mode="RGB")
            rubric_mask_pil = Image.fromarray(full_rubric_alpha, mode="L")
            img.paste(rubric_pil, (0, 0), rubric_mask_pil)

        return img, line_annotations, font_used
