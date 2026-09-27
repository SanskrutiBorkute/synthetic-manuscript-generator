"""
background_generator.py
========================
Generates synthetic historical manuscript background textures:
- Parchment / Vellum (warm cream, subtle grain, aged edges)
- Palm Leaf (Tala/Birch bark, horizontal fiber patterns, warm ochre tones)
- Aged Paper (mottled spots, patina, tea/coffee stain undertones)

Provides a clean interface for synthesizing manuscript canvas substrates
without relying on external static image dependencies.
"""

import random
from typing import Tuple, Optional
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageDraw


class BackgroundGenerator:
    """Generates synthetic textured backgrounds simulating historical manuscript folios."""

    BACKGROUND_TYPES = ("parchment", "palm_leaf", "aged_paper")

    # Rich authentic historical color palettes
    PALETTES = {
        "parchment": {
            "base": (225, 205, 166),       # Warm antique vellum tone
            "grain_intensity": 14,
            "vignette": True,
        },
        "palm_leaf": {
            "base": (192, 156, 106),       # Rich dried palmyra leaf tan/ochre
            "grain_intensity": 20,
            "vignette": True,
        },
        "aged_paper": {
            "base": (220, 194, 148),       # Warm amber handmade rag paper
            "grain_intensity": 16,
            "vignette": True,
        },
    }

    def __init__(self, seed: Optional[int] = None):
        """
        Args:
            seed: Optional random seed for reproducible textures.
        """
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def generate(
        self,
        bg_type: str = "parchment",
        size: Tuple[int, int] = (1600, 1000),
        add_binding_hole: bool = False,
    ) -> Image.Image:
        """
        Generates a synthetic background image.

        Args:
            bg_type: One of 'parchment', 'palm_leaf', or 'aged_paper'.
            size: (width, height) in pixels.
            add_binding_hole: Whether to add a traditional pothi cord hole (palm leaf).

        Returns:
            PIL Image (RGB) with synthetic manuscript substrate texture.
        """
        bg_key = bg_type.lower()
        if bg_key not in self.BACKGROUND_TYPES:
            bg_key = "parchment"

        width, height = size

        # Specialized synthesis for historical dried palm-leaf manuscript folios
        if bg_key == "palm_leaf":
            return self._generate_palm_leaf(width, height, add_binding_hole=True)

        palette = self.PALETTES[bg_key]
        base_color = palette["base"]
        intensity = palette["grain_intensity"]

        # 1. Base substrate tone with subtle natural batch variation
        r_offset = self.rng.randint(-8, 8)
        g_offset = self.rng.randint(-8, 8)
        b_offset = self.rng.randint(-8, 8)
        r = max(0, min(255, base_color[0] + r_offset))
        g = max(0, min(255, base_color[1] + g_offset))
        b = max(0, min(255, base_color[2] + b_offset))

        canvas = np.full((height, width, 3), [r, g, b], dtype=np.float32)

        # 2. Multi-scale paper grain / fiber noise
        noise_fine = self.np_rng.normal(0, intensity * 0.7, (height, width, 3))
        canvas += noise_fine

        # 3. Substrate-specific texture synthesis
        if bg_key == "aged_paper":
            # Uneven handmade paper pulp clumps
            pulp_noise = self.np_rng.normal(0, 16, (height // 4, width // 4, 3)).astype(np.float32)
            pulp_up = cv2.resize(pulp_noise, (width, height), interpolation=cv2.INTER_LINEAR)
            canvas += pulp_up * 0.45

            # Tea/aging mottled discoloration patches
            stain_noise = self.np_rng.normal(0, 14, (height // 12, width // 12, 3)).astype(np.float32)
            stain_up = cv2.resize(stain_noise, (width, height), interpolation=cv2.INTER_CUBIC)
            canvas += stain_up * 0.40

        elif bg_key == "parchment":
            # Vellum follicle patterns and subtle density variation
            vellum_noise = self.np_rng.normal(0, 16, (height // 6, width // 6, 3)).astype(np.float32)
            vellum_up = cv2.resize(vellum_noise, (width, height), interpolation=cv2.INTER_LINEAR)
            canvas += vellum_up * 0.38

        # 4. Subtle edge oxidation / aging vignette
        if palette.get("vignette", True):
            canvas = self._apply_vignette(canvas, width, height, bg_key)

        # Clip and convert to PIL Image
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)
        img = Image.fromarray(canvas, mode="RGB")

        # Smooth slightly to mimic organic material softness
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

        # 5. Add organic foxing spots on aged paper and parchment
        if bg_key in ("aged_paper", "parchment") and self.rng.random() < 0.75:
            img = self._add_foxing(img, width, height)

        return img

    def _generate_palm_leaf(self, width: int, height: int, add_binding_hole: bool = True) -> Image.Image:
        """
        Synthesizes a hyper-realistic historical dried palm-leaf (tala-patra) manuscript folio:
        - Natural warm dried tan/amber leaf palette (strictly no green, no artificial color noise)
        - Multi-scale anisotropic longitudinal fibers running lengthwise across the entire folio
        - Visible parallel vascular vein striations every 10-14px with natural organic variation
        - Transverse cupping curvature (natural cylindrical leaf shading: golden center, deeper amber borders)
        - Handling patina & age discoloration in central reading zone and margins
        - Fine horizontal drying checks & natural fiber imperfections
        - Slightly irregular organic knife-trimmed leaf edges with cured cuticle perimeter
        - Realistic pierced cord-binding hole with smooth continuous abrasion wear halo and 3D aperture depth
        """
        # 1. Authentic dried palmyra leaf tan/amber base coloration
        r_base = self.rng.randint(200, 208)
        g_base = self.rng.randint(162, 170)
        b_base = self.rng.randint(112, 120)

        base_color = np.array([r_base, g_base, b_base], dtype=np.float32)
        canvas = np.ones((height, width, 3), dtype=np.float32) * base_color

        # 2. Broad smooth organic drying, curing, & handling variations (scalar luminance)
        cloud_low = self.np_rng.normal(1.0, 0.05, (max(4, height // 45), max(4, width // 60), 1)).astype(np.float32)
        cloud_up = cv2.resize(cloud_low, (width, height), interpolation=cv2.INTER_CUBIC)
        cloud_smooth = cv2.GaussianBlur(cloud_up, (0, 0), 40.0)
        if cloud_smooth.ndim == 2:
            cloud_smooth = np.expand_dims(cloud_smooth, axis=-1)
        canvas *= cloud_smooth

        # Handling patina along horizontal center (gentle warmth and slight darkening in central reading zone)
        y_norm = np.linspace(-1.0, 1.0, height).astype(np.float32)
        handling = np.exp(- (y_norm / 0.55)**2) * 0.03
        canvas *= (1.0 + np.expand_dims(handling, axis=(1, 2)) * np.array([0.02, 0.01, -0.02], dtype=np.float32))

        # 3. Transverse cupping curvature (natural leaf cross-section shading: lighter golden center, deeper amber edges)
        cupping = (y_norm ** 2) * 0.13  # 13% darkening at top/bottom extremes
        canvas *= (1.0 - np.expand_dims(cupping, axis=(1, 2)))

        # 4. Organic longitudinal wave coordinate (gentle botanical grain undulation)
        x_idx = np.arange(width, dtype=np.float32)
        wave_y = np.sin(x_idx * 0.004) * 2.4 + np.cos(x_idx * 0.011) * 1.5 + np.sin(x_idx * 0.027) * 0.8

        grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
        map_y = np.clip(grid_y + np.expand_dims(wave_y, 0), 0, height - 1).astype(np.float32)

        # 5. Multi-scale anisotropic longitudinal fibers (continuous plant tissue)
        # Scale A: Fine high-density micro-fibers (aspect ratio ~1:14)
        fine_noise = self.np_rng.normal(1.0, 0.07, (height, width // 8)).astype(np.float32)
        fine_stretched = cv2.resize(fine_noise, (width, height), interpolation=cv2.INTER_LINEAR)
        fine_warped = cv2.remap(fine_stretched, grid_x, map_y, interpolation=cv2.INTER_LINEAR)
        fine_warped = cv2.GaussianBlur(fine_warped, (9, 1), 0)
        canvas *= np.expand_dims(fine_warped, axis=-1)

        # Scale B: Medium longitudinal fiber bundles
        med_noise = self.np_rng.normal(1.0, 0.05, (height // 2, width // 16)).astype(np.float32)
        med_stretched = cv2.resize(med_noise, (width, height), interpolation=cv2.INTER_LINEAR)
        med_warped = cv2.remap(med_stretched, grid_x, map_y, interpolation=cv2.INTER_LINEAR)
        med_warped = cv2.GaussianBlur(med_warped, (15, 1), 0)
        canvas *= np.expand_dims(med_warped, axis=-1)

        # 6. Parallel vascular vein striations (distinct parallel fibrovascular ribs every 9-14px)
        vein_step = self.rng.randint(10, 13)
        vein_ys = np.arange(8, height - 8, vein_step)
        vein_ys = [int(vy + self.rng.uniform(-1.8, 1.8)) for vy in vein_ys]

        for vy in vein_ys:
            y_track = np.clip(np.round(vy + wave_y).astype(int), 0, height - 1)
            # Vein strength modulates smoothly across length (fades and deepens naturally)
            vein_mod = 0.65 + 0.35 * np.sin(x_idx * 0.0025 + vy * 0.7)
            v_depth = self.rng.uniform(0.045, 0.085) * vein_mod
            v_highlight = v_depth * 0.40

            for dx in range(width):
                yp = y_track[dx]
                canvas[yp, dx, :] *= (1.0 - v_depth[dx])
                if yp + 1 < height:
                    canvas[yp + 1, dx, :] *= (1.0 + v_highlight[dx])

        # 7. Natural drying checks & fine fiber scratches
        num_scratches = self.rng.randint(25, 40)
        for _ in range(num_scratches):
            sc_y = self.rng.randint(12, height - 12)
            sc_x1 = self.rng.randint(25, width - 300)
            sc_len = self.rng.randint(40, 220)
            sc_x2 = min(width - 20, sc_x1 + sc_len)
            sc_dark = self.rng.uniform(0.10, 0.22)
            y_track = np.clip(np.round(sc_y + wave_y[sc_x1:sc_x2]).astype(int), 0, height - 1)
            taper = np.sin(np.linspace(0, np.pi, sc_x2 - sc_x1))
            for i, x in enumerate(range(sc_x1, sc_x2)):
                canvas[y_track[i], x, :] *= (1.0 - sc_dark * taper[i])

        # 8. Darkened cured cuticle margins (top & bottom edges)
        top_rind = np.expand_dims(np.linspace(1.0, 0.0, 26).astype(np.float32) ** 1.5, axis=(1, 2)) * 0.18
        canvas[:26, :, :] *= (1.0 - top_rind)
        bot_rind = np.expand_dims(np.linspace(0.0, 1.0, 26).astype(np.float32) ** 1.5, axis=(1, 2)) * 0.18
        canvas[-26:, :, :] *= (1.0 - bot_rind)

        # Ends oxidation
        l_rind = np.expand_dims(np.linspace(1.0, 0.0, 32).astype(np.float32) ** 1.7, axis=(0, 2)) * 0.15
        canvas[:, :32, :] *= (1.0 - l_rind)
        r_rind = np.expand_dims(np.linspace(0.0, 1.0, 32).astype(np.float32) ** 1.7, axis=(0, 2)) * 0.15
        canvas[:, -32:, :] *= (1.0 - r_rind)

        # 9. Realistic Cord-Binding Aperture: Soft Continuous Abrasion Halo (Single hole at cx=144)
        if add_binding_hole:
            cx = 144
            cy = height // 2
            r_hole = 12

            dx = grid_x - cx
            dy = grid_y - cy
            dist = np.sqrt(dx**2 + dy**2)

            # Continuous radial cord abrasion halo
            halo_mask = np.clip((34.0 - dist) / 22.0, 0.0, 1.0)
            halo_factor = (halo_mask ** 1.5) * 0.18
            canvas *= (1.0 - np.expand_dims(halo_factor, axis=-1))

            # Compressed dark rim from punch tool
            rim_mask = np.clip(1.0 - np.abs(dist - 13.0) / 1.6, 0.0, 1.0)
            rim_darkening = rim_mask * 0.45
            canvas *= (1.0 - np.expand_dims(rim_darkening, axis=-1))

        canvas = np.clip(canvas, 0, 255).astype(np.uint8)
        img = Image.fromarray(canvas, mode="RGB")

        # Aperture void with 3D shadow and bevel highlight
        if add_binding_hole:
            draw = ImageDraw.Draw(img)
            cx = 144
            cy = height // 2
            r_hole = 12

            # Deep void fill
            draw.ellipse([cx - r_hole, cy - r_hole, cx + r_hole, cy + r_hole], fill=(22, 15, 10))
            # 3D interior shadow on top-left bevel (light from top-left)
            draw.arc([cx - r_hole, cy - r_hole, cx + r_hole, cy + r_hole], start=120, end=300, fill=(10, 6, 4), width=2)
            # Subtle inner bevel highlight on bottom-right
            draw.arc([cx - r_hole + 1, cy - r_hole + 1, cx + r_hole - 1, cy + r_hole - 1], start=300, end=480, fill=(110, 80, 50), width=1)

        # 10. Organic leaf edge contours (slightly irregular knife-cut edge)
        edge_overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        d_edge = ImageDraw.Draw(edge_overlay)

        top_edge_wave = np.sin(x_idx * 0.015) * 1.8 + np.sin(x_idx * 0.055) * 0.9
        bot_edge_wave = np.sin(x_idx * 0.018) * 1.8 + np.cos(x_idx * 0.048) * 0.9

        for x in range(width):
            ey_t = int(2.4 + top_edge_wave[x])
            if ey_t > 0:
                d_edge.line([(x, 0), (x, ey_t)], fill=(68, 45, 24, 150))
            ey_b = int(2.4 + bot_edge_wave[x])
            if ey_b > 0:
                d_edge.line([(x, height - ey_b), (x, height)], fill=(68, 45, 24, 150))

        for y in range(height):
            ex_l = int(2.2 + np.sin(y * 0.035) * 1.1)
            if ex_l > 0:
                d_edge.line([(0, y), (ex_l, y)], fill=(68, 45, 24, 150))
            ex_r = int(2.2 + np.cos(y * 0.03) * 1.1)
            if ex_r > 0:
                d_edge.line([(width - ex_r, y), (width, y)], fill=(68, 45, 24, 150))

        # Corner softening (soft rounded leaf corners)
        corner_r = 7
        for cx_c, cy_c in [(0, 0), (0, height), (width, 0), (width, height)]:
            for dy_c in range(corner_r):
                for dx_c in range(corner_r):
                    if (dx_c**2 + dy_c**2) > corner_r**2:
                        px = cx_c + dx_c if cx_c == 0 else cx_c - 1 - dx_c
                        py = cy_c + dy_c if cy_c == 0 else cy_c - 1 - dy_c
                        d_edge.point((px, py), fill=(55, 35, 18, 200))

        edge_overlay = edge_overlay.filter(ImageFilter.GaussianBlur(radius=0.8))
        img = Image.alpha_composite(img.convert("RGBA"), edge_overlay).convert("RGB")

        return img

    def _apply_vignette(self, canvas: np.ndarray, width: int, height: int, bg_key: str) -> np.ndarray:
        """Simulates aged, oxidized darker borders commonly found on ancient folios."""
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        X, Y = np.meshgrid(x, y)
        dist = np.sqrt(X**2 + Y**2)
        vignette_depth = 0.32 if bg_key == "aged_paper" else 0.26
        vignette_mask = 1.0 - vignette_depth * np.clip(dist - 0.40, 0, 1.0) ** 1.5
        vignette_mask = np.expand_dims(vignette_mask, axis=-1)
        return canvas * vignette_mask

    def _add_foxing(self, img: Image.Image, width: int, height: int) -> Image.Image:
        """Adds subtle reddish-brown chemical oxidation spots (foxing) found in antique paper."""
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        num_spots = self.rng.randint(6, 18)
        for _ in range(num_spots):
            sx = self.rng.randint(30, width - 30)
            sy = self.rng.randint(30, height - 30)
            sr = self.rng.randint(2, 6)
            s_alpha = self.rng.randint(35, 75)
            draw.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], fill=(135, 88, 48, s_alpha))
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=1.2))
        return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    def _draw_binding_holes(self, img: Image.Image, width: int, height: int) -> Image.Image:
        """Draws traditional pierced cord apertures in palm-leaf margins."""
        draw = ImageDraw.Draw(img)
        # Position cord hole in left margin area (safe from text)
        hole_positions = [(int(width * 0.08), height // 2)]
        if width > 1400 and self.rng.random() < 0.4:
            hole_positions.append((int(width * 0.92), height // 2))

        for cx, cy in hole_positions:
            r = self.rng.randint(10, 14)
            # Compressed aged halo
            draw.ellipse([cx - r - 5, cy - r - 5, cx + r + 5, cy + r + 5], outline=(65, 48, 30), width=2)
            draw.ellipse([cx - r - 2, cy - r - 2, cx + r + 2, cy + r + 2], outline=(45, 32, 20), width=2)
            # Pierced hole center
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(32, 22, 14))
        return img
