"""
augmentations.py
================
Pipeline for simulating physical manuscript aging, wear, and environmental degradation:
- Ink bleed & spreading
- Ink fading & oxidation
- Water & humidity stains
- Surface abrasions and fine noise
- Paper creases, folds, and edge wear
"""

import random
from typing import Optional, Dict, Any
import numpy as np
from PIL import Image, ImageFilter, ImageDraw


class ManuscriptAugmentor:
    """Applies realistic optical and physical degradation transforms to synthetic manuscripts."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, seed: Optional[int] = None):
        """
        Args:
            config: Augmentations configuration block from config.yaml.
            seed: Optional seed for reproducible effects.
        """
        self.config = config or {}
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def apply(self, image: Image.Image) -> Image.Image:
        """
        Applies a random suite of configured augmentations to the input image.

        Args:
            image: Rendered PIL Image.

        Returns:
            Augmented PIL Image.
        """
        if not self.config.get("enabled", True):
            return image

        augmented = image.copy()

        # Controlled aging tier: 25% clean, 50% moderate, 25% heavy
        aging_tier = self.rng.choices(["clean", "moderate", "heavy"], weights=[0.25, 0.50, 0.25])[0]

        # 1. Simulate ink bleed / micro-softening
        if aging_tier != "clean" and self._should_apply("ink_bleed"):
            augmented = self._apply_ink_bleed(augmented)

        # 2. Add aging stains (water/humidity drops)
        if aging_tier != "clean" and self._should_apply("stains"):
            max_stains = 2 if aging_tier == "moderate" else 4
            augmented = self._apply_stains(augmented, max_stains=max_stains)

        # 3. Simulate physical paper crease / fold
        if self._should_apply("paper_folds"):
            augmented = self._apply_crease(augmented)

        # 4. Page curl / corner light gradient
        if aging_tier == "heavy" or (self.rng.random() < 0.40):
            augmented = self._apply_page_curl(augmented)

        # 5. Add subtle noise / paper grain
        if self._should_apply("noise"):
            augmented = self._apply_noise(augmented)

        return augmented

    def _should_apply(self, effect_name: str) -> bool:
        """Checks if a given effect is enabled and passes its probability check."""
        cfg = self.config.get(effect_name, {})
        prob = cfg.get("probability", 0.5)
        return self.rng.random() < prob

    def _apply_ink_bleed(self, image: Image.Image) -> Image.Image:
        """Applies slight Gaussian blur to simulate capillary ink bleed into porous paper."""
        radius = self.rng.uniform(0.35, 0.85)
        return image.filter(ImageFilter.GaussianBlur(radius=radius))

    def _apply_stains(self, image: Image.Image, max_stains: int = 3) -> Image.Image:
        """Simulates semi-transparent organic stains or water drops on the manuscript."""
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        w, h = image.size

        num_stains = self.rng.randint(1, max_stains)
        for _ in range(num_stains):
            cx = self.rng.randint(60, w - 60)
            cy = self.rng.randint(60, h - 60)
            rx = self.rng.randint(25, 75)
            ry = self.rng.randint(18, 55)
            alpha = self.rng.randint(28, 62)
            stain_color = (145, 110, 60, alpha)  # Warm brownish water stain
            draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=stain_color)

        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=10))
        base_rgba = image.convert("RGBA")
        combined = Image.alpha_composite(base_rgba, overlay)
        return combined.convert("RGB")

    def _apply_crease(self, image: Image.Image) -> Image.Image:
        """Draws realistic physical creases with paired highlight and shadow lines."""
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        w, h = image.size

        # Vertical, horizontal, or diagonal crease
        crease_type = self.rng.choice(["v", "h", "diag"])
        shadow_col = (25, 18, 10, 52)
        highlight_col = (255, 250, 235, 42)

        if crease_type == "v":
            x = self.rng.randint(w // 4, 3 * w // 4)
            draw.line([(x, 0), (x, h)], fill=shadow_col, width=2)
            draw.line([(x + 2, 0), (x + 2, h)], fill=highlight_col, width=1)
        elif crease_type == "h":
            y = self.rng.randint(h // 4, 3 * h // 4)
            draw.line([(0, y), (w, y)], fill=shadow_col, width=2)
            draw.line([(0, y + 2), (w, y + 2)], fill=highlight_col, width=1)
        else:
            x1 = self.rng.randint(0, w // 2)
            x2 = self.rng.randint(w // 2, w)
            draw.line([(x1, 0), (x2, h)], fill=shadow_col, width=1)
            draw.line([(x1 + 1, 0), (x2 + 1, h)], fill=highlight_col, width=1)

        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=1.2))
        base_rgba = image.convert("RGBA")
        combined = Image.alpha_composite(base_rgba, overlay)
        return combined.convert("RGB")

    def _apply_page_curl(self, image: Image.Image) -> Image.Image:
        """Simulates soft corner/edge gradient shadowing from folio curvature."""
        w, h = image.size
        curl_overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(curl_overlay)

        # Pick random corner
        corner = self.rng.choice(["top_right", "bottom_right", "top_left", "bottom_left"])
        pts = {
            "top_right": [(w - 200, 0), (w, 0), (w, 200)],
            "bottom_right": [(w - 200, h), (w, h), (w, h - 200)],
            "top_left": [(0, 0), (200, 0), (0, 200)],
            "bottom_left": [(0, h), (200, h), (0, h - 200)],
        }
        draw.polygon(pts[corner], fill=(45, 30, 15, 35))
        curl_overlay = curl_overlay.filter(ImageFilter.GaussianBlur(radius=40))

        base_rgba = image.convert("RGBA")
        combined = Image.alpha_composite(base_rgba, curl_overlay)
        return combined.convert("RGB")

    def _apply_noise(self, image: Image.Image) -> Image.Image:
        """Adds subtle Gaussian film/sensor noise."""
        arr = np.array(image, dtype=np.float32)
        noise = self.np_rng.normal(0, 3.0, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")
