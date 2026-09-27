"""
text_loader.py
==============
Loads and preprocesses raw source text from markdown files in data/raw/:
- devanagari_md.md
- Modi_md.md
- sharada_md.md

Provides modular utilities to clean text, split into verses/paragraphs,
and sample passages for synthetic manuscript page rendering.
"""

import os
import re
import random
from typing import Dict, List, Optional


class TextLoader:
    """Manages loading and chunking of text for different Indic scripts."""

    SUPPORTED_SCRIPTS = ("devanagari", "modi", "sharada")

    def __init__(self, raw_data_paths: Optional[Dict[str, str]] = None):
        """
        Initialize the loader with paths to the raw source files.

        Args:
            raw_data_paths: Dictionary mapping script names to file paths.
        """
        self.raw_data_paths = raw_data_paths or {
            "devanagari": "data/raw/devanagari_md.md",
            "modi": "data/raw/Modi_md.md",
            "sharada": "data/raw/sharada_md.md",
        }
        self._cached_texts: Dict[str, List[str]] = {}

    def load_script_lines(self, script_name: str) -> List[str]:
        """
        Loads non-empty lines for the specified script from the raw file.

        Args:
            script_name: One of 'devanagari', 'modi', or 'sharada'.

        Returns:
            List of non-empty cleaned text lines.
        """
        script_key = script_name.lower()
        if script_key not in self.SUPPORTED_SCRIPTS:
            raise ValueError(f"Unsupported script '{script_name}'. Expected one of {self.SUPPORTED_SCRIPTS}")

        if script_key in self._cached_texts:
            return self._cached_texts[script_key]

        file_path = self.raw_data_paths.get(script_key)
        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(f"Raw source file not found for script '{script_name}' at path: {file_path}")

        lines: List[str] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                cleaned = self.clean_line(line)
                if cleaned:
                    lines.append(cleaned)

        self._cached_texts[script_key] = lines
        return lines

    @staticmethod
    def clean_line(line: str) -> str:
        """
        Cleans extraneous whitespace, Markdown artifacts, and trailing symbols.

        Args:
            line: Raw line string from file.

        Returns:
            Sanitized line string.
        """
        # Strip outer whitespace
        cleaned = line.strip()
        # Remove markdown heading hashes if any
        cleaned = re.sub(r"^#+\s*", "", cleaned)
        # Normalize internal multiple spaces/tabs
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        return cleaned

    def sample_passage(self, script_name: str, num_lines: int = 5, rng: Optional[random.Random] = None) -> List[str]:
        """
        Samples a consecutive or random passage suitable for fitting on a manuscript page.

        Args:
            script_name: Script to sample from.
            num_lines: Number of text lines desired.
            rng: Optional random.Random instance for reproducibility.

        Returns:
            List of sampled text lines.
        """
        lines = self.load_script_lines(script_name)
        if not lines:
            return []

        random_gen = rng or random.Random()
        if len(lines) <= num_lines:
            return lines

        # Select a contiguous chunk of lines to simulate coherent manuscript passages
        max_start = len(lines) - num_lines
        start_idx = random_gen.randint(0, max_start)
        return lines[start_idx : start_idx + num_lines]

    def get_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Returns basic statistics (total lines, total characters) for each script file.
        """
        stats = {}
        for script in self.SUPPORTED_SCRIPTS:
            try:
                lines = self.load_script_lines(script)
                total_chars = sum(len(line) for line in lines)
                stats[script] = {
                    "total_lines": len(lines),
                    "total_characters": total_chars,
                }
            except Exception as e:
                stats[script] = {"error": str(e)}
        return stats
