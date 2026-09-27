"""
Synthetic Manuscript Generator Package
======================================
Modular pipeline for generating synthetic historical Indic manuscript images
with realistic textures, scripts (Devanagari, Modi, Sharada), layouts,
augmentations, and ground-truth annotations.
"""

from .text_loader import TextLoader
from .background_generator import BackgroundGenerator
from .layout_engine import LayoutEngine
from .text_renderer import TextRenderer
from .augmentations import ManuscriptAugmentor
from .annotations import AnnotationManager
from .dataset_splitter import DatasetSplitter

__all__ = [
    "TextLoader",
    "BackgroundGenerator",
    "LayoutEngine",
    "TextRenderer",
    "ManuscriptAugmentor",
    "AnnotationManager",
    "DatasetSplitter",
]

__version__ = "0.1.0"
