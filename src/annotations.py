"""
annotations.py
==============
Generates and manages ground-truth labels for OCR and document analysis:
- Per-sample JSON metadata with line transcriptions, bounding boxes, and generation params
- Standard COCO-format dataset annotation export
- Facilitates evaluation and downstream training
"""

import json
import os
from typing import List, Dict, Any, Optional


class AnnotationManager:
    """Creates, serializes, and formats ground-truth annotations."""

    def __init__(self, output_dir: str = "data/output"):
        """
        Args:
            output_dir: Root directory where annotations and images will be saved.
        """
        self.output_dir = output_dir
        self.records: List[Dict[str, Any]] = []

    def create_sample_record(
        self,
        image_id: str,
        image_filename: str,
        width: int,
        height: int,
        script: str,
        lines: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Constructs a structured record for a single generated manuscript page.

        Args:
            image_id: Unique string identifier for the sample.
            image_filename: Relative filename of the image.
            width: Image width in pixels.
            height: Image height in pixels.
            script: Script name ('devanagari', 'modi', 'sharada').
            lines: List of line records with 'bbox' and 'text'.
            metadata: Additional generation metadata (background type, augmentations).

        Returns:
            Annotation dictionary.
        """
        record = {
            "image_id": image_id,
            "file_name": image_filename,
            "width": width,
            "height": height,
            "script": script,
            "lines": lines,
            "metadata": metadata or {},
        }
        self.records.append(record)
        return record

    def format_markdown_annotation(self, record: Dict[str, Any]) -> str:
        """
        Formats a sample record into a structured Markdown (.md) annotation document.
        MANDATORY per technical assignment requirements.

        Args:
            record: Sample annotation record.

        Returns:
            Formatted Markdown string containing full ground-truth transcriptions,
            line-level bounding boxes, and generation metadata.
        """
        image_id = record.get("image_id", "unknown")
        file_name = record.get("file_name", "")
        script = record.get("script", "").capitalize()
        width = record.get("width", 0)
        height = record.get("height", 0)
        lines = record.get("lines", [])
        metadata = record.get("metadata", {})

        md_lines = [
            f"# Manuscript Annotation: {image_id}",
            "",
            "## Summary",
            f"- **Image File**: `{file_name}`",
            f"- **Image ID**: `{image_id}`",
            f"- **Script**: {script}",
            f"- **Dimensions**: {width} × {height} px",
            f"- **Total Lines**: {len(lines)}",
            "",
            "## Ground Truth Transcription",
            "",
        ]

        # Full transcribed passage
        if lines:
            for line in lines:
                md_lines.append(f"{line.get('text', '')}  ")
        else:
            md_lines.append("*No text lines recorded.*")

        # Line-by-line bounding box breakdown
        md_lines.extend([
            "",
            "## Line-Level Annotations",
            "",
            "| Line # | Bounding Box [x1, y1, x2, y2] | Text |",
            "| :---: | :--- | :--- |",
        ])

        for line in lines:
            l_idx = line.get("line_index", 0) + 1
            bbox_str = str(line.get("bbox", []))
            text_str = line.get("text", "").replace("|", "\\|")
            md_lines.append(f"| {l_idx} | `{bbox_str}` | {text_str} |")

        # Generation metadata section
        if metadata:
            md_lines.extend([
                "",
                "## Generation Metadata",
                "",
            ])
            for k, v in metadata.items():
                md_lines.append(f"- **{k}**: {v}")

        md_lines.append("")
        return "\n".join(md_lines)

    def save_sample_markdown(self, record: Dict[str, Any], filepath: str) -> None:
        """
        Saves the mandatory paired Markdown (.md) annotation file for a generated image.

        Args:
            record: Sample record dictionary.
            filepath: Destination path ending with .md.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        content = self.format_markdown_annotation(record)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

    def save_sample_json(self, record: Dict[str, Any], filepath: str) -> None:
        """Saves an individual sample's ground truth to an optional JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

    def export_coco_format(self, output_filepath: str) -> None:
        """
        Exports all recorded samples into a standard COCO-style JSON annotation file.

        Args:
            output_filepath: Target path for the unified COCO JSON file.
        """
        categories = [
            {"id": 1, "name": "devanagari", "supercategory": "script"},
            {"id": 2, "name": "modi", "supercategory": "script"},
            {"id": 3, "name": "sharada", "supercategory": "script"},
        ]
        script_to_id = {c["name"]: c["id"] for c in categories}

        images = []
        annotations = []
        ann_id = 1

        for img_idx, rec in enumerate(self.records, start=1):
            images.append({
                "id": img_idx,
                "file_name": rec["file_name"],
                "width": rec["width"],
                "height": rec["height"],
            })

            cat_id = script_to_id.get(rec["script"].lower(), 1)

            for line in rec["lines"]:
                x1, y1, x2, y2 = line["bbox"]
                w = max(0, x2 - x1)
                h = max(0, y2 - y1)
                annotations.append({
                    "id": ann_id,
                    "image_id": img_idx,
                    "category_id": cat_id,
                    "bbox": [x1, y1, w, h],
                    "area": float(w * h),
                    "iscrowd": 0,
                    "text": line.get("text", ""),
                })
                ann_id += 1

        coco_data = {
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }

        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(coco_data, f, ensure_ascii=False, indent=2)
