"""
dataset_splitter.py
===================
Splits the generated synthetic manuscript dataset into stratified
train, validation, and test subsets based on script categories.
"""

import json
import os
import random
from typing import List, Dict, Any, Optional


class DatasetSplitter:
    """Manages reproducible stratified dataset partitioning into train, val, and test splits."""

    def __init__(
        self,
        train_ratio: float = 0.85,
        val_ratio: float = 0.10,
        test_ratio: float = 0.05,
        split_counts: Optional[Dict[str, int]] = None,
        seed: Optional[int] = 42,
    ):
        """
        Args:
            train_ratio: Proportion of data allocated for training (default: 0.85).
            val_ratio: Proportion allocated for validation (default: 0.10).
            test_ratio: Proportion allocated for testing (default: 0.05).
            split_counts: Optional exact counts per script, e.g. {'train': 85, 'val': 10, 'test': 5}.
            seed: Random seed for reproducible partitioning.
        """
        assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Split ratios must sum to 1.0"
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.split_counts = split_counts or {"train": 85, "val": 10, "test": 5}
        self.seed = seed

    def split_records(self, records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Performs stratified splitting grouped by script.
        Ensures exactly 85 train, 10 validation, 5 test per script for 100 samples.

        Args:
            records: List of sample annotation dictionaries.

        Returns:
            Dictionary with keys 'train', 'val', 'test', containing partitioned records.
        """
        rng = random.Random(self.seed)

        # Group records by script
        by_script: Dict[str, List[Dict[str, Any]]] = {}
        for rec in records:
            script = rec.get("script", "unknown")
            by_script.setdefault(script, []).append(rec)

        splits: Dict[str, List[Dict[str, Any]]] = {
            "train": [],
            "val": [],
            "test": [],
        }

        for script, items in by_script.items():
            shuffled = list(items)
            rng.shuffle(shuffled)
            n = len(shuffled)

            # If exact counts are specified and match length
            if self.split_counts and sum(self.split_counts.values()) == n:
                n_train = self.split_counts.get("train", 85)
                n_val = self.split_counts.get("val", 10)
            else:
                n_train = int(n * self.train_ratio)
                n_val = int(n * self.val_ratio)

            train_items = shuffled[:n_train]
            val_items = shuffled[n_train : n_train + n_val]
            test_items = shuffled[n_train + n_val :]

            splits["train"].extend(train_items)
            splits["val"].extend(val_items)
            splits["test"].extend(test_items)

        return splits

    def save_split_manifests(
        self,
        splits: Dict[str, List[Dict[str, Any]]],
        output_dir: str,
    ) -> None:
        """
        Saves split indices and manifests into JSON and Markdown files.

        Args:
            splits: Dictionary of split lists.
            output_dir: Directory where train.json/md, val.json/md, test.json/md will be saved.
        """
        os.makedirs(output_dir, exist_ok=True)
        for split_name, records in splits.items():
            # JSON manifest
            json_path = os.path.join(output_dir, f"{split_name}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)

            # Markdown catalog manifest
            md_path = os.path.join(output_dir, f"{split_name}.md")
            md_lines = [
                f"# Dataset Split: {split_name.upper()}",
                "",
                f"- **Total Samples**: {len(records)}",
                "",
                "| # | Image File | Paired Annotation (.md) | Script | Lines |",
                "| :---: | :--- | :--- | :--- | :---: |",
            ]
            for i, r in enumerate(records, 1):
                img_f = r.get("file_name", "")
                base_name = os.path.splitext(img_f)[0]
                md_f = f"{base_name}.md"
                script = r.get("script", "")
                n_lines = len(r.get("lines", []))
                md_lines.append(f"| {i} | `{img_f}` | `{md_f}` | {script} | {n_lines} |")

            md_lines.append("")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write("\n".join(md_lines))
