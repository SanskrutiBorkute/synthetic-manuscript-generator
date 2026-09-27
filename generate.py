"""
generate.py
===========
Main entry point for the Synthetic Manuscript Generator project.

Orchestrates the modular pipeline:
1. Loads configuration from config/config.yaml
2. Reads raw text via TextLoader (Devanagari, Modi, Sharada)
3. Synthesizes backgrounds via BackgroundGenerator
4. Computes spatial layouts via LayoutEngine
5. Renders glyphs & ink via TextRenderer
6. Applies physical aging & degradation via ManuscriptAugmentor
7. Formats annotations via AnnotationManager
8. Partitions dataset into train/val/test via DatasetSplitter
"""

import argparse
import os
import random
import sys
import yaml

from src import (
    TextLoader,
    BackgroundGenerator,
    LayoutEngine,
    TextRenderer,
    ManuscriptAugmentor,
    AnnotationManager,
    DatasetSplitter,
)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Loads YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_script_sample(
    script: str,
    sample_id: str,
    cfg: dict,
    text_loader: TextLoader,
    bg_generator: BackgroundGenerator,
    layout_engine: LayoutEngine,
    text_renderer: TextRenderer,
    augmentor: ManuscriptAugmentor,
    annotation_mgr: AnnotationManager,
    output_dir: str,
    rel_img_path: str = None,
    rng: random.Random = None,
    bg_type: str = None,
    layout_type: str = None,
) -> tuple:
    """
    Generates a single synthetic manuscript folio image and its paired .md annotation file.

    Args:
        script: Script name ('devanagari', 'modi', 'sharada').
        sample_id: Identifier for output filenames (e.g. 'Image_001').
        cfg: Configuration dictionary.
        text_loader: TextLoader instance.
        bg_generator: BackgroundGenerator instance.
        layout_engine: LayoutEngine instance.
        text_renderer: TextRenderer instance.
        augmentor: ManuscriptAugmentor instance.
        annotation_mgr: AnnotationManager instance.
        output_dir: Destination folder.
        rel_img_path: Path relative to dataset root for manifest files.
        rng: Optional random generator for reproducible variations.
        bg_type: Optional explicit background style ('aged_paper', 'palm_leaf', 'parchment').
        layout_type: Optional explicit layout ('standard', 'multi_block', 'marginal_commentary').

    Returns:
        Tuple of (image_file_path, md_annotation_file_path, record_dict).
    """
    local_rng = rng or random.Random()

    # 1. Background style selection by script tradition if not explicitly given
    if bg_type is None:
        if script == "devanagari":
            bg_type = local_rng.choices(["aged_paper", "parchment", "palm_leaf"], weights=[0.45, 0.40, 0.15])[0]
        elif script == "modi":
            bg_type = local_rng.choices(["aged_paper", "parchment", "palm_leaf"], weights=[0.50, 0.35, 0.15])[0]
        else:  # sharada
            bg_type = local_rng.choices(["palm_leaf", "parchment", "aged_paper"], weights=[0.45, 0.35, 0.20])[0]

    # 2. Appropriate historical folio aspect ratios
    is_palm_leaf = (bg_type == "palm_leaf")
    if is_palm_leaf:
        # Style B: Elongated horizontal palm-leaf proportion (3:1)
        width = 1800
        height = 600
    else:
        # Style A & C: Classical manuscript folio
        width = cfg.get("canvas", {}).get("width", 1600)
        height = cfg.get("canvas", {}).get("height", 1000)

    # 3. Sample generous authentic text passages from data/raw/ to naturally fill writing area
    raw_lines = text_loader.sample_passage(script, num_lines=local_rng.randint(35, 50), rng=local_rng)
    sec_lines = text_loader.sample_passage(script, num_lines=local_rng.randint(25, 40), rng=local_rng)

    # 4. Synthesize procedural background substrate
    bg_img = bg_generator.generate(
        bg_type=bg_type,
        size=(width, height),
        add_binding_hole=is_palm_leaf,
    )

    # 5. Compute traditional manuscript layout (naturally filling writing region)
    font_size = local_rng.choice([28, 30, 32])
    page_layout = layout_engine.compute_layout(
        canvas_size=(width, height),
        text_lines=raw_lines,
        font_size=font_size,
        layout_type=layout_type,
        secondary_lines=sec_lines,
        is_palm_leaf=is_palm_leaf,
        script=script,
    )

    # 6. Render handwritten calligraphy and rubricated punctuation
    ink_colors = {
        "devanagari": (26, 20, 18),   # Carbon soot black
        "modi": (43, 29, 12),         # Iron gall brown
        "sharada": (24, 20, 18),      # Deep Kashmiri lampblack
    }
    ink_color = ink_colors.get(script, (26, 20, 18))
    rendered_img, line_annotations, font_used = text_renderer.render_page(
        base_image=bg_img,
        page_layout=page_layout,
        script_name=script,
        font_size=font_size,
        ink_color=ink_color,
    )

    # 6. Apply physical manuscript aging and degradation
    final_img = augmentor.apply(rendered_img)

    # 7. Save Image
    os.makedirs(output_dir, exist_ok=True)
    img_filename = f"{sample_id}.png"
    img_path = os.path.join(output_dir, img_filename)
    final_img.save(img_path, format="PNG")

    # 8. Record and save paired .md annotation (MANDATORY)
    metadata = {
        "script": script,
        "font_used": os.path.basename(font_used),
        "background_type": bg_type,
        "layout_type": page_layout.layout_type,
        "font_size": font_size,
        "ink_color": list(ink_color),
        "source_file": cfg.get("paths", {}).get("scripts_data", {}).get(script),
        "total_lines_rendered": len(line_annotations),
    }
    manifest_filename = rel_img_path if rel_img_path else img_filename
    record = annotation_mgr.create_sample_record(
        image_id=sample_id,
        image_filename=manifest_filename,
        width=width,
        height=height,
        script=script,
        lines=line_annotations,
        metadata=metadata,
    )

    md_filename = f"{sample_id}.md"
    md_path = os.path.join(output_dir, md_filename)
    annotation_mgr.save_sample_markdown(record, md_path)

    return img_path, md_path, record


def main():
    parser = argparse.ArgumentParser(description="Synthetic Manuscript Generator Entry Point")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to YAML configuration file (default: config/config.yaml)",
    )
    parser.add_argument(
        "--visual-test",
        action="store_true",
        help="Generate 1 sample per script in data/output/samples/ for visual inspection",
    )
    parser.add_argument(
        "--single-sample",
        action="store_true",
        help="Generate a single specified sample",
    )
    parser.add_argument(
        "--script",
        default="devanagari",
        choices=["devanagari", "modi", "sharada"],
        help="Target script for single sample generation (default: devanagari)",
    )
    parser.add_argument(
        "--sample-id",
        default="sample_devanagari_002",
        help="Identifier for the single generated sample (default: sample_devanagari_002)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Display dataset generation plan without writing files",
    )
    parser.add_argument(
        "--output-dir",
        default="data/output",
        help="Root destination directory for dataset (default: data/output)",
    )

    args = parser.parse_args()

    print("==================================================")
    print("      SYNTHETIC MANUSCRIPT GENERATOR (v0.2.0)     ")
    print("==================================================")

    # 1. Load Configuration
    print(f"\n[1/4] Loading project configuration from '{args.config}'...")
    cfg = load_config(args.config)
    print("      Configuration loaded successfully.")

    # 2. Initialize and Verify Text Loader
    print("\n[2/4] Initializing TextLoader and inspecting raw source texts...")
    raw_paths = cfg.get("paths", {}).get("scripts_data", {})
    text_loader = TextLoader(raw_data_paths=raw_paths)
    stats = text_loader.get_stats()

    for script, data in stats.items():
        if "error" in data:
            print(f"      - {script.capitalize()}: ERROR ({data['error']})")
        else:
            print(f"      - {script.capitalize()}: {data['total_lines']} lines ({data['total_characters']:,} characters)")

    # 3. Instantiate Pipeline Modules
    print("\n[3/4] Initializing pipeline components...")
    seed = cfg.get("project", {}).get("seed", 42)
    bg_generator = BackgroundGenerator(seed=seed)
    layout_engine = LayoutEngine(config=cfg, seed=seed)
    text_renderer = TextRenderer(fonts_config=cfg.get("fonts", {}), seed=seed)
    augmentor = ManuscriptAugmentor(config=cfg.get("augmentations", {}), seed=seed)
    annotation_mgr = AnnotationManager(output_dir=args.output_dir)
    splitter = DatasetSplitter(
        train_ratio=cfg.get("generation", {}).get("split_ratio", {}).get("train", 0.85),
        val_ratio=cfg.get("generation", {}).get("split_ratio", {}).get("val", 0.10),
        test_ratio=cfg.get("generation", {}).get("split_ratio", {}).get("test", 0.05),
        split_counts=cfg.get("generation", {}).get("split_counts_per_script", {"train": 85, "val": 10, "test": 5}),
        seed=seed,
    )
    print("      All pipeline components initialized cleanly.")

    # Check for Single Sample Mode
    if args.single_sample:
        print(f"\n[4/4] Executing Single Sample Generation for '{args.script}'...")
        sample_dir = os.path.join(args.output_dir, "samples")
        img_path, md_path, _ = generate_script_sample(
            script=args.script,
            sample_id=args.sample_id,
            cfg=cfg,
            text_loader=text_loader,
            bg_generator=bg_generator,
            layout_engine=layout_engine,
            text_renderer=text_renderer,
            augmentor=augmentor,
            annotation_mgr=annotation_mgr,
            output_dir=sample_dir,
        )

        img_ok = os.path.exists(img_path) and os.path.getsize(img_path) > 0
        md_ok = os.path.exists(md_path) and os.path.getsize(md_path) > 0

        if img_ok and md_ok:
            print(f"        [OK] Image: {img_path} ({os.path.getsize(img_path):,} bytes)")
            print(f"        [OK] Paired .md: {md_path} ({os.path.getsize(md_path):,} bytes)")
        else:
            print(f"        [FAILED] Image OK: {img_ok}, MD OK: {md_ok}")

        print("\n==================================================")
        print(f" Single Sample Generated: {args.sample_id}")
        print(f" Script: {args.script}")
        print(f" Image: {img_path}")
        print(f" Annotation: {md_path}")
        print("==================================================")
        return

    # Check for Visual Test Mode (6 Folios: 2 Devanagari, 2 Modi, 2 Sharada)
    if args.visual_test:
        print("\n[4/4] Executing Six Visual Manuscript Tests...")
        sample_dir = os.path.join(args.output_dir, "samples")
        os.makedirs(sample_dir, exist_ok=True)
        generated_pairs = []

        test_configs = [
            # 1. Devanagari Style A: Aged Handmade Rag Paper, Classical Standard Folio
            ("devanagari", "visual_devanagari_folio1", "aged_paper", "standard"),
            # 2. Devanagari Style C: Aged Parchment, Multi-Block Layout with Divider
            ("devanagari", "visual_devanagari_folio2", "parchment", "multi_block"),
            # 3. Modi Style A: Aged Paper, Marginal Commentary Layout
            ("modi", "visual_modi_folio1", "aged_paper", "marginal_commentary"),
            # 4. Modi Style B: Palm-Leaf (Tala-Patra), Long Horizontal Folio
            ("modi", "visual_modi_folio2", "palm_leaf", "standard"),
            # 5. Sharada Style B: Palm-Leaf (Tala-Patra), Long Horizontal Folio
            ("sharada", "visual_sharada_folio1", "palm_leaf", "standard"),
            # 6. Sharada Style C: Aged Parchment, Multi-Block Sacred Text Layout
            ("sharada", "visual_sharada_folio2", "parchment", "multi_block"),
        ]

        test_rng = random.Random(42)
        for script, sample_id, bg_type, layout_type in test_configs:
            print(f"      Generating '{sample_id}' ({script}, {bg_type}, {layout_type})...")
            img_path, md_path, rec = generate_script_sample(
                script=script,
                sample_id=sample_id,
                cfg=cfg,
                text_loader=text_loader,
                bg_generator=bg_generator,
                layout_engine=layout_engine,
                text_renderer=text_renderer,
                augmentor=augmentor,
                annotation_mgr=annotation_mgr,
                output_dir=sample_dir,
                rng=test_rng,
                bg_type=bg_type,
                layout_type=layout_type,
            )
            generated_pairs.append((script, sample_id, bg_type, layout_type, img_path, md_path, len(rec["lines"])))

        print("\n==================================================")
        print(" Six Visual Manuscript Tests Completed Successfully:")
        for script, sid, bg, ltype, img, md, n_lines in generated_pairs:
            print(f"   * [{script.upper()}] {sid} | Substrate: {bg} | Layout: {ltype} | Lines: {n_lines}")
            print(f"     Image: {img}")
            print(f"     Annotation: {md}")
        print("==================================================")
        return

    # Dry-Run Mode
    if args.dry_run:
        gen_cfg = cfg.get("generation", {})
        total_target = gen_cfg.get("total_samples", 300)
        per_script = gen_cfg.get("samples_per_script", {})
        split_counts = gen_cfg.get("split_counts_per_script", {"train": 85, "val": 10, "test": 5})

        print("\n[4/4] Dataset Generation Plan (Dry Run):")
        print(f"      - Total Target: {total_target} manuscript images")
        print(f"      - Breakdown per script: {per_script}")
        print(f"      - Quotas per script: {split_counts['train']} train, {split_counts['val']} val, {split_counts['test']} test")
        print(f"      - Total across 3 scripts: {split_counts['train']*3} train, {split_counts['val']*3} val, {split_counts['test']*3} test")
        print("      - Annotations: Paired .md ground-truth file for EVERY image (MANDATORY)")
        print("\n[INFO] Dry run finished. Run without '--dry-run' to generate full dataset.")
        print("==================================================")
        return

    # 4. Full Dataset Generation Mode (300 Images: 100 Devanagari, 100 Modi, 100 Sharada)
    print("\n[4/4] Executing Full Dataset Generation (300 Images)...")
    scripts = ["devanagari", "modi", "sharada"]
    split_counts = cfg.get("generation", {}).get("split_counts_per_script", {"train": 85, "val": 10, "test": 5})

    splits_dict = {
        "train": [],
        "val": [],
        "test": [],
    }

    global_rng = random.Random(seed)

    for script in scripts:
        print(f"\n      Processing Script: '{script.upper()}' (100 folios)...")
        # Define sequential splits: 85 train, 10 val, 5 test
        split_plan = [
            ("train", "train", split_counts["train"], 1),
            ("val", "validation", split_counts["val"], split_counts["train"] + 1),
            ("test", "test", split_counts["test"], split_counts["train"] + split_counts["val"] + 1),
        ]

        for split_key, folder_name, count, start_num in split_plan:
            dest_dir = os.path.join(args.output_dir, script, folder_name)
            os.makedirs(dest_dir, exist_ok=True)
            print(f"        -> Generating {count} {split_key.upper()} samples in '{script}/{folder_name}/'...")

            for i in range(count):
                sample_num = start_num + i
                sample_id = f"Image_{sample_num:03d}"
                rel_img_path = f"{script}/{folder_name}/{sample_id}.png"

                img_path, md_path, rec = generate_script_sample(
                    script=script,
                    sample_id=sample_id,
                    cfg=cfg,
                    text_loader=text_loader,
                    bg_generator=bg_generator,
                    layout_engine=layout_engine,
                    text_renderer=text_renderer,
                    augmentor=augmentor,
                    annotation_mgr=annotation_mgr,
                    output_dir=dest_dir,
                    rel_img_path=rel_img_path,
                    rng=global_rng,
                )
                splits_dict[split_key].append(rec)

    # 5. Save Split Manifests (train.json/md, val.json/md, test.json/md)
    print(f"\n[5/5] Saving dataset split manifests to '{args.output_dir}'...")
    splitter.save_split_manifests(splits_dict, output_dir=args.output_dir)

    # Also save validation.json and validation.md as convenience aliases
    val_json_path = os.path.join(args.output_dir, "val.json")
    validation_json_path = os.path.join(args.output_dir, "validation.json")
    if os.path.exists(val_json_path):
        with open(val_json_path, "r", encoding="utf-8") as f_src, open(validation_json_path, "w", encoding="utf-8") as f_dst:
            f_dst.write(f_src.read())

    print("\n==================================================")
    print("      DATASET GENERATION COMPLETED SUCCESSFULLY   ")
    print("==================================================")
    print(f" Total Samples Generated: {len(splits_dict['train']) + len(splits_dict['val']) + len(splits_dict['test'])}")
    print(f" Train: {len(splits_dict['train'])} (85 Devanagari, 85 Modi, 85 Sharada)")
    print(f" Validation: {len(splits_dict['val'])} (10 Devanagari, 10 Modi, 10 Sharada)")
    print(f" Test: {len(splits_dict['test'])} (5 Devanagari, 5 Modi, 5 Sharada)")
    print(f" Root Directory: {args.output_dir}")
    print("==================================================")


if __name__ == "__main__":
    main()
