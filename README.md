# Synthetic Indic Manuscript Generator

An end-to-end procedural synthesis pipeline for generating authentic historical Indic manuscript folios in **Devanagari**, **Modi**, and **Sharada** scripts. The pipeline simulates calligraphic reed-pen writing, multi-tonal manuscript inks, historical aged paper and palm-leaf substrates, traditional scribal layouts, physical folio degradation, and exact synchronized ground-truth Markdown annotations.

---

## 1. Project Purpose & Technical Objectives

Historical Indic manuscripts represent centuries of intellectual heritage across diverse writing traditions. However, training robust Document Layout Analysis (DLA) and Optical Character Recognition (OCR) models on historical Indic texts is hindered by a scarcity of high-quality annotated training data.

This project delivers a synthetic manuscript generator that bridges this domain gap by generating photorealistic manuscript folios adhering to historical scribal practices, physical material science, and strict OCR ground-truth fidelity:
- **Strict Source Ground Truth**: 100% of rendered text originates directly from unmodified historical source corpora without character invention, substitution, or normalization.
- **Calligraphic Writing Realism**: Replaces sterile digital typography with simulated reed-pen (*kalam*) dynamics, baseline undulation, pressure-dependent stroke widths, and intact connected *shirorekha* headlines.
- **Historical Materiality**: Procedurally synthesizes aged handmade rag paper, palm-leaf (*tala-patra*) folios with binding holes, and parchment substrates with longitudinal fibers, foxing, water stains, and 3D folding creases.
- **Traditional Layout Systems**: Supports single-column folios, multi-block structured texts, and marginal gloss (*tippani*) commentary layouts with faint lead-point ruling guides.
- **Exact Paired Annotations**: Every generated image is accompanied by an identical-name `.md` ground-truth file tracking full transcriptions, layout metadata, and pixel-accurate line bounding boxes.

---

## 2. Supported Scripts & Historical Sources

All text content is extracted directly from the raw Markdown files in `data/raw/`. Under the **Absolute Source-Text Rule**, these files are strictly immutable:

| Script | Source File | Tradition & Description | Typographic & Scribal Configuration |
| :--- | :--- | :--- | :--- |
| **Devanagari** | `data/raw/devanagari_md.md` | Classical Sanskrit and Hindi literature. | Uses historical *Jaini Purva* manuscript font and cursive *Tillana* scribal hand; preserves connected *shirorekha*; vermilion rubrication on dandas (`।`, `॥`). |
| **Modi** | `data/raw/Modi_md.md` | Historical Marathi administrative and epistolary script. | Cursive baseline flow, continuous stroke momentum, administrative margin commentary layouts. |
| **Sharada** | `data/raw/sharada_md.md` | Classical Kashmiri sacred and philosophical manuscripts. | Sacred northern Indic folio proportions, birch-bark/parchment styling, Kashmiri soot-black ink. |

---

## 3. Dataset Structure & Stratified Splits

The pipeline synthesizes exactly **300 manuscript images** (100 per script) divided into a stratified 85% train / 10% validation / 5% test split:

```text
data/output/
├── devanagari/
│   ├── train/          # 85 folios (Image_001.png - Image_085.png + paired .md)
│   ├── validation/     # 10 folios (Image_086.png - Image_095.png + paired .md)
│   └── test/           #  5 folios (Image_096.png - Image_100.png + paired .md)
├── modi/
│   ├── train/          # 85 folios (Image_001.png - Image_085.png + paired .md)
│   ├── validation/     # 10 folios (Image_086.png - Image_095.png + paired .md)
│   └── test/           #  5 folios (Image_096.png - Image_100.png + paired .md)
├── sharada/
│   ├── train/          # 85 folios (Image_001.png - Image_085.png + paired .md)
│   ├── validation/     # 10 folios (Image_086.png - Image_095.png + paired .md)
│   └── test/           #  5 folios (Image_096.png - Image_100.png + paired .md)
├── train.json          # Complete COCO-style manifest for 255 train samples
├── val.json            # Complete COCO-style manifest for 30 validation samples
└── test.json           # Complete COCO-style manifest for 15 test samples
```

### Paired Markdown Annotation Schema
For every `Image_XXX.png`, an identical `Image_XXX.md` is generated containing:
1. **Frontmatter Metadata**: Script, folio dimensions, substrate style, layout pattern, scribe profile, and aging tier.
2. **Full Transcription**: Contiguous Unicode text verbatim from the raw source.
3. **Line-Level Bounding Boxes**: Normalized pixel bounding boxes `[x1, y1, x2, y2]` corresponding to the exact non-zero ink positions on the canvas.

---

## 4. Pipeline Architecture & Modular Components

```text
synthetic-manuscript-generator/
├── generate.py                 # Master orchestrator & CLI
├── requirements.txt            # Core dependencies (Pillow, OpenCV, NumPy, PyYAML)
├── config/
│   └── config.yaml             # Central configuration (dimensions, layouts, styles)
├── src/
│   ├── text_loader.py          # Pure raw text extraction & contiguous block sampling
│   ├── background_generator.py # Procedural texture synthesis (paper, palm-leaf, vellum)
│   ├── layout_engine.py        # Manuscript geometry, margins, ruling guides, gloss blocks
│   ├── text_renderer.py        # Reed-pen mechanics, ink palettes, rubrication, bboxes
│   ├── augmentations.py        # 3D crease folds, page curling, capillary ink bleed, stains
│   ├── annotations.py          # Ground-truth .md and JSON manifest generation
│   └── dataset_splitter.py     # Partitioning & indexing across train/val/test splits
├── scripts/
│   └── validate_dataset.py     # 15-point dataset validation & integrity verification
└── fonts/                      # Historical TTF/OTF fonts
```

### Module Responsibilities

1. **`text_loader.py`**:
   - Reads `data/raw/*.md` without modifying Unicode codepoints.
   - Splits text into paragraphs and contiguous line chunks without token invention.
2. **`background_generator.py`**:
   - **Style A (Aged Handmade Paper)**: Fibrous pulp textures, warm amber hues, organic edge vignettes, and foxing stains.
   - **Style B (Palm-Leaf Folio)**: Elongated horizontal proportions (e.g., 1800×650), longitudinal leaf grain striations, darkened natural edges, and traditional cord binding holes (*pothi* cord piercings).
   - **Style C (Aged Manuscript Folio)**: Vellum/parchment tone, darkened borders, and subtle surface warping.
3. **`layout_engine.py`**:
   - **Standard Layout**: Central primary text block with generous scribal margins.
   - **Multi-Block Layout**: Primary text block accompanied by a secondary source excerpt separated by ornamental manuscript rules.
   - **Marginal Gloss Layout**: Core text flanked by side commentary blocks (*hashiya* or *tippani*).
   - Generates coordinate grids for faint horizontal writing guidelines.
4. **`text_renderer.py`**:
   - Simulates reed-pen (*kalam*) dynamics: angled stroke pressure modulation (45° cut nib) creating thick downstrokes and thin cross-strokes.
   - Applies organic baseline undulation while maintaining intact *shirorekha* connectivity.
   - Multi-tonal ink palettes: Indian Carbon Soot, Iron Gall Brown, Kashmiri Lampblack, and Vermilion/Cinnabar Red for punctuation rubrication (`।`, `॥`).
   - Extracts exact line bounding boxes from non-zero alpha ink masks.
5. **`augmentations.py`**:
   - Tiered physical degradation: Clean (15%), Moderate (55%), Heavy (30%).
   - 3D physical folding creases with paired specular highlights and cast shadows.
   - Corner/edge page curl gradient shading.
   - Capillary ink feathering and paper bleed-through.

---

## 5. Installation & Setup

### Requirements
- Python 3.10+
- OS: Windows, Linux, or macOS

### Step 1: Create Virtual Environment
```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On Linux/macOS:
source .venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 6. Running the Generator

### Full Dataset Generation (300 Folios)
Generates the complete 300-image dataset (100 Devanagari, 100 Modi, 100 Sharada) organized into `data/output/`:
```bash
python generate.py
```

### Dry Run (Plan Verification)
Verifies configuration files, font paths, and text extraction without rendering:
```bash
python generate.py --dry-run
```

### Visual Test Mode (1 Sample Per Script)
Generates one test folio per script into `data/output/samples/` for rapid inspection:
```bash
python generate.py --visual-test
```

### Single Sample Generation
Generate a specific script sample with custom ID:
```bash
python generate.py --single-sample --script devanagari --sample-id sample_devanagari_custom
```

---

## 7. Dataset Validation

Run the comprehensive 15-point dataset validation suite:
```bash
python scripts/validate_dataset.py --output-dir data/output
```

The validator verifies:
1. File existence and image integrity for all 300 folios.
2. Mandatory paired `.md` annotation file exists and is non-empty for every image.
3. Canvas dimensions match metadata records.
4. Script class balance strictly matches quotas (85 train, 10 val, 5 test per script).
5. All bounding boxes have positive area and are within canvas boundaries `[0, W]` and `[0, H]`.
6. Transcriptions contain valid, non-empty Unicode strings.

---

## 8. Hugging Face Dataset Formatting

To package the generated dataset for the Hugging Face Hub:

```python
from datasets import Dataset, Features, Image, Value, Sequence

# Load from generated manifests
import json

with open("data/output/train.json", "r", encoding="utf-8") as f:
    train_records = json.load(f)

# Reformat for Hugging Face
hf_data = []
for rec in train_records:
    hf_data.append({
        "image": f"data/output/{rec['file_name']}",
        "script": rec["script"],
        "text": rec["text"],
        "style": rec["style"],
        "layout": rec["layout"],
        "lines": rec["lines"],
    })

dataset = Dataset.from_list(hf_data)
# dataset.push_to_hub("Sanskruti05/synthetic-indic-manuscripts")
```

---

## 9. Key Design Decisions

1. **Intact Shirorekha Connectivity**: Breaking words into independently rotated characters destroys the continuous Devanagari headline. Our approach applies smooth baseline perturbation and angled nib pressure variations to whole words, preserving script integrity while achieving handwritten rhythm.
2. **Historical Rubrication**: In Indian manuscript traditions, scribes reserved red cinnabar/vermilion ink for verse terminators (*dandas* `।` and double *dandas* `॥`). We isolate punctuation marks into a rubrication channel rendered in `(155, 34, 28)` without modifying raw source text.
3. **Exact Bounding Box Mapping**: Rather than estimating typography bounds using digital font metrics, bounding boxes are computed directly from the rendered ink mask, guaranteeing 100% alignment with the physical image.
