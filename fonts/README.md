# Fonts Directory

Place TrueType (.ttf) or OpenType (.otf) font files here for the three supported Indic scripts:

## Recommended Fonts

### 1. Devanagari (`fonts/devanagari/`)
- `NotoSansDevanagari-Regular.ttf` (Google Noto Fonts)
- `YatraOne-Regular.ttf` (Stylized calligraphy manuscript look)
- `Lohit-Devanagari.ttf`

### 2. Modi Script (`fonts/modi/`)
- `NotoSansModi-Regular.ttf` (Google Noto Fonts)

### 3. Sharada Script (`fonts/sharada/`)
- `NotoSansSharada-Regular.ttf` (Google Noto Fonts)

> **Note**: The font paths can be configured directly in `config/config.yaml`.
> When no specific font file is found, `src/text_renderer.py` gracefully falls back to default system rendering during initial dry-runs.
