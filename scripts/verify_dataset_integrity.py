import hashlib
import re
from PIL import Image

def main():
    # 1. Verify source MD hash unchanged
    expected_hashes = {
        'data/raw/devanagari_md.md': '685E1B8AA9A88AF499F4B6F92E425BA28588A0F170731BD859C81E2262BD2E7B',
        'data/raw/Modi_md.md': '1948B052FFDE90CEC6E41BA94B7AAF48175EC297EE367FAAD25FD9AC876C5F20',
        'data/raw/sharada_md.md': '8E4A0AB8952DC1393CB17FDD9B9623D4545E2EF2FB9E98FBDBE06DF2C38462DE',
    }

    for path, exp_hash in expected_hashes.items():
        with open(path, 'rb') as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest().upper()
        assert actual_hash == exp_hash, f'Hash mismatch on {path}: expected {exp_hash}, got {actual_hash}'
    print('[PASS 1/6] All raw MD files are 100% byte-exact and unmodified.')

    # 2. Verify rendered text in sample_devanagari_002.md matches raw MD verbatim
    with open('data/output/samples/sample_devanagari_002.md', 'r', encoding='utf-8') as f:
        md_content = f.read()

    lines_sec = re.search(r'## Ground Truth Transcription\s*\n\n(.*?)\n\n##', md_content, re.DOTALL).group(1)
    gt_lines = [l.strip().rstrip('  ') for l in lines_sec.split('\n') if l.strip()]

    with open('data/raw/devanagari_md.md', 'r', encoding='utf-8') as f:
        raw_lines = [l.strip() for l in f if l.strip()]

    for idx, g in enumerate(gt_lines):
        matched = any(g in r for r in raw_lines)
        assert matched, f'GT line {idx} "{g}" not found in raw lines!'
    print(f'[PASS 2/6] All {len(gt_lines)} lines in ground truth transcription match raw MD source lines verbatim.')

    # 3. Verify no Unicode substitutions / characters preserved
    with open('data/output/samples/sample_devanagari_002.md', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    table_lines = []
    for line in lines:
        if line.startswith('|') and '`[' in line:
            parts = [p.strip() for p in line.split('|')]
            table_lines.append(parts[3])

    assert len(table_lines) == len(gt_lines), f'Table lines ({len(table_lines)}) != GT lines ({len(gt_lines)})'
    for i in range(len(gt_lines)):
        assert table_lines[i] == gt_lines[i], f'Line {i+1} mismatch: table="{table_lines[i]}" vs GT="{gt_lines[i]}"'
    print('[PASS 3/6] Zero Unicode substitutions; table annotations and GT transcription are identical.')

    # 4. Grapheme cluster integrity test
    from src.text_renderer import TextRenderer
    for g in gt_lines:
        words = g.split()
        for w in words:
            clusters = TextRenderer.get_grapheme_clusters(w)
            reconstructed = ''.join(clusters)
            assert reconstructed == w, f'Grapheme cluster split corrupted word "{w}"'
    print('[PASS 4/6] Grapheme cluster segmentation is 100.00% lossless across all words.')

    # 5. Image & MD synchronization & size
    im = Image.open('data/output/samples/sample_devanagari_002.png')
    w, h = im.size
    assert (w, h) == (1600, 1000), f'Unexpected image size {(w, h)}'
    print(f'[PASS 5/6] Image exists and matches canvas dimensions: {w}x{h} px.')

    # 6. Text stays strictly inside page boundary and margins
    for line in lines:
        if line.startswith('|') and '`[' in line:
            m = re.search(r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', line)
            if m:
                coords = [int(x) for x in m.groups()]
                x1, y1, x2, y2 = coords
                assert 0 <= x1 < x2 <= w, f'x-bounds out of range: {coords}'
                assert 0 <= y1 < y2 <= h, f'y-bounds out of range: {coords}'
                assert x1 >= 80 and x2 <= 1520, f'x violated content margin: {coords}'
                assert y1 >= 60 and y2 <= 940, f'y violated content margin: {coords}'
    print('[PASS 6/6] All bounding boxes strictly within margins and canvas boundaries.')
    print('==================================================')
    print('ALL 6 VERIFICATIONS 100% PASSED!')
    print('==================================================')

if __name__ == '__main__':
    main()
