#!/usr/bin/env python3
"""Remove only selected caption-like titles from GMT vector exports.

Keep axes, legends, metric identifiers, sample sizes and panel letters.
"""
from pathlib import Path
from collections import Counter
import sys
import pymupdf as f

B = Path(__file__).resolve().parents[1]
TITLES = {
    '2': ['Event centroids', 'Broad / severe / major hierarchy',
          'Event descriptors', 'Unique-event timing'],
    '5': ['Focal absolute differences', 'Focal normalized differences'],
    '6': ['Five prespecified regions', 'Diagnostics with regional support',
          'Primary wind contrasts'],
    'S1': ['Northern Hemisphere annual context', 'Detection example'],
    'S4': ['Focal-strength sensitivity;'],
}

def geometry(page):
    return [{k:v for k,v in d.items() if k != 'seqno'} for d in page.get_drawings()]

def clean(key):
    if key not in TITLES: return
    path = B / 'full_rendered' / f'Figure_{key}.pdf'
    doc = f.open(path); page = doc[0]
    original_geometry = geometry(page)
    original_words = Counter(w[4] for w in page.get_text('words'))
    removed = Counter(); rectangles = []
    for title in TITLES[key]:
        hits = page.search_for(title)
        if not hits: continue  # Idempotent on already cleaned exports.
        assert len(hits) == 1, (key, title, hits)
        rectangles.extend(hits); removed.update(title.split())
    if not rectangles:
        doc.close(); print(f'Figure {key}: already title-clean', flush=True); return
    for rect in rectangles: page.add_redact_annot(rect, fill=None, cross_out=False)
    page.apply_redactions(images=0, graphics=0, text=0)
    assert geometry(page) == original_geometry, 'Graphic geometry changed'
    assert Counter(w[4] for w in page.get_text('words')) == original_words - removed, (key, 'Unexpected text change')
    assert not any(page.search_for(t) for t in TITLES[key])
    payload = doc.tobytes(garbage=4, deflate=True)
    doc.close(); path.write_bytes(payload)
    with f.open(path) as output:
        output[0].get_pixmap(dpi=600, alpha=False).save(path.with_suffix('.png'))
        path.with_suffix('.svg').write_text(output[0].get_svg_image(text_as_path=False))
    print(f'Figure {key}: {len(rectangles)} titles removed; other text and paths unchanged', flush=True)

