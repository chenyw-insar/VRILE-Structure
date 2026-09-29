#!/usr/bin/env python3
"""Vector panel exports from the exact adopted GMT PDFs, without panel letters.

Usage: python -B scripts/export_panels.py [all|2|...|S4]
This is vector extraction, NOT a new scientific plot or bitmap crop. Plot marks,
axes, legends, scales and scientific labels are retained. Caption-like titles
are removed upstream; isolated (a), (b), ... glyphs are deleted here.
Numbering is added only by the whole-figure vector assembler.
"""
from pathlib import Path
import re,json,sys
from collections import Counter
import pymupdf as f
from remove_caption_titles import clean as remove_caption_titles

B=Path(__file__).resolve().parents[1]
KEYS=['2','5','6','S1','S2','S3','S4']
PAD=3.0

def rectangles(key,w,h,page):
    if key in ['S4']:return [('whole',(0,0,w,h))]
    if key=='2':return [('a',(0,0,223,245)),('b',(223,0,w,245)),('c',(0,247,223,h)),('d',(223,247,w,h))]
    if key=='5':return [('a',(0,0,141,178)),('b',(141,0,w,178)),('c',(0,180,228,h)),('d',(228,180,w,h))]
    if key=='6':return [('a',(0,0,231,275)),('b',(231,0,w,275)),('c',(0,275,w,h))]
    if key=='S1':return [('a',(0,0,266,202.32)),('b',(267,0,w,h))]
    if key=='S2':return [('a',(0,0,w,193)),('b',(0,197,w,h))]
    if key=='S3':
        med=sorted(r.y0 for r in page.search_for('median ='))
        assert len(med)==3
        a,b=med[1]-2,med[2]-2
        return [('a',(0,0,w,a)),('b',(0,a,w,b)),('c',(0,b,w,h))]
    raise KeyError(key)

def save_panel(doc,rect,path):
    r=f.Rect(rect);out=f.open();p=out.new_page(width=r.width+2*PAD,height=r.height+2*PAD)
    p.show_pdf_page(f.Rect(PAD,PAD,r.width+PAD,r.height+PAD),doc,0,clip=r)
    assert not p.get_images()
    out.save(path.with_suffix('.pdf'),garbage=4,deflate=True)
    path.with_suffix('.svg').write_text(p.get_svg_image(text_as_path=False))
    p.get_pixmap(dpi=600,alpha=False).save(path.with_suffix('.png'))
    out.close()

def export(key):
    remove_caption_titles(key)
    src=B/'full_rendered'/f'Figure_{key}.pdf'
    # Same accepted S3 script and input; only PDF crop margin was widened to
    # retain ticks whose glyph bounds extended outside the old zero-margin page.
    assert src.exists(),'Missing current GMT intermediate PDF'
    doc=f.open(src);p=doc[0]
    labels=[]
    before_drawings=p.get_drawings()
    original_words=Counter(w[4] for w in p.get_text('words') if not re.fullmatch(r'\([a-z]\)',w[4]))
    for word in p.get_text('words'):
        if re.fullmatch(r'\([a-z]\)',word[4]):
            r=f.Rect(word[:4]);labels.append(dict(text=word[4],rect=list(r)))
            p.add_redact_annot(r,fill=None,cross_out=False)
    if labels:p.apply_redactions(images=0,graphics=0,text=0)
    # Text deletion changes display-list sequence indices, not graphic paths.
    def geometry(drawings):
        return [{k:v for k,v in d.items() if k!='seqno'} for d in drawings]
    assert geometry(p.get_drawings())==geometry(before_drawings),'Only panel-letter text may change'
    assert Counter(w[4] for w in p.get_text('words'))==original_words,'Scientific text changed'
    assert not any(re.fullmatch(r'\([a-z]\)',w[4]) for w in p.get_text('words'))
    rr=rectangles(key,p.rect.width,p.rect.height,p)
    # Catch accidentally clipped scientific words, including rotated axis text.
    missing=[]
    for word in p.get_text('words'):
        r=f.Rect(word[:4]);ok=False
        for _,rect in rr:
            outer=f.Rect(rect);outer.x0-=.6;outer.y0-=.6;outer.x1+=.6;outer.y1+=.6
            if outer.contains(r):ok=True;break
        if not ok:missing.append((word[4],list(r)))
    assert not missing,(key,'Words outside panel crops',missing)
    dest=B/'panels'/f'Figure_{key}';dest.mkdir(parents=True,exist_ok=True)
    rows=[]
    for label,rect in rr:
        name=f'Figure_{key}'+('' if label=='whole' else label)+'_clean'
        save_panel(doc,rect,dest/name)
        rows.append(dict(panel=label,rect=list(rect),file=str((dest/name).relative_to(B))))
    record=dict(figure=key,source_pdf=str(src.relative_to(B)),width=p.rect.width,height=p.rect.height,
                removed_letters=labels,panels=rows,graphic_paths_preserved=True,
                scientific_text_coverage='PASS',png_dpi=600)
    (dest/'panel_layout.json').write_text(json.dumps(record,indent=2)+'\n')
    print(f'Figure {key}: {len(rows)} clean vector panels; scientific text/paths retained',flush=True)
    doc.close()

