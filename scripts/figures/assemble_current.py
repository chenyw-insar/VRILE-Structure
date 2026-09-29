"""Approved F3/F4 vector assembly; letters are added to whole figures only."""
from pathlib import Path
import re
import pymupdf as pdf

MM=72/25.4

def emit(doc, stem):
    stem.parent.mkdir(parents=True,exist_ok=True)
    if len(doc)!=1 or doc[0].get_images(): raise ValueError('Expected single-page vector scientific figure')
    stem.with_suffix('.pdf').write_bytes(doc.tobytes(garbage=4,deflate=True))
    stem.with_suffix('.svg').write_text(doc[0].get_svg_image(text_as_path=False))
    doc[0].get_pixmap(dpi=600,alpha=False).save(stem.with_suffix('.png'))

def export(src,stem):
    with pdf.open(src) as doc:
        if re.search(r'\([a-d]\)',doc[0].get_text()): raise ValueError('Panel letters found in clean panel')
        emit(doc,stem)

def put(page,path,x,y,width):
    with pdf.open(path) as doc:
        height=width*doc[0].rect.height/doc[0].rect.width
        page.show_pdf_page(pdf.Rect(x,y,x+width,y+height),doc,0)

def assemble(key, gmt, out):
    target=out/'panels'/f'Figure_{key}'
    if key=='3':
        for lab in 'abcd': export(gmt/f'Figure_3{lab}_compact_aligned_clean.pdf',target/f'Figure_3{lab}_clean')
        doc=pdf.open();page=doc.new_page(width=163.5*MM,height=120*MM)
        for lab,(x,y,w) in {'a':(0,0,72),'b':(78.5,0,85),'c':(78.5,61,40),'d':(123.5,61,40)}.items():
            put(page,target/f'Figure_3{lab}_clean.pdf',x*MM,y*MM,w*MM)
            page.insert_text(((x+.4)*MM,(y+4.4)*MM),f'({lab})',fontname='hebo',fontsize=10.5,color=(.10,.14,.18))
    elif key=='4':
        for lab in 'ab': export(gmt/f'Figure_4{lab}_paired_delta_clean.pdf',target/f'Figure_4{lab}_clean')
        export(gmt/'Figure_4c_clean.pdf',target/'Figure_4c_clean')
        export(gmt/'Figure_4_primary_legend.pdf',target/'shared_primary_legend')
        export(gmt/'Figure_4_paired_reference.pdf',target/'paired_reference_note')
        width=163.5*MM;top=15;heat_h=103*MM;legend_h=12*MM
        ly=top+heat_h+2;cy=ly+legend_h+9+16;h=cy+54.5*MM+2
        doc=pdf.open();page=doc.new_page(width=width,height=h)
        for lab,x,y in [('a',3,0),('b',83.5*MM,0),('c',3,cy-16)]:
            page.insert_text((x,y+10),f'({lab})',fontsize=10.5,fontname='hebo',color=(.10,.14,.18))
        put(page,target/'Figure_4a_clean.pdf',0,top,82*MM)
        put(page,target/'Figure_4b_clean.pdf',83.5*MM,top,80*MM)
        put(page,target/'shared_primary_legend.pdf',0,ly,width)
        put(page,target/'paired_reference_note.pdf',0,ly+6.5*MM,width)
        put(page,target/'Figure_4c_clean.pdf',0,cy,width)
    else: raise ValueError(key)
    emit(doc,out/'figures'/f'Figure_{key}');doc.close()
