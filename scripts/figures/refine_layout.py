#!/usr/bin/env python3
"""Reflow existing GMT vector panels; no data plotting or inference.

Trim empty title bands, relocate four Figure 5 annotations to footers, keep the
S4 sample size inside the upper-left frame, and pack panels with label shelves.
Every map and data plot is translated/uniformly scaled only.
"""
from pathlib import Path
from collections import Counter
import json,sys,subprocess,shutil
import pymupdf as f

B=Path(__file__).resolve().parents[1]
OUT=B/'layout_refined'
KEYS=['2','5','6','S1','S2','S3','S4']
PAD=3.; LABEL=15.; GAP=15.; ROW_GAP=16.
MOVE={}  # Filled from current input labels by the entrypoint.
S4_NOTE=None


def words(page):
    return Counter(w[4] for w in page.get_text('words') if w[4].strip())

def bounds(page):
    # Raster inspection determines only empty margins; exported art stays vector.
    pix=page.get_pixmap(matrix=f.Matrix(2,2),colorspace=f.csGRAY,alpha=False)
    mask=pix.samples.translate(bytes(1 if v<249 else 0 for v in range(256)))
    rows=[(y,mask[y*pix.width:(y+1)*pix.width]) for y in range(pix.height)]
    rows=[(y,row) for y,row in rows if b'\x01' in row]
    x0=min(row.find(b'\x01') for _,row in rows);x1=max(row.rfind(b'\x01') for _,row in rows)
    r=f.Rect(x0/2,rows[0][0]/2,(x1+1)/2,(rows[-1][0]+1)/2)
    for w in page.get_text('words'):
        if w[4].strip(): r|=f.Rect(w[:4])
    r=f.Rect(r.x0-1,r.y0-1,r.x1+1,r.y1+1)&page.rect
    return r

def output(doc,stem,png=True):
    stem.parent.mkdir(parents=True,exist_ok=True)
    doc.save(stem.with_suffix('.pdf'),garbage=4,deflate=True)
    stem.with_suffix('.svg').write_text(doc[0].get_svg_image(text_as_path=False))
    if png: doc[0].get_pixmap(dpi=600,alpha=False).save(stem.with_suffix('.png'))

def panel(key,lab,src,stem,png):
    original=f.open(src);doc=f.open(src);page=doc[0];before=words(page)
    annotation=MOVE.get((key,lab));r=None
    translations=[]
    if (key,lab)==('5','b'):
        translations=[('Primary — absolute loss',0,-9.6),('Secondary — normalized loss',0,-9.6)]
    moved=[]
    for phrase,dx,dy in translations:
        hits=page.search_for(phrase);assert len(hits)==1,(key,lab,phrase)
        rr=hits[0];page.add_redact_annot(rr,fill=None,cross_out=False)
        moved.append((f.Rect(rr.x0-.15,rr.y0-.15,rr.x1+.15,rr.y1+.15),dx,dy))
    if annotation:
        found=page.search_for(annotation)
        assert len(found)==1,(key,lab,annotation,found)
        r=found[0];page.add_redact_annot(r,fill=None,cross_out=False)
    axis=None;inside_note=None
    if (key,lab)==('5','a'):
        axis=page.search_for('Event-start (X1)')[0]
        page.add_redact_annot(axis,fill=None,cross_out=False)
    if (key,lab)==('S4','whole'):
        inside_note=page.search_for(S4_NOTE)[0]
        page.add_redact_annot(inside_note,fill=None,cross_out=False)
    if moved or r or axis or inside_note:
        page.apply_redactions(images=0,graphics=0,text=0)
    for rr,dx,dy in moved:
        page.show_pdf_page(rr+f.Rect(dx,dy,dx,dy),original,0,clip=rr)
    if axis:
        page.insert_text((axis.x0+4,axis.y1),'Event-start (X1)',rotate=90,
                         fontsize=9.3,fontname='helv',color=(.2,.2,.2))
    if inside_note:
        frame=[d['rect'] for d in page.get_drawings()
               if d['rect'].width>300 and d['rect'].height<.1
               and page.rect.contains(d['rect'])]
        top=min(frame,key=lambda z:z.y0)
        page.insert_text((top.x0+7,top.y0+12),S4_NOTE,fontsize=10,
                         fontname='hebo',color=(.15,.15,.15))
    clip=bounds(page)
    if key=='5' and lab in ['a','b']:
        clip.y0=40.5;clip.y1=175.1 # same data-row and annotation baseline
    if key=='5' and lab in ['c','d']:
        clip.y0=47.;clip.y1=241. # common physical plot top/bottom, not ink extrema
    footer=(r.height+9) if r else 0
    width=max(clip.width,(r.width if r else 0))+2*PAD
    result=f.open();q=result.new_page(width=width,height=clip.height+2*PAD+footer)
    x=PAD+(width-2*PAD-clip.width)/2
    q.show_pdf_page(f.Rect(x,PAD,x+clip.width,PAD+clip.height),doc,0,clip=clip)
    if r:
        rr=f.Rect(r.x0-.15,r.y0-.15,r.x1+.15,r.y1+.15)
        x=(width-rr.width)/2;y=PAD+clip.height+7
        q.show_pdf_page(f.Rect(x,y,x+rr.width,y+rr.height),original,0,clip=rr)
    assert words(q)==before,(key,lab,'Text lost or duplicated',words(q)-before,before-words(q))
    assert not q.get_images(),(key,lab,'Unexpected raster')
    output(result,stem,png)
    info=dict(panel=lab,file=str(stem.relative_to(B)),width=width,height=q.rect.height,
              input_width=original[0].rect.width,input_height=original[0].rect.height,
              moved_annotation=annotation,source_clip=list(clip))
    original.close();doc.close();result.close()
    return info

def grid(rows,cols=None,gap=GAP,rowgap=ROW_GAP):
    items=[];labels=[];y=0.;width=0.
    for row in rows:
        widths=cols or [p['width'] for p in row]
        x=0.;h=max(p['height'] for p in row)
        for i,p in enumerate(row):
            item=dict(p,x=x+(widths[i]-p['width'])/2,y=y+LABEL)
            items.append(item);labels.append(dict(text='('+p['panel']+')',x=x,y=y))
            x+=widths[i]+gap
        width=max(width,x-gap);y+=LABEL+h+rowgap
    return items,labels,width,y-rowgap

def assemble(key,panels,png):
    p={p['panel']:p for p in panels}
    if key in ['S4']:
        one=panels[0];items=[dict(one,x=0,y=0)];labels=[];w=one['width'];h=one['height']
    elif key=='2':
        col=max(p['a']['width'],p['b']['width'],p['c']['width'],p['d']['width'])
        items,labels,w,h=grid([[p['a'],p['b']],[p['c'],p['d']]],[col,col],gap=18)
    elif key in ['6']:
        items,labels,w,h=grid([[p['a'],p['b']]])
        w=max(w,p['c']['width']);topwidth=items[-1]['x']+items[-1]['width']
        for i in items:i['x']+=(w-topwidth)/2
        labels[1]['x']=items[1]['x']
        y=h+ROW_GAP
        items.append(dict(p['c'],x=(w-p['c']['width'])/2,y=y+LABEL));labels.append(dict(text='(c)',x=0,y=y))
        h=y+LABEL+p['c']['height']
    elif key=='5':
        items,labels,w,h=grid([[p['a'],p['b']]],gap=18)
        bottomwidth=p['c']['width']+p['d']['width']+24;w=max(w,bottomwidth)
        y=h+ROW_GAP
        # Equal plot-width, equal source font scale, shared row baseline.
        x=(w-bottomwidth)/2
        for z in [p['c'],p['d']]:
            items.append(dict(z,x=x,y=y+LABEL));labels.append(dict(text='('+z['panel']+')',x=x,y=y));x+=z['width']+24
        h=y+LABEL+max(p['c']['height'],p['d']['height'])
    elif key=='S1':items,labels,w,h=grid([[p['a'],p['b']]],gap=18)
    else:items,labels,w,h=grid([[z] for z in panels],rowgap=12)
    doc=f.open();page=doc.new_page(width=w+2*PAD,height=h+2*PAD)
    expected=Counter()
    for item in items:
        item['x']+=PAD;item['y']+=PAD
        with f.open(B/(item['file']+'.pdf')) as d:
            expected.update(words(d[0]));r=f.Rect(item['x'],item['y'],item['x']+item['width'],item['y']+item['height'])
            page.show_pdf_page(r,d,0)
    for label in labels:
        label['x']+=PAD;label['y']+=PAD
        page.insert_text((label['x'],label['y']+9.5),label['text'],fontname='hebo',fontsize=10,color=(.1,.1,.1))
        expected.update([label['text']])
    assert words(page)==expected,(key,'Assembly text mismatch')
    output(doc,OUT/'figures'/f'Figure_{key}',png)
    rec=dict(figure=key,width=page.rect.width,height=page.rect.height,items=items,labels=labels)
    (OUT/'figures'/f'Figure_{key}_layout.json').write_text(json.dumps(rec,indent=2)+'\n')
    print(f'Figure {key}: reflowed {len(items)} units; {page.rect.width:.1f} x {page.rect.height:.1f} pt',flush=True)
    doc.close()
