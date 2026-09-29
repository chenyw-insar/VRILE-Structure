#!/usr/bin/env python3
"""Check this run's figure files, vector formats and panels; no science calls."""
from pathlib import Path
import argparse, hashlib, json, re, sys
from prepare_inputs import KEYS

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output_dir',type=Path,required=True)
    p.add_argument('--figure',choices=(*KEYS,'all'),default='all')
    args=p.parse_args();keys=KEYS if args.figure=='all' else (args.figure,);root=args.output_dir.resolve();images={}
    from plot_runtime import require_current_runtime
    runtime=require_current_runtime()
    import pymupdf as pdf
    print('PLOTTING_RUNTIME='+json.dumps(runtime,sort_keys=True))
    for key in keys:
        tree=root/f'Figure_{key}';manifest=json.loads((tree/'render_manifest.json').read_text())
        if manifest['figure']!=key or manifest['scientific_run'] or manifest['raw_read']:raise ValueError('Wrong figure manifest scope')
        for row in manifest['files']:
            f=tree/row['path']
            if not f.resolve().is_relative_to(tree):raise ValueError('Unsafe render manifest path')
            if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=row['sha256']:raise ValueError(f'Render member hash mismatch: {f}')
        full=tree/'figures'/f'Figure_{key}.pdf';panels=sorted((tree/'panels'/f'Figure_{key}').glob('*.pdf'))
        expected={'2':4,'3':4,'4':5,'5':4,'6':3,'S1':2,'S2':2,'S3':3,'S4':1}[key]
        if len(panels)!=expected:raise ValueError(f'Wrong panel/helper count: {key}')
        for f in [full,*panels]:
            for ext in ('pdf','svg','png'):
                if not f.with_suffix('.'+ext).is_file():raise ValueError(f'Missing output format: {f} {ext}')
            with pdf.open(f) as d:
                if len(d)!=1 or d[0].get_images():raise ValueError(f'Not a single-page vector figure: {f}')
                if f!=full and re.search(r'\([a-z]\)',d[0].get_text()):raise ValueError(f'Unclean panel letters: {f}')
            png=f.with_suffix('.png');pix=pdf.Pixmap(str(png))
            if pix.xres!=600 or pix.yres!=600:raise ValueError(f'PNG is not 600 dpi: {png}')
            if '<image' in f.with_suffix('.svg').read_text():raise ValueError(f'Raster embedded in SVG: {f}')
            images[str(png.relative_to(root))]=hashlib.sha256(png.read_bytes()).hexdigest()
    print('FIGURES_CHECKED='+','.join(keys))
    print('SELECTED_VECTOR_AND_PANEL_STRUCTURE=PASS\nFIGURE_CHECK_RECOMPUTES_SCIENCE=NO')

if __name__=='__main__':
    try:main()
    except (ValueError,FileNotFoundError,RuntimeError) as e:print(str(e),file=sys.stderr);raise SystemExit(2)
