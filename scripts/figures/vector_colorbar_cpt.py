#!/usr/bin/env python3
"""Represent a fine continuous GMT CPT as midpoint-colored vector slices.

This CPT is for the colorbar only. Data cells retain the continuous mapping.
"""
from pathlib import Path
import sys

NAMES={}
for row in Path('/usr/share/X11/rgb.txt').read_text().splitlines():
    z=row.split()
    if len(z)>=4 and z[0].isdigit():NAMES[''.join(z[3:]).lower()]=list(map(float,z[:3]))

def rgb(s):
    if s in ('white','black'):return [255.0 if s=='white' else 0.0]*3
    if s.lower() in NAMES:return NAMES[s.lower()]
    v=list(map(float,s.split('/')))
    return v*3 if len(v)==1 else v

src,dst=map(Path,sys.argv[1:])
out=[]
for line in src.read_text().splitlines():
    z=line.split()
    if not z or z[0][0] in '#BFN':out.append(line);continue
    float(z[0]);float(z[2])
    a,b=rgb(z[1]),rgb(z[3]);assert len(a)==len(b)==3
    color='/'.join(f'{(x+y)/2:.6f}' for x,y in zip(a,b))
    out.append(f'{z[0]}\t{color}\t{z[2]}\t{color}')
dst.write_text('\n'.join(out)+'\n')
