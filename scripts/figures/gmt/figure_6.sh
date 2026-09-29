#!/usr/bin/env bash
set -euo pipefail
I="${1:?processed input directory required}"; O="${2:?output directory required}"
mkdir -p "$O/user" "$O/tmp"
export GMT_USERDIR="$O/user" GMT_TMPDIR="$O/tmp" GMT_SESSION_NAME="vrile_current_${$}"
cd "$O"
INPUTS="$I"
panel_label() {
  printf '0 100 (%s)\n' "$3" |
    gmt text -R0/100/0/100 -JX1c/1c -F+f10p,Helvetica-Bold,gray10+jTL -N -Xa"$1" -Ya"$2"
}

gmt begin Figure_6 pdf,png A+m5p,E600
  gmt set GMT_GRAPHICS_DPU 600 MAP_FRAME_TYPE plain \
    FONT_ANNOT_PRIMARY 9.2p,Helvetica,gray20 FONT_LABEL 9.2p,Helvetica,gray20 \
    FONT_TITLE 10.5p,Helvetica-Bold,gray15 MAP_GRID_PEN_PRIMARY 0.25p,gray88 \
    MAP_TICK_PEN_PRIMARY 0.6p,gray30 MAP_FRAME_PEN 0.8p,gray25 \
    MAP_TITLE_OFFSET 5p FORMAT_GEO_MAP dddF

  # Panel a: same projection, mask cells, labels, symbol dimensions, and colors.
  gmt coast -R-180/180/60/90 -JA0/90/6.9c -Dl -Ggray90 -S#F8FBFC -W0.3p,gray60 \
    -Bxa45g45 -Bya10g10 -BWSne+t"Five prespecified regions" -Xa1.25c -Ya6.5c
  for spec in '2 #8FB3CF' '5 #D8A36A' '6 #8EBE9A' '7 #B49BC8' '1 #E0C36E'; do
    set -- $spec
    gmt plot "$INPUTS/f3_region_$1.tsv" -Ss0.069c -G"$2" -Xa1.25c -Ya6.5c
  done
  gmt text "$INPUTS/f3_region_labels.tsv" -F+f9.2p,Helvetica-Bold,gray15+jCM -Xa1.25c -Ya6.5c
  printf '50 -15 NSIDC-0780 regions\n' |
    gmt text -R0/100/0/100 -JX6.9c/6.9c -F+f9.2p,Helvetica,gray30+jCM -N -Xa1.25c -Ya6.5c
  panel_label 0.95c 13.65c a

  # Panel b: exact accepted four-row display and categorical palette.
  # Preserve the map-label clearance after raising the ordinary font size.
  bx=11.20c
  gmt makecpt -C#EEF1F4,#80ADA7,#234E70 -T-0.5/2.5/1 -H > "$GMT_TMPDIR/regional.cpt"
  gmt basemap -R0/5/0/4 -JX6.55c/4.4c -B+n -Xa"$bx" -Ya9.0c
  gmt plot "$INPUTS/f3_matrix_selected.tsv" -i0,1,2 -Sr1.08c/0.84c -C"$GMT_TMPDIR/regional.cpt" -Xa"$bx" -Ya9.0c
  awk '$3==2 {print $1,$2,$4}' "$INPUTS/f3_matrix_selected.tsv" |
    gmt text -F+f10p,Helvetica-Bold,white+jCM -Xa"$bx" -Ya9.0c
  awk '$3!=2 {print $1,$2,$4}' "$INPUTS/f3_matrix_selected.tsv" |
    gmt text -F+f10p,Helvetica-Bold,gray15+jCM -Xa"$bx" -Ya9.0c
  printf '2.5 4.74 Diagnostics with regional support\n' |
    gmt text -F+f10p,Helvetica-Bold,gray15+jCM -N -Xa"$bx" -Ya9.0c
  printf '%s\n' '0.5 4.22 Beaufort' '1.5 4.22 Laptev' '2.5 4.22 Kara' '3.5 4.22 Barents' '4.5 4.22 Central' |
    gmt text -F+f9.2p,Helvetica-Bold,gray20+jCM -N -Xa"$bx" -Ya9.0c
  printf '%s\n' '0 3.5 SIC state' '0 2.5 SIC change' '0 1.5 Wind speed' '0 0.5 Meridional wind' |
    gmt text -F+f9.2p,Helvetica,gray20+jRM -N -Xa"$bx" -Ya9.0c
  gmt basemap -R0/5/0/3 -JX6.55c/2.1c -B+n -Xa"$bx" -Ya6.2c
  printf '0.18 2.5 2\n0.18 1.5 1\n0.18 0.5 0\n' |
    gmt plot -Ss0.34c -C"$GMT_TMPDIR/regional.cpt" -Xa"$bx" -Ya6.2c
  printf '0.18 2.5 R\n' | gmt text -F+f9.2p,Helvetica-Bold,white+jCM -Xa"$bx" -Ya6.2c
  printf '0.18 1.5 P\n0.18 0.5 N\n' | gmt text -F+f9.2p,Helvetica-Bold,gray15+jCM -Xa"$bx" -Ya6.2c
  printf '0.50 2.5 Robustness-supported\n0.50 1.5 Primary-supported\n0.50 0.5 Not supported\n' |
    gmt text -F+f9.2p,Helvetica,gray20+jLM -Xa"$bx" -Ya6.2c
  printf '2.5 -0.60 All eight diagnostics: Table S1\n' |
    gmt text -F+f9.2p,Helvetica,gray30+jCM -N -Xa"$bx" -Ya6.2c
  panel_label 10.92c 13.65c b

  # Panel c has ONE continuous data coordinate area: one R, J, and origin.
  printf '50 50 Primary wind contrasts\n' |
    gmt text -R0/100/0/100 -JX15.2c/1c -F+f10p,Helvetica-Bold,gray15+jCM -N -Xa0.95c -Ya4.25c
  panel_label 0.95c 3.87c c
  printf '%s\n' 'N 2' \
    'S 0.12c c 0.18c #315F8E 0.5p,white 0.35c Laptev \034 meridional wind (v10)' \
    'S 0.12c s 0.225c #247F79 0.5p,white 0.35c Beaufort \034 wind speed' |
    gmt legend -R0/100/0/100 -JX14.6c/1c -Dx0c/0c+w14.6c+jBL \
      --FONT_ANNOT_PRIMARY=9.2p,Helvetica,gray20 -Xa2.0c -Ya3.65c

  # All six data points and CI endpoints use these exact four options.
  coord=(-R-0.2/2.9/0.5/3.5 -JX14.6c/2.3c -Xa2.0c -Ya1.05c)
  gmt basemap "${coord[@]}" -Bxa0.5f0.1+l"Mean event\035background contrast (m s@+-1@+)" -By0 -BWSen
  printf '0 0.5\n0 3.5\n' | gmt plot "${coord[@]}" -W0.65p,gray60,-
  for idx in 1 2; do
    # GMT specifies the square by diagonal; 0.225c matches the circle's area.
    if [[ "$idx" == 1 ]]; then offset=0.15; color='#315F8E'; symbol=c; size=0.18c; else offset=-0.15; color='#247F79'; symbol=s; size=0.225c; fi
    awk -v d="$offset" '{print ">";print $3,$2+d;print $4,$2+d}' "$INPUTS/plot_c${idx}_primary_buffers.tsv" |
      gmt plot "${coord[@]}" -W1.25p,"$color"
    awk -v d="$offset" '{print ">";print $3,$2+d-0.045;print $3,$2+d+0.045;print ">";print $4,$2+d-0.045;print $4,$2+d+0.045}' "$INPUTS/plot_c${idx}_primary_buffers.tsv" |
      gmt plot "${coord[@]}" -W0.9p,"$color"
    awk -v d="$offset" '{print $1,$2+d}' "$INPUTS/plot_c${idx}_primary_buffers.tsv" |
      gmt plot "${coord[@]}" -S"$symbol$size" -G"$color" -W0.5p,white
  done
  printf '%s\n' '-0.23542857142857143 3 100 km' '-0.23542857142857143 2 300 km' '-0.23542857142857143 1 500 km' |
    gmt text "${coord[@]}" -F+f9.2p,Helvetica,gray20+jRM -N
gmt end
