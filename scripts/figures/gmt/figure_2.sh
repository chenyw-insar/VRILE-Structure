#!/usr/bin/env bash
set -euo pipefail
I="${1:?processed input directory required}"; O="${2:?output directory required}"
mkdir -p "$O/user" "$O/tmp"
export GMT_USERDIR="$O/user" GMT_TMPDIR="$O/tmp" GMT_SESSION_NAME="vrile_current_${$}"
cd "$O"
F2_INPUTS="$I"
want() { [[ -z "${PANEL_ONLY:-}" || "${PANEL_ONLY}" == "$1" ]]; }
B_WIDTH=5.15; B_HEIGHT=6.32; B_X=9.975; B_BAR=0.80527; B_LABEL_Y=17.25
set_style() {
  gmt set \
    GMT_GRAPHICS_DPU 600 \
    MAP_FRAME_TYPE plain \
    FONT_ANNOT_PRIMARY 8.5p,Helvetica,gray20 \
    FONT_LABEL 9p,Helvetica,gray20 \
    FONT_TITLE 10.5p,Helvetica-Bold,gray15 \
    MAP_GRID_PEN_PRIMARY 0.25p,gray88 \
    MAP_TICK_PEN_PRIMARY 0.6p,gray30 \
    MAP_FRAME_PEN 0.8p,gray25 \
    MAP_TITLE_OFFSET 5p \
    FORMAT_GEO_MAP dddF
}

panel_label() {
  local x="$1" y="$2" label="$3"
  printf '0 100 (%s)\n' "$label" |
    gmt text -R0/100/0/100 -JX1c/1c -F+f10p,Helvetica-Bold,gray10+jTL -N -Xa"$x" -Ya"$y"
}

build_f2_final() {
  local stem="Figure_2${PANEL_ONLY:-}"
  gmt begin "$stem" pdf,png,eps A+m5p,E600
    set_style
    if want a; then

    gmt coast -R-180/180/50/90 -JA0/90/5.9c -Dl -Ggray88 -S#F7FBFD -W0.35p,gray55 \
      -Bxa45g45 -Bya10g10 -BWSne+t"Event centroids" -Xa1.3c -Ya11.0c
    gmt plot "$F2_INPUTS/f2_map_broad_only.tsv" -Sc0.026c -G#A8B7C3@20 -Xa1.3c -Ya11.0c
    gmt plot "$F2_INPUTS/f2_map_severe_nonmajor.tsv" -Sc0.040c -G#4C9193@8 -Xa1.3c -Ya11.0c
    gmt plot "$F2_INPUTS/f2_map_major_severe.tsv" -Sc0.060c -G#B56A35 -Xa1.3c -Ya11.0c
    printf '%s\n' \
      'S 0.12c c 0.08c #A8B7C3 - 0.25c Broad only' \
      'S 0.12c c 0.10c #4C9193 - 0.25c Severe, non-major' \
      'S 0.12c c 0.12c #B56A35 - 0.25c Major' |
      gmt legend -R-180/180/50/90 -JA0/90/5.9c -DjBR+w3.2c+o0.08c -F+gwhite@15+p0.4p,gray55+r2p \
        --FONT_ANNOT_PRIMARY=8p,Helvetica,gray20 -Xa1.3c -Ya11.0c
    panel_label 1.02c 17.25c a
    fi
    if want b; then

    gmt basemap -R0.3/3.7/0/10000 -JX"${B_WIDTH}c/${B_HEIGHT}c" -Bxa0 \
      -Bya2000f1000g2000+l"Number of events" -BWSen+t"Broad / severe / major hierarchy" -Xa"${B_X}c" -Ya11.0c
    awk '$1==1 {print $1,$2}' "$F2_INPUTS/f2_counts.tsv" | gmt plot -Sb"${B_BAR}c+b0" -G#A8B7C3 -W0.6p,gray35 -Xa"${B_X}c" -Ya11.0c
    awk '$1==2 {print $1,$2}' "$F2_INPUTS/f2_counts.tsv" | gmt plot -Sb"${B_BAR}c+b0" -G#4C9193 -W0.6p,gray35 -Xa"${B_X}c" -Ya11.0c
    awk '$1==3 {print $1,$2}' "$F2_INPUTS/f2_counts.tsv" | gmt plot -Sb"${B_BAR}c+b0" -G#B56A35 -W0.6p,gray35 -Xa"${B_X}c" -Ya11.0c
    awk '{y=$2+350; if(y>9800)y=9800; print $1,y,$2}' "$F2_INPUTS/f2_counts.tsv" |
      gmt text -F+f8.5p,Helvetica-Bold,gray20+jCM -Xa"${B_X}c" -Ya11.0c
    printf '1 -450 Broad\n2 -450 Severe\n3 -450 Major\n' |
      gmt text -F+f8p,Helvetica,gray25+jCM -N -Xa"${B_X}c" -Ya11.0c
    panel_label 9.52c "${B_LABEL_Y}c" b
    fi
    if want c; then

    gmt basemap -R0/65/25000/10000000 -JX5.9c/6.1cl -Bxa10f5g10+l"Event duration (days)" \
      -Bya1p+l"Maximum area (km@+2@+; log scale)" -BWSen+t"Event descriptors" -Xa1.3c -Ya2.4c
    awk '{printf "%.15g %.15g\n",$1,10^$2}' "$F2_INPUTS/f2_scatter_broad_only.tsv" |
      gmt plot -Sc0.035c -G#A8B7C3@35 -Xa1.3c -Ya2.4c
    awk '{printf "%.15g %.15g\n",$1,10^$2}' "$F2_INPUTS/f2_scatter_severe_nonmajor.tsv" |
      gmt plot -Sc0.045c -G#4C9193@20 -Xa1.3c -Ya2.4c
    awk '{printf "%.15g %.15g\n",$1,10^$2}' "$F2_INPUTS/f2_scatter_major_severe.tsv" |
      gmt plot -Sc0.060c -G#B56A35@5 -Xa1.3c -Ya2.4c
    panel_label 1.02c 8.4c c
    fi
    if want d; then

    gmt makecpt -Cbatlow -T20/75/5 -H > "$GMT_TMPDIR/f2_heat.cpt"
    printf '%s\n' \
      '0 afg 1-15 Jun' '1 afg 16-30 Jun' '2 afg 1-15 Jul' \
      '3 afg 16-31 Jul' '4 afg 1-15 Aug' '5 afg 16-31 Aug' > "$GMT_TMPDIR/f2_halfmonths.txt"
    gmt basemap -R1988.5/2025.5/-0.5/5.5 -JX5.5c/6.1c -Bxa10f5g10+l"Year" \
      -Byc"$GMT_TMPDIR/f2_halfmonths.txt" -BWSen+t"Unique-event timing" -Xa9.8c -Ya2.4c
    gmt plot "$F2_INPUTS/f2_heatmap.tsv" -i0,1,2 -Sr0.145c/0.81c -C"$GMT_TMPDIR/f2_heat.cpt" -W0.05p,white -Xa9.8c -Ya2.4c
    # GMT 6.6 scales colorbar fonts with bar length. A normal GMT basemap
    # supplies the aligned physical-size annotations (measured in final PDF).
    gmt colorbar -C"$GMT_TMPDIR/f2_heat.cpt" -Dx0c/0c+w4.1c/0.24c+h -B0 -Xa10.5c -Ya1.18c
    gmt basemap -R20/75/0/1 -JX4.1c/0.24c -Bxa10+l"Number of events" -By0 -BS -Xa10.5c -Ya1.18c
    panel_label 9.52c 8.4c d
    fi
  gmt end
}

build_f2_final
