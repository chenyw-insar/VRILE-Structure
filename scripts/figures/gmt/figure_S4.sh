#!/usr/bin/env bash
set -euo pipefail
I="${1:?processed input directory required}"; O="${2:?output directory required}"
mkdir -p "$O/user" "$O/tmp"
export GMT_USERDIR="$O/user" GMT_TMPDIR="$O/tmp" GMT_SESSION_NAME="vrile_current_${$}"
cd "$O"
INPUT="$I"
source "$I/display.env"
style() {
  gmt set GMT_GRAPHICS_DPU 600 MAP_FRAME_TYPE plain FONT_ANNOT_PRIMARY 8.5p,Helvetica,gray15 \
    FONT_LABEL 9p,Helvetica,gray15 FONT_TITLE 10p,Helvetica-Bold,gray10 \
    MAP_FRAME_PEN 0.65p,gray45 MAP_TICK_PEN_PRIMARY 0.5p,gray45 \
    MAP_TICK_LENGTH_PRIMARY 3p MAP_ANNOT_OFFSET_PRIMARY 4p MAP_LABEL_OFFSET 5p \
    MAP_GRID_PEN_PRIMARY 0.25p,gray88 MAP_TITLE_OFFSET 7p FORMAT_FLOAT_OUT %.17g
}
intervals() {
  local file="$1" originx="$2" originy="$3" color="$4" shape="$5"
  awk '{print ">"; print $3,$2; print $4,$2}' "$file" |
    gmt plot -W1.4p,"$color" -Xa"$originx" -Ya"$originy"
  awk '{print $3,$2; print $4,$2}' "$file" |
    gmt plot -Sy0.13c -W1p,"$color" -Xa"$originx" -Ya"$originy"
  gmt plot "$file" -i0,1 -S"$shape"0.20c -G"$color" -W0.5p,white -Xa"$originx" -Ya"$originy"
}

gmt begin Figure_S4 pdf,png A+m6p,E600
  style
  gmt basemap -R0/0.012/0.5/2.5 -JX10.8c/3.3c -Bxa0.003f0.001g0.003+l"BASE normalized extra-focal intercept" \
    -Byc"$INPUT/adjusted_ticks.txt" -BWSen -Xa2.2c -Ya1.5c --FORMAT_FLOAT_MAP=%.3f
  intervals "$INPUT/plot_adjusted.tsv" 2.2c 1.5c '#B56A35' d
  printf '0 2.79 Focal-strength sensitivity; n = %s\n' "$N_CASES" | gmt text -F+f10p,Helvetica-Bold,gray10+jLM -N -Xa2.2c -Ya1.5c
gmt end
