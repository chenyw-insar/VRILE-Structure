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

gmt begin Figure_S1 pdf,png A+m6p,E600
  style
  gmt basemap -R1989/2025/8.5/12.5 -JX7.3c/4.4c -Bxc"$INPUT/f1_year_ticks.txt"+l"Year" \
    -Bya1f0.5g1+l"Annual mean SIE (10@+6@+ km@+2@+)" -BWSen -Xa1.6c -Ya2.1c
  gmt plot "$INPUT/f1_annual.tsv" -i0,1 -W1.2p,#2F5D8C -Xa1.6c -Ya2.1c
  awk '$3==1 {print $1,$2}' "$INPUT/f1_annual.tsv" | gmt plot -Sc0.23c -G#C58B2A -W0.6p,gray20 -Xa1.6c -Ya2.1c
  awk '$3==1 {print $1,$2+0.35,$1}' "$INPUT/f1_annual.tsv" | gmt text -F+f9p,Helvetica-Bold,#8B5A23+jCM -Xa1.6c -Ya2.1c
  printf '1989 12.83 (a) Northern Hemisphere annual context\n' | gmt text -F+f10p,Helvetica-Bold,gray10+jLM -N -Xa1.6c -Ya2.1c

  gmt basemap -R0/30/6.8/9.9 -JX7.8c/4.4c -Bxc"$INPUT/f1_date_ticks.txt"+l"$SELECTED_YEAR" \
    -Bya0.5f0.1g0.5+l"SIE (10@+6@+ km@+2@+)" -BWSen -Xa11.1c -Ya2.1c
  gmt plot "$INPUT/f1_window.tsv" -W1.25p,#2F5D8C -Xa11.1c -Ya2.1c
  gmt plot "$INPUT/f1_window_highlight.tsv" -W1.9p,#B56A35 -Xa11.1c -Ya2.1c
  gmt plot "$INPUT/f1_window_highlight.tsv" -Sc0.19c -G#B56A35 -W0.5p,white -Xa11.1c -Ya2.1c
  gmt plot "$INPUT/f1_window_anchor.tsv" -Sd0.23c -G#E6AB02 -W0.7p,gray20 -Xa11.1c -Ya2.1c
  printf '15.3 %s\n19.5 %s\n' "$ANCHOR_LEADER_Y" "$ANCHOR_END_Y" | gmt plot -W0.6p,gray50 -Xa11.1c -Ya2.1c
  printf '18.2 %s Anchor: %s\n' "$ANCHOR_TEXT_Y" "$ANCHOR_LABEL" | gmt text -F+f8.5p,Helvetica,gray20+jLM -Xa11.1c -Ya2.1c
  printf '0 10.15 (b) %s detection example\n' "$EVENT_ID" | gmt text -F+f10p,Helvetica-Bold,gray10+jLM -N -Xa11.1c -Ya2.1c
  printf '15 5.9 %s\n15 5.64 %s\n' "$CHANGE_LABEL" "$DETECTOR_LABEL" |
    gmt text -F+f8.5p,Helvetica,gray20+jCM -N -Xa11.1c -Ya2.1c
gmt end
