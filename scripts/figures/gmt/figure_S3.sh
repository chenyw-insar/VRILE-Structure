#!/usr/bin/env bash
set -euo pipefail
I="${1:?processed input directory required}"; O="${2:?output directory required}"
mkdir -p "$O/user" "$O/tmp"
export GMT_USERDIR="$O/user" GMT_TMPDIR="$O/tmp" GMT_SESSION_NAME="vrile_current_${$}"
cd "$O"
INPUT_DIR="$I"
draw() {
  local letter="$1" key="$2" region="$3" ticks="$4" axis="$5"
  local median
  median="$(awk -v key="$key" '$1==key {print $2}' "$INPUT_DIR/s1_summary.tsv")"
  gmt begin "Figure_S3${letter}" pdf A+m4p
    gmt set GMT_GRAPHICS_DPU 600 MAP_FRAME_TYPE plain \
      FONT_ANNOT_PRIMARY 8.5p,Helvetica,gray20 FONT_LABEL 9p,Helvetica,gray20 \
      FONT_TITLE 10.5p,Helvetica-Bold,gray15 MAP_GRID_PEN_PRIMARY 0.25p,gray88 \
      MAP_TICK_PEN_PRIMARY 0.6p,gray30 MAP_FRAME_PEN 0.8p,gray25 MAP_TITLE_OFFSET 5p FORMAT_GEO_MAP dddF
    gmt basemap -R"$region" -JX15.0c/2.4c -Bx"${ticks}"+l"$axis" -By0 -BWSen
    gmt plot "$INPUT_DIR/s1_${key}_points.tsv" -Sc0.060c -G#5F8BA5@55 -W0.15p,white
    printf '%s 0.65\n%s 1.35\n' "$median" "$median" | gmt plot -W1.2p,#B44B4B
    printf '%s 1.31 median = %.2f\n' "$median" "$median" | gmt text -F+f8p,Helvetica-Bold,#8B3045+jCM
  gmt end
}
draw a neff 0/12/0.6/1.4 a2f1g2 'Effective named-region number'
draw b largest 0/65/0.6/1.4 a10f5g10 'Largest named-region share (%)'
draw c top3 25/90/0.6/1.4 a10f5g10 'Top-three named-region share (%)'
