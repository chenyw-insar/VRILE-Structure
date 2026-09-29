#!/usr/bin/env bash
set -euo pipefail
I="${1:?processed input directory required}"; O="${2:?output directory required}"
mkdir -p "$O/user" "$O/tmp"
export GMT_USERDIR="$O/user" GMT_TMPDIR="$O/tmp" GMT_SESSION_NAME="vrile_current_${$}"
cd "$O"
INPUT="$I"
text() { gmt text "$1" -F+f"$2",Helvetica,"$3"+j"$4" -N; }
gmt begin Figure_S2 pdf,png A+m5p,E600
  gmt set FONT_ANNOT_PRIMARY 9.5p,Helvetica FONT_LABEL 9.5p,Helvetica MAP_FRAME_PEN 0.5p,gray65
  gmt basemap -R0/170/0/145 -JX17c/14.5c -B+n
  gmt plot "$INPUT/x1_bands.tsv" -L -G'#E8EFF5' -W0.4p,'#B5C8D9'
  gmt plot "$INPUT/x2_bands.tsv" -L -G'#EFF6F4' -W0p
  gmt plot "$INPUT/matched_bands.tsv" -L -G'#F1E5DC' -W0.5p,'#A66846'
  gmt plot "$INPUT/axes.tsv" -W0.55p,gray65
  gmt plot "$INPUT/anchor_lines.tsv" -W0.65p,gray55,-
  gmt plot "$INPUT/patches_eligible.tsv" -W1.8p,'#247F79'
  gmt plot "$INPUT/patches_not_eligible.tsv" -W1.4p,gray60
  gmt plot "$INPUT/ends_eligible.tsv" -Sc0.17c -G'#247F79' -W0.5p,white
  gmt plot "$INPUT/ends_not_eligible.tsv" -Sc0.17c -Gwhite -W0.75p,gray55
  gmt plot "$INPUT/local_starts.tsv" -Sc0.18c -G'#315F8E' -W0.5p,white
  gmt plot "$INPUT/matched_anchors.tsv" -W0.9p,'#A66846'
  text "$INPUT/labels_left.tsv" 9.5p gray25 LM
  text "$INPUT/labels_center.tsv" 9.5p gray25 CM
  text "$INPUT/labels_teal.tsv" 9.5p '#247F79' CM
  text "$INPUT/labels_blue.tsv" 9.5p '#315F8E' CM
  text "$INPUT/labels_brown.tsv" 9.5p '#825035' CM
  text "$INPUT/labels_right.tsv" 9.5p gray40 RM
  gmt text "$INPUT/titles.tsv" -F+f10.5p,Helvetica-Bold,gray15+jLM -N
  printf '0 77\n170 77\n' | gmt plot -W0.5p,gray82
gmt end
