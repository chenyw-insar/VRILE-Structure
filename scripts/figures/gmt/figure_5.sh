#!/usr/bin/env bash
set -euo pipefail
I="${1:?processed input directory required}"; O="${2:?output directory required}"
mkdir -p "$O/user" "$O/tmp"
export GMT_USERDIR="$O/user" GMT_TMPDIR="$O/tmp" GMT_SESSION_NAME="vrile_current_${$}"
cd "$O"
W="$I"
source "$I/display.env"
text() {
  local size="$1" face="$2" color="$3" just="$4"
  gmt text -R0/16.3/0/14.7 -JX16.3c/14.7c -F+f"${size}p,${face},${color}"+j"$just" -N -Xa0c -Ya0c
}
intervals() {
  local file="$1" x="$2"
  awk '{print ">"; print $3,$2; print $4,$2}' "$file" | gmt plot -W1.3p,#2F5D8C -Xa"${x}c" -Ya9.6c
  awk '{print $3,$2;print $4,$2}' "$file" | gmt plot -Sy0.13c -W0.9p,#2F5D8C -Xa"${x}c" -Ya9.6c
  gmt plot "$file" -i0,1 -Sc0.18c -G#2F5D8C -W0.45p,white -Xa"${x}c" -Ya9.6c
}
dots() {
  local name="$1" x="$2" y="$3" size="$4"
  gmt plot "$W/${name}_mean.tsv" -W1.1p,#A55B35 -Xa"${x}c" -Ya"${y}c"
  gmt plot "$W/${name}_median.tsv" -W1.1p,#41484E,-- -Xa"${x}c" -Ya"${y}c"
  gmt plot "$W/${name}_ordered.tsv" -i0,1 -Sc"${size}c" -G#315F8E -W0.15p,white -Xa"${x}c" -Ya"${y}c"
}

gmt begin Figure_5 pdf,png A+m8p,E600
  gmt set PS_CHAR_ENCODING ISOLatin1+ GMT_GRAPHICS_DPU 600 MAP_FRAME_TYPE plain \
    FONT_ANNOT_PRIMARY 9.3p,Helvetica,gray15 FONT_LABEL 9.3p,Helvetica,gray15 \
    MAP_FRAME_PEN 0.6p,gray45 MAP_TICK_PEN_PRIMARY 0.5p,gray45 MAP_TICK_LENGTH_PRIMARY 2.5p \
    MAP_ANNOT_OFFSET_PRIMARY 3p MAP_LABEL_OFFSET 4p MAP_GRID_PEN_PRIMARY 0.25p,gray90 FORMAT_FLOAT_OUT %.17g
  printf '0.15 14.38 (a)\n5.20 14.38 (b)\n' | text 10.2 Helvetica-Bold gray10 LM
  printf '0.15 13.87 Major local events; n = %s\n5.20 13.87 X2-linked cases; n = %s\n' "$N_MAJOR" "$N_CASES" | text 9.3 Helvetica gray25 LM
  printf '2.735 13.29 Footprint linkage (X2)\n1.8875 12.83 Yes\n3.5825 12.83 No\n' | text 9.3 Helvetica gray20 CM
  gmt basemap -R0/2/0/2 -JX3.39c/2.66c -B+n -Xa1.04c -Ya9.60c
  printf '0 0\n2 0\n2 2\n0 2\n' | gmt plot -L -G#F7F8F8 -W0.65p,gray50 -Xa1.04c -Ya9.6c
  printf '0 1\n2 1\n>\n1 0\n1 2\n' | gmt plot -W0.55p,gray65 -Xa1.04c -Ya9.6c
  awk '{print $1,$2,$3}' "$I/f5a_matrix.tsv" | gmt text -F+f11p,Helvetica,gray15+jCM -Xa1.04c -Ya9.6c
  printf '0.93 11.595 Yes\n0.93 10.265 No\n' | text 9.3 Helvetica gray20 RM
  printf '0.18 10.93 Event-start (X1)\n' | gmt text -R0/16.3/0/14.7 -JX16.3c/14.7c -F+f9.3p,Helvetica,gray20+a90+jCM -N -Xa0c -Ya0c
  printf '2.735 8.99 X1 yes: %s  |  X2 yes: %s\n' "$N_X1" "$N_X2" | text 9.3 Helvetica gray25 CM

  printf '8.645 12.95 Primary \034 absolute loss\n13.515 12.95 Secondary \034 normalized loss\n' | text 9.3 Helvetica-Bold gray20 CM
  printf '6.60 11.817 Focal\n6.60 10.930 Residual\n6.60 10.043 Other-region\n' | text 9.3 Helvetica gray20 RM
  gmt basemap -R0/85/0.5/3.5 -JX3.71c/2.66c -Bxa20f10g20 -By0 -BS -Xa6.79c -Ya9.6c
  printf '0 0.5\n0 3.5\n' | gmt plot -W0.8p,gray55,-- -Xa6.79c -Ya9.6c
  intervals "$I/plot_absolute.tsv" 6.79
  gmt basemap -R0/0.065/0.5/3.5 -JX3.71c/2.66c -Bxa0.02f0.01g0.02 -By0 -BS -Xa11.66c -Ya9.6c --FORMAT_FLOAT_MAP=%.2f
  printf '0 0.5\n0 3.5\n' | gmt plot -W0.8p,gray55,-- -Xa11.66c -Ya9.6c
  intervals "$I/plot_normalized.tsv" 11.66
  printf '8.645 8.98 Mean contrast (10@+3@+ km@+2@+eq)\n13.515 8.98 Mean normalized contrast\n' | text 9.3 Helvetica gray20 CM

  printf '0.88 8.07 (c) Focal absolute differences\n9.15 8.07 (d) Focal normalized differences\n' | text 9.8 Helvetica gray10 LM
  printf '0.88 7.55 Mean: %s  |  Median: %s\n9.15 7.55 Mean: %s  |  Median: %s\n' "$ABS_MEAN" "$ABS_MEDIAN" "$NORM_MEAN" "$NORM_MEDIAN" | text 9.3 Helvetica gray30 LM
  gmt basemap -R0/280/-100/1100 -JX6.13c/5.63c -Bxc"$W/rank_ticks.tsv"+l"Ordered case rank" -Bya200f100g200+l"Difference (10@+3@+ km@+2@+eq)" -BWS -Xa1.38c -Ya0.98c
  printf '0 0\n280 0\n' | gmt plot -W0.6p,gray55,. -Xa1.38c -Ya0.98c
  dots absolute 1.38 0.98 0.06
  # The inset occupies a data-free upper-left region. All supplied cases remain in the primary axes; the inset is a display zoom only.
  gmt basemap -R0/280/-85/125 -JX3.86c/2.12c -Bxc"$W/inset_ticks.tsv" -Bya50g50 -BWS+gwhite -Xa2.13c -Ya4.11c --FONT_ANNOT_PRIMARY=9.3p,Helvetica,gray30 --MAP_FRAME_PEN=0.4p,gray50
  printf '0 0\n280 0\n' | gmt plot -W0.5p,gray55,. -Xa2.13c -Ya4.11c
  dots absolute 2.13 4.11 0.055
  printf '2.13 6.60 Central-range zoom\n' | text 9.3 Helvetica gray30 LM
  gmt basemap -R0/280/-0.25/0.75 -JX6.13c/5.63c -Bxc"$W/rank_ticks.tsv"+l"Ordered case rank" -Bya0.2f0.1g0.2+l"Normalized difference" -BWS -Xa9.15c -Ya0.98c
  printf '0 0\n280 0\n' | gmt plot -W0.6p,gray55,. -Xa9.15c -Ya0.98c
  dots normalized 9.15 0.98 0.06
gmt end
