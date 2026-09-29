#!/usr/bin/env bash
set -euo pipefail
I="${1:?processed input directory required}"; O="${2:?output directory required}"
mkdir -p "$O/user" "$O/tmp"
export GMT_USERDIR="$O/user" GMT_TMPDIR="$O/tmp" GMT_SESSION_NAME="vrile_current_${$}"
cd "$O"
M="$I"
source "$I/display.env"
style() {
  gmt set FONT_ANNOT_PRIMARY 9p,Helvetica,35/45/55 FONT_LABEL 9.3p,Helvetica,35/45/55 \
    FONT 9p,Helvetica,35/45/55 MAP_FRAME_TYPE plain MAP_FRAME_PEN 0.5p,115/125/132 \
    MAP_TICK_PEN_PRIMARY 0.5p,115/125/132 MAP_TICK_LENGTH_PRIMARY 2p MAP_ANNOT_OFFSET_PRIMARY 3p \
    MAP_GRID_PEN_PRIMARY 0.25p,225/230/234 PS_CHAR_ENCODING ISOLatin1+ PS_PAGE_COLOR white FORMAT_GEO_MAP dddF
}
canvas() {
  local w="$1" h="$2"
  printf '0 0\n%s 0\n%s %s\n0 %s\n0 0\n' "$w" "$w" "$h" "$h" |
    gmt plot -R0/"$w"/0/"$h" -JX"${w}c/${h}c" -L -Gwhite -W0p,white
}

gmt begin Figure_3a_compact_aligned_clean pdf A+m0p
  style; canvas 7.2 11.9
  # Original projection, coordinates, component IDs and largest-component outline.
  MAP=(-R-180/180/50/90 -JA0/90/6.0c -Xa0.6c -Ya4.45c)
  gmt coast "${MAP[@]}" -Dl -Ggray90 -S#F8FBFC -W0.198p,gray55 -Bxa45g45 -Bya10g10 -BWSne
  gmt makecpt -C#174A72,#DCEAF3 -T1/"$COMPONENT_CPT_END"/1 -Z -H > "$O/compact_aligned_components.cpt"
  awk '$4==0 {print $1,$2,$3}' "$M/f4_components.tsv" |
    gmt plot "${MAP[@]}" -Ss0.0429545455c -C"$O/compact_aligned_components.cpt"
  awk '$4==1 {print $1,$2}' "$M/f4_components.tsv" |
    gmt plot "${MAP[@]}" -Ss0.0429545455c -G#174A72
  gmt plot "$M/f4_largest_perimeter.gmt" "${MAP[@]}" -W0.725p,#D69A2D
  BAR=(-R0/100/0/1 -JX6.0c/0.38c -Xa0.6c -Ya3.10c)
  gmt basemap "${BAR[@]}" -B0
  printf '0 0\n%s 0\n%s 1\n0 1\n' "$REP_SHARE" "$REP_SHARE" | gmt plot "${BAR[@]}" -L -G#174A72 -W0.4p,gray55
  printf '%s 0\n100 0\n100 1\n%s 1\n' "$REP_SHARE" "$REP_SHARE" | gmt plot "${BAR[@]}" -L -G#DCEAF3 -W0.4p,gray55
  printf '%s 0.5 %s%%\n' "$REP_SHARE_X" "$REP_SHARE" | gmt text "${BAR[@]}" -F+f9p,Helvetica-Bold,white+jCM
  printf '%s 0.5 %s%%\n' "$REP_OTHER_X" "$REP_OTHER" | gmt text "${BAR[@]}" -F+f9p,Helvetica-Bold,gray20+jCM
  TEXT=(-R0/7.2/0/11.9 -JX7.2c/11.9c -Xa0c -Ya0c)
  printf '1.434 2.60 largest\n4.434 2.60 other components\n' |
    gmt text "${TEXT[@]}" -F+f9p,Helvetica,35/45/55+jCM
  printf '3.6 1.90 Share of resolved\n3.6 1.50 retained-component loss\n' |
    gmt text "${TEXT[@]}" -F+f9p,Helvetica,35/45/55+jCM
  printf '3.6 0.80 %s\n3.6 0.30 %s components; largest outlined\n' "$REP_LABEL" "$N_COMPONENTS" |
    gmt text "${TEXT[@]}" -F+f9p,Helvetica,35/45/55+jCM
gmt end

gmt begin Figure_3b_compact_aligned_clean pdf A+m0p
  style; canvas 8.5 5.7
  C=(-R0/100/0/100 -JX6.85c/3.10c -Xa1.25c -Ya1.10c)
  gmt basemap "${C[@]}" -Bxa20f10 -Bya25+l"Cumulative events (%)" -BWSen
  printf '0 50\n100 50\n' | gmt plot "${C[@]}" -W0.45p,175/185/192,-
  # Both are solid step curves, with equal line width. No smoothing.
  gmt plot "$I/largest_ecdf.tsv" "${C[@]}" -W1.35p,35/83/119
  gmt plot "$I/top3_ecdf.tsv" "${C[@]}" -W1.35p,159/100/62
  printf '%s 50\n' "$MED_LARGEST" | gmt plot "${C[@]}" -Sc0.145c -Gwhite -W1.1p,35/83/119
  printf '%s 50\n' "$MED_TOP3" | gmt plot "${C[@]}" -Ss0.145c -Gwhite -W1.1p,159/100/62
  printf '11 42 Median\n11 29 %s%%\n' "$MED_LARGEST_LABEL" | gmt text "${C[@]}" -F+f9p,Helvetica-Bold,35/83/119+jCM
  printf '77 42 Median\n77 29 %s%%\n' "$MED_TOP3_LABEL" | gmt text "${C[@]}" -F+f9p,Helvetica-Bold,159/100/62+jCM
  printf '4 92 n = %s\n' "$N_EVENTS" | gmt text "${C[@]}" -F+f9p,Helvetica,70/80/90+jML
  TEXT=(-R0/8.5/0/5.7 -JX8.5c/5.7c -Xa0c -Ya0c)
  printf '1.25 5.23\n1.95 5.23\n' | gmt plot "${TEXT[@]}" -W1.35p,35/83/119
  printf '2.15 5.23 Largest-component share\n' | gmt text "${TEXT[@]}" -F+f9.3p,Helvetica,35/83/119+jML
  printf '1.25 4.72\n1.95 4.72\n' | gmt plot "${TEXT[@]}" -W1.35p,159/100/62
  printf '2.15 4.72 Top-three share\n' | gmt text "${TEXT[@]}" -F+f9.3p,Helvetica,159/100/62+jML
  printf '4.675 0.25 Share of resolved retained-component loss (%%)\n' |
    gmt text "${TEXT[@]}" -F+f9p,Helvetica,35/45/55+jCM
gmt end

vertical_dot_panel() {
  local lab="$1" name="$2" ymin="$3" ymax="$4" tick="$5" ylabel="$6" med="$7"
  gmt begin "Figure_3${lab}_compact_aligned_clean" pdf A+m0p
    style; canvas 4.0 5.8
    C=(-R"-0.15/1.5/${ymin}/${ymax}" -JX2.35c/4.45c -Xa1.35c -Ya0.40c)
    gmt basemap "${C[@]}" -Bx0 -Bya"$tick"+l"$ylabel" -BW
    # Only swap plotting columns: numeric value -> y, saved display jitter -> x.
    # Reuse each saved event, IQR endpoint and median exactly; no new summaries.
    gmt plot "$I/${name}_dots.tsv" -i1,0 "${C[@]}" -Sc0.076c -G65/112/148 -W0.12p,white
    gmt plot "$I/${name}_iqr.tsv" -i1,0 "${C[@]}" -W2.5p,103/117/128
    gmt plot "$I/${name}_caps.tsv" -i1,0 "${C[@]}" -W0.8p,103/117/128
    gmt plot "$I/${name}_median.tsv" -i1,0 "${C[@]}" -W1.4p,30/46/60
    printf '2.45 5.40 Median %s\n' "$med" |
      gmt text -R0/4/0/5.8 -JX4c/5.8c -Xa0c -Ya0c -F+f9.3p,Helvetica-Bold,35/45/55+jCM
  gmt end
}
vertical_dot_panel c neff 0 20 5 'Effective component number' "$MED_NEFF_LABEL"
vertical_dot_panel d spread 900 2800 500 'Weighted spread (km)' "$MED_SPREAD_LABEL km"
