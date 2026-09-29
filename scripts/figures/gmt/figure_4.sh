#!/usr/bin/env bash
set -euo pipefail
I="${1:?processed input directory required}"; O="${2:?output directory required}"
mkdir -p "$O/user" "$O/tmp"
export GMT_USERDIR="$O/user" GMT_TMPDIR="$O/tmp" GMT_SESSION_NAME="vrile_current_${$}"
cd "$O"
source "$I/display.env"
HELPERS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
style() {
  gmt set FONT_ANNOT_PRIMARY 9.2p,Helvetica,35/45/55 FONT_LABEL 9.5p,Helvetica,35/45/55 \
    FONT 9.2p,Helvetica,35/45/55 MAP_FRAME_TYPE plain MAP_FRAME_PEN 0.5p,100/115/125 \
    MAP_TICK_PEN_PRIMARY 0.45p,100/115/125 MAP_TICK_LENGTH_PRIMARY 1.5p MAP_ANNOT_OFFSET_PRIMARY 2p \
    MAP_GRID_PEN_PRIMARY 0.25p,225/230/234 PS_CHAR_ENCODING ISOLatin1+ PS_PAGE_COLOR white
}
canvas() {
  local w="$1" h="$2"
  printf '0 0\n%s 0\n%s %s\n0 %s\n0 0\n' "$w" "$w" "$h" "$h" |
    gmt plot -R0/"$w"/0/"$h" -JX"${w}c/${h}c" -L -Gwhite -W0p,white
}
# Continuous color mapping for data, fine constant-color vector slices for bars.
gmt makecpt -C#F3F7FB,#B7D3E8,#6195BD,#173E63 -T0/20/0.05 -Z > "$I/neff.cpt"
gmt makecpt -C#FFF8EA,#F0CCA0,#CC9860,#805325 -T0/100/0.25 -Z > "$I/largest.cpt"
for name in neff largest; do
  limit="$(tr -d '\n' < "$I/${name}_delta_limit.txt")"
  gmt makecpt -C#2B6689,#F7F7F4,#A45D37 -T"-${limit}/${limit}/0.05" -Z > "$I/${name}_delta.cpt"
  "${VRILE_PLOT_PYTHON:?}" -B "$HELPERS/vector_colorbar_cpt.py" "$I/${name}.cpt" "$I/${name}_bar.cpt"
  "${VRILE_PLOT_PYTHON:?}" -B "$HELPERS/vector_colorbar_cpt.py" "$I/${name}_delta.cpt" "$I/${name}_delta_bar.cpt"
done

heatmap() {
  local variant="$1" name="$2" lab="$3" width x title limit
  if [[ "$name" == neff ]]; then width=8.2; x=1.0; title='Effective component number'; else width=8.0; x=0.8; title='Largest-component share (%)'; fi
  limit="$(tr -d '\n' < "$I/${name}_delta_limit.txt")"
  local mid; mid="$(awk -v x="$x" 'BEGIN {print x+3.45}')"
  gmt begin "Figure_4${lab}_${variant}_clean" pdf A+m0p
    style; canvas "$width" 10.3
    C=(-R0.5/9.5/0.5/$EVENT_TOP -JX6.9c/6.3c -Xa"${x}c" -Ya2.0c)
      gmt plot "$I/${name}_none.tsv" "${C[@]}" -Sr0.766667c/"${CELL_HEIGHT}c" -C"$I/${name}.cpt"
      gmt plot "$I/${name}_delta.tsv" "${C[@]}" -Sr0.766667c/"${CELL_HEIGHT}c" -C"$I/${name}_delta.cpt"
    if [[ "$name" == neff ]]; then
      gmt basemap "${C[@]}" -Bx0 -Byc"$I/event_ticks.txt"+l"Event (date order)" -BWSen
    else
      gmt basemap "${C[@]}" -B0 -BwsEN
    fi
    printf '>\n3.5 0.5\n3.5 %s\n>\n6.5 0.5\n6.5 %s\n' "$EVENT_TOP" "$EVENT_TOP" | gmt plot "${C[@]}" -W0.85p,white
    printf '0.5 0.5\n1.5 0.5\n1.5 %s\n0.5 %s\n0.5 0.5\n' "$EVENT_TOP" "$EVENT_TOP" | gmt plot "${C[@]}" -W0.85p,25/45/65
    printf '%s 9.94 %s\n' "$mid" "$title" |
      gmt text -R0/"$width"/0/10.3 -JX"${width}c/10.3c" -Xa0c -Ya0c -F+f9.5p,Helvetica-Bold+jMC
      awk -v x="$x" 'BEGIN {print x+1.15,9.46,"Absolute"; print x+4.6,9.46,"Change from None"}' | gmt text -F+f9p,Helvetica,60/75/85+jMC
    awk -v x="$x" 'BEGIN {print x+1.15,9.04,"None"; print x+3.45,9.04,"0.15"; print x+5.75,9.04,"0.30"}' | gmt text -F+f9.2p,Helvetica+jMC
    awk -v x="$x" 'BEGIN {for (j=0;j<9;j++) {t=(j%3==0?">4":(j%3==1?">20":">50"));print x+(j+.5)*6.9/9,8.60,t}}' | gmt text -F+f9.2p,Helvetica+jMC
      local tick=10 dtick=7; [[ "$name" == largest ]] && { tick=50; dtick=25; }
      local c1 c2; c1="$(awk -v x="$x" 'BEGIN {print x+1.05}')"; c2="$(awk -v x="$x" 'BEGIN {print x+4.80}')"
      # GMT 6.6 scales colorbar text by sqrt(length_cm/15). Compensate here
      # to retain an actual 9.2 pt font, and leave room between end labels.
      gmt colorbar -C"$I/${name}_bar.cpt" -Np -Dx"${c1}c/1.40c+w1.95c/0.19c+h+jTC" -Bxa"$tick" -Xa0c -Ya0c --FONT_ANNOT_PRIMARY=25.52p
      gmt colorbar -C"$I/${name}_delta_bar.cpt" -Np -Dx"${c2}c/1.40c+w3.9c/0.19c+h+jTC" -Bxa"$dtick" -Xa0c -Ya0c --FONT_ANNOT_PRIMARY=18.05p
      printf '%s 0.59 Absolute\n' "$c1" | gmt text -R0/"$width"/0/10.3 -JX"${width}c/10.3c" -Xa0c -Ya0c -F+f9p,Helvetica+jMC
      if [[ "$name" == largest ]]; then
        printf '%s 0.59 Change (percentage points)\n' "$c2" | gmt text -F+f9p,Helvetica+jMC
      else
        printf '%s 0.59 Change (effective number)\n' "$c2" | gmt text -F+f9p,Helvetica+jMC
      fi
  gmt end
}
for variant in paired_delta; do
  heatmap "$variant" neff a
  heatmap "$variant" largest b
done

gmt begin Figure_4c_clean pdf A+m0p
  style; canvas 16.35 5.45
  C=(-R0.5/9.5/55/90 -JX14.65c/2.95c -Xa1.30c -Ya1.25c)
  gmt basemap "${C[@]}" -Bpxc"$I/cell_ticks.txt" -Bpya10+l"Loss coverage (%)" -BWSen
  printf '>\n3.5 55\n3.5 90\n>\n6.5 55\n6.5 90\n' | gmt plot "${C[@]}" -W0.4p,205/215/222,-
  colors=(49/95/142 36/127/121 156/101/66)
  for j in 0 1 2 3 4 5 6 7 8; do
    g=$((j/3))
    awk '{print $1,$3;print $1,$4}' "$I/coverage_${j}.tsv" | gmt plot "${C[@]}" -W2.2p,"${colors[$g]}"
    gmt plot "$I/coverage_${j}.tsv" "${C[@]}" -i0,1 -Sc0.16c -Gwhite -W1p,"${colors[$g]}"
  done
  printf '1.30 5.08 Median (%%)\n' | gmt text -R0/16.35/0/5.45 -JX16.35c/5.45c -Xa0c -Ya0c -F+f9.2p,Helvetica-Bold+jML
  printf '15.95 5.08 Bars: interquartile range\n' | gmt text -F+f9.2p,Helvetica+jMR
  for j in 0 1 2 3 4 5 6 7 8; do
    awk '{printf "%.6f 4.65 %.1f\n",1.3+($1-.5)*14.65/9,$2}' "$I/coverage_${j}.tsv" |
      gmt text -F+f9.5p,Helvetica+jMC
  done
  printf '3.741667 0.30 None\n8.625 0.30 0.15\n13.508333 0.30 0.30\n' | gmt text -F+f9.5p,Helvetica+jMC
gmt end

gmt begin Figure_4_paired_reference pdf A+m0p
  style; canvas 16.35 0.55
  printf '8.175 0.275 Change from None: same event and retained-size threshold\n' | gmt text -F+f9.2p,Helvetica+jMC
gmt end

gmt begin Figure_4_primary_legend pdf A+m0p
  style; canvas 16.35 0.65
  printf '0.9 0.20\n1.20 0.20\n1.20 0.49\n0.9 0.49\n0.9 0.20\n' | gmt plot -W0.85p,25/45/65
  printf '1.42 0.345 Primary: None, >4 cells\n8.1 0.345 Unboxed columns: sensitivity settings\n' | gmt text -F+f9.2p,Helvetica+jML
gmt end
