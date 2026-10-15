#!/bin/bash
# Compare BCDaily.ncl and the Python port on synthetic data.
#
# usage: tests/ncl_compare/run_comparison.sh <workdir> [calendar]
#
# Needs NCL (on Levante: module ncl/6.6.2-gcc-11.2.0).  NCL runs take a few
# minutes per case; the four cases run in parallel.
set -euo pipefail
W=$(realpath -m "$1"); CAL=${2:-noleap}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
cd "$REPO"

[ -d "$W/tas" ] || uv run python "$HERE/make_synth.py" "$W" "$CAL"

if ! command -v ncl >/dev/null; then
    source /etc/profile.d/modules.sh 2>/dev/null || true
    module load ncl/6.6.2-gcc-11.2.0
fi

# var  dec  subdir  file
CASES=("tas 6 rcp tas_day_GCM_rcp_2010-2019" "tas 2 hist tas_day_GCM_hist_1970-1979"
       "pr 2 hist pr_day_GCM_hist_1970-1979" "pr 7 rcp pr_day_GCM_rcp_2020-2029")

for c in "${CASES[@]}"; do
    set -- $c; R=$W/$1
    mkdir -p "$R/ncl_hist" "$R/ncl_rcp"
    [ -f "$R/ncl_$3/$4.nc" ] || ncl -Q "dhist=\"$R/hist\"" "d21=\"$R/rcp\"" "dhistout=\"$R/ncl_hist\"" \
        "d21out=\"$R/ncl_rcp\"" "dobs=\"$R/obs\"" "var=\"$1\"" dec=$2 ref_st=1970 ref_end=1989 \
        min_lat=-90. max_lat=90. min_lon=0. max_lon=360. BCDaily.ncl > "$W/ncl_$1_$2.log" 2>&1 &
done
wait

for c in "${CASES[@]}"; do
    set -- $c; R=$W/$1
    uv run daily-bcsd --dhist "$R/hist" --d21 "$R/rcp" --dhistout "$R/py_hist" --d21out "$R/py_rcp" \
        --dobs "$R/obs" --var "$1" --dec "$2" --ref-st 1970 --ref-end 1989 \
        --chunk-lat 3 --chunk-lon 4 --workers 4 --overwrite 2>&1 | grep -iE "error|traceback" || true
    echo "=== $1 dec=$2"
    uv run python "$HERE/compare.py" "$R/ncl_$3/$4.nc" "$R/py_$3/$4.nc" "$1"
done
