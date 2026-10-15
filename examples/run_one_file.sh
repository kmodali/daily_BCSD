#!/bin/bash
# Bias-correct ONE GCM file with daily-bcsd.
#
# Edit the settings below, or override any of them from the environment:
#   DEC=3 VAR=tas examples/run_one_file.sh
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)

VAR=${VAR:-pr}
DHIST=${DHIST:-/path/to/gcm/historical/$VAR}      # historical GCM files (*.nc)
D21=${D21:-/path/to/gcm/ssp585/$VAR}              # future GCM files (*.nc)
DHISTOUT=${DHISTOUT:-/path/to/bc/historical/$VAR} # output for historical files
D21OUT=${D21OUT:-/path/to/bc/ssp585/$VAR}         # output for future files
DOBS=${DOBS:-/path/to/obs_on_gcm_grid/$VAR}       # obs files <var>_*.nc on the GCM grid
DEC=${DEC:-0}                                     # 0-based file index (historical first)
REF_ST=${REF_ST:-1985}
REF_END=${REF_END:-2014}

# Region (degrees, in the files' convention), e.g. South Asia: 5 40 60 100
MIN_LAT=${MIN_LAT:--90}; MAX_LAT=${MAX_LAT:-90}
MIN_LON=${MIN_LON:-0};   MAX_LON=${MAX_LON:-360}

# Memory: ~ tile_points x (days in file + 2 x reference days) x 12 bytes per worker.
# Laptop: CHUNK=16 WORKERS=2.  Compute node: CHUNK=32..64 WORKERS=<cores>.
CHUNK=${CHUNK:-32}
WORKERS=${WORKERS:-4}
NC_CACHE_MB=${NC_CACHE_MB:-4}

uv run --project "$REPO" daily-bcsd \
    --dhist "$DHIST" --d21 "$D21" --dhistout "$DHISTOUT" --d21out "$D21OUT" \
    --dobs "$DOBS" --var "$VAR" --dec "$DEC" --ref-st "$REF_ST" --ref-end "$REF_END" \
    --min-lat "$MIN_LAT" --max-lat "$MAX_LAT" --min-lon "$MIN_LON" --max-lon "$MAX_LON" \
    --chunk-lat "$CHUNK" --chunk-lon "$CHUNK" --workers "$WORKERS" --nc-cache-mb "$NC_CACHE_MB" \
    "$@"   # extra options, e.g. --overwrite or --pad 4 --buff 15
