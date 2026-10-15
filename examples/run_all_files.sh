#!/bin/bash
# Bias-correct ALL GCM files (historical + future) one after the other.
#
# Uses the same settings as run_one_file.sh (edit there or override from the
# environment).  Existing outputs are skipped, so the loop can be restarted
# after an interruption.
#   VAR=tas DHIST=... D21=... examples/run_all_files.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

VAR=${VAR:-pr}
export DHIST=${DHIST:-/path/to/gcm/historical/$VAR}
export D21=${D21:-/path/to/gcm/ssp585/$VAR}

N=$(( $(find "$DHIST" -maxdepth 1 -name '*.nc' | wc -l) + $(find "$D21" -maxdepth 1 -name '*.nc' | wc -l) ))
echo "Correcting $N files of $VAR"

for ((dec = 0; dec < N; dec++)); do
    DEC=$dec "$HERE/run_one_file.sh" "$@"
done
