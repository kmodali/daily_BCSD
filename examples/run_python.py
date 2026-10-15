"""Bias-correct all GCM files of several variables from Python.

usage: uv run python examples/run_python.py

Edit BASE, VARS and the reference period below.  Existing outputs are skipped.
"""

import glob
import logging
import os

from daily_bcsd import bc_daily

BASE = os.environ.get("BASE", "/path/to/data")
VARS = os.environ.get("VARS", "pr tas").split()
REF_ST, REF_END = 1985, 2014
BOX = (-90, 90, 0, 360)  # min_lat, max_lat, min_lon, max_lon

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")  # show progress

for var in VARS:
    dhist, d21 = f"{BASE}/gcm/historical/{var}", f"{BASE}/gcm/ssp585/{var}"
    nfiles = len(glob.glob(f"{dhist}/*.nc")) + len(glob.glob(f"{d21}/*.nc"))
    for dec in range(nfiles):
        out = bc_daily(
            dhist, d21,
            f"{BASE}/bc/historical/{var}", f"{BASE}/bc/ssp585/{var}",
            f"{BASE}/obs_on_gcm_grid/{var}",
            var, dec, REF_ST, REF_END, *BOX,
            chunk_lat=32, chunk_lon=32,  # tile size: sets memory per worker
            workers=4,                   # tiles in parallel
            nc_chunk_cache=4 * 2**20,    # bytes, set before files are opened
        )
        print("written", out)
