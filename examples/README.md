# Examples

Ready-to-edit scripts for running the daily bias correction (`daily-bcsd`). The shell scripts
take their settings from variables at the top. Each of these can also be overridden from the
environment, so the scripts can be used without editing them.

| File | What it shows |
|---|---|
| [`run_one_file.sh`](run_one_file.sh) | Correct one GCM file. All options in one place: paths, file index, reference period, region, tile size, workers, netCDF cache. |
| [`run_all_files.sh`](run_all_files.sh) | Correct all historical + future files one after the other. It can be restarted, because existing outputs are skipped. |
| [`slurm_array.sh`](slurm_array.sh) | Levante batch job: one array task per file. |
| [`run_python.py`](run_python.py) | The same from Python, looping over variables and files with `bc_daily()`. |
| [`check_result.py`](check_result.py) | Quick statistics of GCM vs. bias-corrected (vs. obs) for one file. |
| [`numpy_core.py`](numpy_core.py) | The correction core on plain numpy arrays, without files or dask. |

## Expected data layout

The scripts assume this layout (change the paths to match yours):

```
/path/to/gcm/historical/<var>/*.nc     # e.g. one file per decade; names sort in time order
/path/to/gcm/ssp585/<var>/*.nc
/path/to/obs_on_gcm_grid/<var>/<var>_<year>.nc   # obs aggregated to the GCM grid
/path/to/bc/historical/<var>/          # output, created if missing
/path/to/bc/ssp585/<var>/
```

## Quick start

```bash
# one file
VAR=pr DEC=0 DHIST=/work/xx/gcm/historical/pr D21=/work/xx/gcm/ssp585/pr \
  DHISTOUT=/work/xx/bc/historical/pr D21OUT=/work/xx/bc/ssp585/pr DOBS=/work/xx/obs/pr \
  examples/run_one_file.sh

# all files, low memory (laptop)
CHUNK=16 WORKERS=2 VAR=tas DHIST=... D21=... DHISTOUT=... D21OUT=... DOBS=... \
  examples/run_all_files.sh

# a region only
MIN_LAT=5 MAX_LAT=40 MIN_LON=60 MAX_LON=100 VAR=tasmax DEC=5 ... examples/run_one_file.sh

# extra daily-bcsd options are passed through
examples/run_one_file.sh --overwrite --buff 15 --pad 4

# Levante: set the paths (export them or edit run_one_file.sh), set --array, then
sbatch examples/slurm_array.sh

# from Python
BASE=/work/xx VARS="pr tas" uv run python examples/run_python.py

# check a result
uv run python examples/check_result.py gcm_file.nc bc_file.nc pr /work/xx/obs/pr/pr_*.nc

# play with the core
uv run python examples/numpy_core.py
```

## Choosing tile size and workers

Memory per worker is about `tile points × (days in file + 2 × reference days) × 12 bytes`:

| Tiles (`CHUNK`) | 30-year reference, decade file | Suggested use |
|---|---|---|
| 16 × 16 | ~85 MB per worker | laptop, `WORKERS=2` |
| 32 × 32 | ~340 MB per worker | default; compute node with 1 GB per core |
| 64 × 64 | ~1.4 GB per worker | large-memory nodes, fewer and larger reads |

The run logs its own estimate (`~NN MB per tile`) before computing starts. Peak memory is
roughly a few hundred MB plus `WORKERS ×` that value.

All examples were tested on the synthetic data from `tests/ncl_compare/make_synth.py`.
