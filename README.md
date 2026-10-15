# daily_BCSD
This repository contains code for the daily Bias Correction/Spatial Disaggregation (BCSD) statistical downscaling method used to generate the following datasets:
- NEX-GDDP: https://www.nccs.nasa.gov/services/data-collections/land-based-products/nex-gddp
- NEX-GDDP-CMIP6: https://www.nccs.nasa.gov/services/data-collections/land-based-products/nex-gddp-cmip6
- NEX-DCP30-CMIP6: https://data.nas.nasa.gov/nex-dcp30-cmip6
- NEX-HIDCP-CMIP6: https://data.nas.nasa.gov/nex-hidcp-cmip6

See Thrasher, et al., 2012, (https://doi.org/10.5194/hess-16-3309-2012) and Thrasher, et al., 2022, (https://doi.org/10.1038/s41597-022-01393-4) for more details.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18525436.svg)](https://doi.org/10.5281/zenodo.18525436)

## Contents

| Path | What it is |
|---|---|
| `BCDaily.ncl`, `BCFuncs.ncl`, `get1Dindex2.ncl` | Original NCL bias correction (BC) step |
| `SDDaily.ncl` | Original NCL spatial disaggregation (SD) step |
| `src/daily_bcsd/` | Python port of the BC step (numpy + xarray + dask) |
| `tests/` | Unit tests, an end-to-end test and an NCL comparison harness |
| `examples/` | Ready-to-edit run scripts: single file, all files, SLURM array, Python API, result check, numpy core |

## Python port of the BC step

`src/daily_bcsd` reimplements `BCDaily.ncl` with numpy and xarray, computing lazily with dask.
It is designed to process large datasets with little memory:

- **Tiled processing.** Every grid point is corrected independently, so the domain is cut into
  lat/lon tiles that each carry the full time series. Each tile is read, corrected and written on
  its own. Peak memory depends on the tile size and the number of workers, not on the domain size.
- **Statistics per day of year.** The CDFs depend only on the day of year. They are built at most
  366 times per file, instead of once per day as in NCL.
- **Same results as NCL.** The NCL behaviour is reproduced, including its quirks and its float32
  arithmetic, so results match the original datasets (see [Testing](#testing)).

| Python | Replaces |
|---|---|
| `daily_bcsd/bcfuncs.py` | `BCFuncs.ncl`: distribution tails (Weibull, EVI, normal), `stat4` (`dim_stat4_n`), `lookup` |
| `daily_bcsd/bc_daily.py` | `BCDaily.ncl` and `get1Dindex2.ncl`: file handling, monthly `tas` trend, per-tile bias correction (`bias_correct_block`), output, CLI |

### Method in short

For the GCM file selected with `--dec`:

1. **`tas*` only:** compute monthly GCM anomalies relative to the reference-period climatology,
   and smooth them per calendar month with a `2*pad+1`-year running mean. This trend is zero up to
   the end of the reference period. Subtract it from the daily data.
2. **For every day:** take all reference-period GCM days and all obs days within ±`buff` days of
   its day of year, and sort them.
3. **Quantile mapping:** map the GCM value to its quantile in the GCM distribution, then map that
   quantile to a value in the obs distribution. Values outside the sampled range are extrapolated
   with fitted distributions:
   - precipitation-like variables: Weibull below the range, EVI (Gumbel) above it;
   - temperature: normal distribution.
4. **Precipitation-like variables:** set values ≤ 0 to 0. **`tas*`:** add the trend back.
5. Points that are missing in the first obs time step stay missing (the obs mask).

## Installation

The project uses [uv](https://docs.astral.sh/uv/) and needs Python ≥ 3.14.

```bash
cd daily_BCSD
uv sync            # creates .venv with all dependencies
uv run daily-bcsd --help
```

Dependencies: numpy, xarray, dask, flox, netCDF4 (and cftime through netCDF4).

## Usage

One run corrects one GCM file:

```bash
uv run daily-bcsd \
    --dhist  /path/gcm/historical   --d21    /path/gcm/ssp585 \
    --dhistout /path/bc/historical  --d21out /path/bc/ssp585 \
    --dobs   /path/obs_on_gcm_grid  --var pr --dec 7 \
    --ref-st 1985 --ref-end 2014 \
    --min-lat -60 --max-lat 90 --min-lon 0 --max-lon 360 \
    --chunk-lat 32 --chunk-lon 32 --workers 4
```

From Python:

```python
from daily_bcsd import bc_daily
bc_daily(dhist, d21, dhistout, d21out, dobs, "pr", 7, 1985, 2014,
         -60, 90, 0, 360, chunk_lat=32, chunk_lon=32, workers=4)
```

More complete, tested scripts are in [`examples/`](examples/README.md): single file, all files,
SLURM array, Python API, result check and the numpy core.

To correct a full set of files, loop over `--dec`, for example in a SLURM array job:

```bash
#SBATCH --array=0-24
uv run daily-bcsd ... --dec ${SLURM_ARRAY_TASK_ID}
```

### Required parameters

| Parameter | NCL variable | Description |
|---|---|---|
| `--dhist DIR` | `dhist` | Directory with the historical GCM files (`*.nc`). Files are sorted alphabetically, so their names must sort in time order. |
| `--d21 DIR` | `d21` | Directory with the future GCM files (`*.nc`). They are appended after the historical files. |
| `--dhistout DIR` | `dhistout` | Output directory for corrected historical files. The output path is the input path with `dhist` replaced by `dhistout`, so the file name stays the same. Created if missing. |
| `--d21out DIR` | `d21out` | Output directory for corrected future files (`d21` replaced by `d21out`). |
| `--dobs DIR` | `dobs` | Directory with the observations aggregated to the GCM grid, named `<var>_*.nc`. Only files from the one whose name contains `ref_st` to the one whose name contains `ref_end` are used. |
| `--var NAME` | `var` | Variable name in both the GCM and obs files. Names containing `tas` (`tas`, `tasmax`, `tasmin`) are treated as temperature. Every other variable is treated like precipitation. |
| `--dec N` | `dec` | 0-based index of the file to correct in the combined, sorted list (historical files first, then future). |
| `--ref-st YEAR` | `ref_st` | First year of the reference (training) period. |
| `--ref-end YEAR` | `ref_end` | Last year of the reference period. |

### Optional parameters

| Parameter | Default | Description |
|---|---|---|
| `--min-lat`, `--max-lat` | -90, 90 | Latitude bounds of the processed box (degrees). Ascending and descending latitude both work. |
| `--min-lon`, `--max-lon` | 0, 360 | Longitude bounds (degrees, in the files' own convention). |
| `--pad YEARS` | 4 | Half width of the `tas` trend window. The window is `2*pad+1` years. |
| `--buff DAYS` | 15 | Half width of the day-of-year window used to build the CDFs. The window is `2*buff+1` days. |
| `--chunk-lat N`, `--chunk-lon N` | 32, 32 | Tile size in grid points. Sets the memory per tile. |
| `--workers N` | all cores | Number of tiles processed in parallel (dask threads). |
| `--overwrite` | off | Recompute even if the output file exists. Without it, existing files are skipped. |
| `--nc-cache-mb MB` | 4 | netCDF/HDF5 chunk cache per variable and per open file, set before any file is opened (see [Memory and performance](#memory-and-performance)). |

### Input requirements

- NetCDF files with dimensions named `time`, `lat` and `lon`, and data shaped `(time, lat, lon)`.
- Times must be CF-decodable. Any cftime calendar works. For `360_day` calendars a 360-day year is
  used, otherwise 365 days.
- Within the lat/lon box, the GCM and obs grids must have the same size. They are combined by
  index, as in NCL, and the output takes the obs coordinates. A size mismatch raises an error.
- GCM files must together span whole years without gaps. One reference year must not appear in
  two files.

### Output

- NetCDF4-classic, compression level 5, unlimited `time` dimension.
- Variable stored as float32 with `_FillValue = 1e20`, keeping `units`, `long_name` and
  `standard_name` from the GCM file.
- Time values and encoding (units, calendar) are copied from the input file. lat/lon come from the
  obs files.
- The file is written as `<name>.part` and renamed when complete. An interrupted run therefore
  never leaves a file that blocks a rerun.

### Memory and performance

Memory per tile is roughly

```
(days in file + days in reference period + obs days) × tile points × 4 bytes × 3
```

For example, a decade with a 30-year reference period and 32×32 tiles needs about 340 MB per tile.
Peak memory is about a few hundred MB of baseline plus `workers` × that amount. Reduce
`--chunk-lat`/`--chunk-lon` or `--workers` on small machines. The script logs the estimate before
computing.

Measured on Levante with a 160×200 grid (about 1.9 GB of pr input) and 32×40 tiles:

| Workers | Peak memory |
|---|---|
| 1 | ~0.6 GB |
| 4 | ~1.7 GB |

**netCDF chunk cache.** Every open file keeps its own HDF5 chunk cache. netCDF's default is
64 MB, which grows to several GB with dozens of input files, so the port lowers it to 4 MB.
Each tile reads every HDF5 chunk only once, so a large cache rarely helps. If your files have
large compressed chunks that several reads share, it can be raised with `--nc-cache-mb` (or
`bc_daily(..., nc_chunk_cache=bytes)`). The cost is up to that size × the number of open files.

One 1,280-point tile for one decade takes about 16 s. Larger tiles mean fewer, bigger reads.
Very small tiles on files chunked per time step make each file be read many times over.

## Testing

```bash
uv sync                  # includes pytest (dev dependency)
uv run pytest -q         # ~1 min, no NCL needed
```

- `tests/test_bcfuncs.py`: unit tests of the numpy functions.
  - `lookup` must be bit-identical to a literal loop translation of the NCL function, including
    ties, values exactly on nodes and values outside the range.
  - The row-wise binary search, `stat4`, the NCL Weibull clipping quirk, and float32 rounding of
    `exp` are also tested.
- `tests/test_bc_daily.py`: end-to-end runs on small synthetic data (`pr` and `tas`). They check
  the output format, the obs mask, non-negative precipitation, that results do not depend on the
  tile size, and that existing outputs are skipped.

### Comparison with NCL

`tests/ncl_compare/run_comparison.sh` generates synthetic data and runs both `BCDaily.ncl` and the
Python port on four cases, then compares the outputs:

```bash
tests/ncl_compare/run_comparison.sh /scratch/$USER/bcsd_compare [calendar]
```

It needs NCL; on Levante the script loads `ncl/6.6.2-gcc-11.2.0`. The NCL runs take several minutes,
and existing NCL outputs are reused on later runs. Last result:

| Case | Agreement with NCL |
|---|---|
| pr, decade inside the reference period | bit-identical |
| pr, future decade (EVI tail extrapolation) | bit-identical |
| tas, decade inside the reference period | bit-identical |
| tas, future decade (trend) | max 3e-5 K (one float32 rounding step), in 1 of ~290,000 values |

Run this comparison after any change to the numerical code.

## Limitations

- **Only the BC step is ported.** The spatial disaggregation step (`SDDaily.ncl`) still needs NCL.
- **NCL quirks are reproduced on purpose**, so that results match the published datasets:
  - the Weibull table's upper clip has no effect, so skewness > 8.214 gives a missing value in
    that tail;
  - the obs day-of-year window wraps at a hard-coded 365, even for 360-day GCM calendars;
  - the leap day (Dec 31 in leap years, day-of-year index 365) is never part of a CDF window.
- **Grids are matched by index.** Obs must already be aggregated to the GCM grid, and coordinate
  values are not checked, only sizes.
- **Fixed dimension names**: `time`, `lat`, `lon`.
- **One file per run.** Every run re-reads the reference period and, for `tas`, all GCM files to
  build the trend.
- **Threads only.** Parallelism uses dask's threaded scheduler within one process. Use a job array
  across `--dec` values to spread work over nodes.
- **Edge cases may differ from NCL:**
  - GCM files with gaps (missing months): NCL places monthly means by position, Python by date;
  - GCM files whose time units differ from file to file: NCL does not convert them, Python
    decodes each file correctly;
  - obs with missing values at unmasked points, where the position of missing values after
    sorting may differ.
- **Deliberate differences from NCL:**
  - a single historical file is used (NCL's `nfiles_hist.gt.1` check drops it);
  - the output directory is created if missing;
  - output is written atomically;
  - variable attributes are kept.

## Contributing

1. Create a branch from `main`. Use `uv sync` for the environment, and `uv add <pkg>` (or
   `uv add --dev <pkg>`) to add dependencies, so that `pyproject.toml` and `uv.lock` stay in sync.
2. Keep numerical behaviour identical to NCL unless a change is intended:
   - run `uv run pytest -q` for every change;
   - run `tests/ncl_compare/run_comparison.sh` for any change to `bcfuncs.py`,
     `bias_correct_block` or `monthly_trend`;
   - describe any intended deviation from NCL in the commit message and under
     [Limitations](#limitations).
3. Keep float32 semantics in the core. Do not let float64 constants upcast float32 arrays: use
   Python floats or float32 arrays. Evaluate `exp`/`log`/`pow` through `_exp`/`_log`/`_pow`.
4. Keep everything lazy. Avoid `.values`/`.load()` on data variables in the driver, and avoid
   operations that change the spatial chunking (e.g. dask `reshape`).
5. Add or adjust tests in `tests/` together with code changes.

## License

CC0 1.0 Universal; see [LICENSE](LICENSE).
