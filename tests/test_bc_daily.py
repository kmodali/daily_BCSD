"""End-to-end test of the python BC step on small synthetic data (no NCL needed)."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from daily_bcsd import bc_daily

HERE = Path(__file__).parent


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth")
    subprocess.run([sys.executable, str(HERE / "ncl_compare" / "make_synth.py"), str(root)], check=True)
    return root


def run(root, var, dec, chunks):
    r = root / var
    return bc_daily(str(r / "hist"), str(r / "rcp"), str(r / "out_hist"), str(r / "out_rcp"), str(r / "obs"),
                    var, dec, 1970, 1989, -90.0, 90.0, 0.0, 360.0, chunk_lat=chunks[0], chunk_lon=chunks[1],
                    workers=2, overwrite=True)


@pytest.mark.parametrize("var,dec", [("pr", 7), ("tas", 6)])
def test_output_and_tiling_invariance(synth, var, dec):
    out = xr.open_dataset(run(synth, var, dec, (3, 4)), decode_times=False)
    src = xr.open_dataset(sorted((synth / var / "rcp").glob("*.nc"))[dec - 5], decode_times=False)
    x = out[var].values

    assert out[var].dtype == np.float32 and x.shape == src[var].shape
    np.testing.assert_array_equal(out.time.values, src.time.values)
    assert out.time.attrs["calendar"] == "noleap"
    # obs-masked points stay missing, all others are filled
    assert np.isnan(x[:, 0, 0]).all() and np.isnan(x[:, 3, 7]).all()
    assert np.isnan(x).sum() == 2 * x.shape[0]
    if var == "pr":
        assert np.nanmin(x) >= 0

    # the result must not depend on the tile size
    other = xr.open_dataset(run(synth, var, dec, (8, 10)), decode_times=False)[var].values
    np.testing.assert_array_equal(x, other)


def test_existing_output_is_skipped(synth):
    path = run(synth, "pr", 2, (8, 10))
    mtime = Path(path).stat().st_mtime
    r = synth / "pr"
    bc_daily(str(r / "hist"), str(r / "rcp"), str(r / "out_hist"), str(r / "out_rcp"), str(r / "obs"),
             "pr", 2, 1970, 1989, -90.0, 90.0, 0.0, 360.0)
    assert Path(path).stat().st_mtime == mtime
