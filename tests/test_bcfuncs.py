"""Unit tests for the numpy port of BCFuncs.ncl (no NCL needed)."""

import numpy as np
import pytest

from daily_bcsd import bcfuncs as bf


def lookup_loop(q, v1, v2, par, lu, mean, stdev, skew):
    """Literal translation of NCL `lookup` (sample axis first), as the reference."""
    n = v1.shape[0]
    tiny = 1.0 / (n + 1.0)
    val = np.zeros_like(q)
    for ndx in range(n - 1):
        d = v1[ndx + 1] - v1[ndx]
        d = np.where(d == 0, v1[ndx + 1] - q, d)
        d = np.where(d == 0, -9999.0, d)
        A = (v1[ndx + 1] - q) / d
        hit = (q <= v1[ndx + 1]) & (q >= v1[ndx]) & (val == 0)
        val = np.where(hit, A * v2[ndx] + (1 - A) * v2[ndx + 1], val)
    lo, hi = q < v1[0], q > v1[-1]
    if lu == bf.QUAN and par == bf.PRCP:
        val = np.where(lo, bf.get_F_from_data_weibul(mean, stdev, skew, q, tiny), val)
        val = np.where(hi, bf.get_F_from_data_EVI(mean, stdev, q, tiny), val)
    elif lu == bf.QUAN:
        val = np.where(lo | hi, bf.get_F_from_data_normal(mean, stdev, q, tiny), val)
    elif par == bf.PRCP:
        val = np.where(lo, bf.get_data_from_F_weibul(mean, stdev, skew, q), val)
        val = np.where(hi, bf.get_data_from_F_EVI(mean, stdev, q), val)
    else:
        val = np.where(lo | hi, bf.get_data_from_F_normal(mean, stdev, q), val)
    return val


def samples(par, rng, shape):
    if par == bf.PRCP:  # many exact zeros -> ties
        return np.where(rng.random(shape) < 0.5, 0.0, rng.gamma(0.7, 5.0, shape)).astype(np.float32)
    return rng.normal(280.0, 3.0, shape).round(1).astype(np.float32)  # rounding -> ties


@pytest.mark.parametrize("par", [bf.TEMP, bf.PRCP])
def test_lookup_matches_ncl_loop(par):
    rng = np.random.default_rng(1)
    P, n = 2000, 200
    s = np.sort(samples(par, rng, (P, n)), axis=-1)
    quant = (np.arange(1, n + 1, dtype=np.float32) / np.float32(n + 1))
    mean, var, skew = bf.stat4(s)
    sd = np.sqrt(var)
    g = samples(par, rng, (P,)) * np.float32(1.3)  # some values beyond the sample range
    g[:50] = s[:50, 7]  # exactly on nodes

    with np.errstate(all="ignore"):
        q = bf.lookup(g[:, None], s, quant, par, bf.QUAN, mean, sd, skew)[:, 0]
        q_ref = lookup_loop(g, s.T, np.broadcast_to(quant[:, None], s.T.shape), par, bf.QUAN, mean, sd, skew)
        np.testing.assert_array_equal(q, q_ref)

        x = bf.lookup(q[:, None], quant, s, par, bf.DAT, mean, sd, skew)[:, 0]
        x_ref = lookup_loop(q, np.broadcast_to(quant[:, None], s.T.shape), s.T, par, bf.DAT, mean, sd, skew)
        np.testing.assert_array_equal(x, x_ref)


def test_lookup_propagates_missing():
    v1 = np.sort(np.random.default_rng(0).normal(size=(3, 50)), axis=-1).astype(np.float32)
    v1[2, -1] = np.nan
    q = np.array([[np.nan], [0.1], [0.1]], dtype=np.float32)
    quant = np.arange(1, 51, dtype=np.float32) / np.float32(51)
    m, v, s = bf.stat4(v1)
    out = bf.lookup(q, v1, quant, bf.TEMP, bf.QUAN, m, np.sqrt(v), s)
    assert np.isnan(out[0, 0]) and not np.isnan(out[1, 0]) and np.isnan(out[2, 0])


def test_searchsorted_rowwise():
    rng = np.random.default_rng(2)
    a = np.sort(rng.integers(0, 20, (50, 31)).astype(np.float32), axis=-1)
    v = rng.integers(-2, 22, (50, 7)).astype(np.float32)
    for side in ("left", "right"):
        ref = np.stack([np.searchsorted(a[i], v[i], side=side) for i in range(len(a))])
        np.testing.assert_array_equal(bf._searchsorted(a, v, side), ref)


def test_stat4():
    rng = np.random.default_rng(3)
    x = rng.gamma(2.0, 3.0, (10, 500)).astype(np.float32)
    x[0, :100] = np.nan
    mean, var, skew = bf.stat4(x)
    assert mean.dtype == np.float32
    for i in range(len(x)):
        xi = x[i][~np.isnan(x[i])].astype(np.float64)
        sd = xi.std(ddof=1)
        np.testing.assert_allclose(mean[i], xi.mean(), rtol=1e-6)
        np.testing.assert_allclose(var[i], sd**2, rtol=1e-6)
        np.testing.assert_allclose(skew[i], np.mean(((xi - xi.mean()) / sd) ** 3), rtol=1e-5)


def test_weibul_params_keeps_ncl_clipping():
    shape, loc, bnd = bf.weibul_params(np.array([-5.0, 2.0, 8.214, 9.0], dtype=np.float32))
    # below the table is clipped to the first entry
    np.testing.assert_allclose([shape[0], loc[0], bnd[0]], [1 / 0.02, 0.446, 40.005], rtol=1e-6)
    np.testing.assert_allclose([shape[1], loc[1], bnd[1]], [1.0, 0.0, 1.0], rtol=1e-6)
    assert np.isfinite(shape[2])
    # above the table: NCL's upper clip is overwritten -> missing
    assert np.isnan(shape[3]) and np.isnan(loc[3]) and np.isnan(bnd[3])


def test_float32_tail_saturation_like_ncl():
    # exp(-3.8e-8) rounds to 0.99999994 in float32; numpy's own float32 exp gives 1.0
    F = bf._exp(np.float32(-3.806905e-08))
    assert F.dtype == np.float32 and F < 1
