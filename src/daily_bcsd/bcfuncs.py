"""Vectorised numpy port of BCFuncs.ncl.

Distribution functions are elementwise.  Functions working on samples
(``stat4``, ``lookup``) expect the sample axis to be the *last* axis, i.e.
arrays shaped ``(..., n)`` -- the layout ``xarray.apply_ufunc`` hands to a
function with a core dimension.

The NCL behaviour is reproduced on purpose, including its quirks (noted where
they occur), so that results match the original BCSD datasets.  Pass float32
arrays to get NCL's float arithmetic: it matters in the tails, where e.g. the
EVI CDF saturates to 1 (and is then clipped to 1 - TINY) far sooner in float32.
Constants are Python floats or float32 so they do not upcast.
"""

import math

import numpy as np

# `par` and `lu` arguments of `lookup`
TEMP, PRCP = 0, 1
QUAN, DAT = 0, 1  # map data -> quantile, or quantile -> data

_SK = np.array([-1.000, -0.971, -0.917, -0.867, -0.638, -0.254, 0.069, 0.359, 0.631, 0.896, 1.160,
                1.430, 1.708, 2.000, 2.309, 2.640, 2.996, 3.382, 3.802, 4.262, 4.767, 5.323, 5.938,
                6.619, 7.374, 8.214], dtype=np.float32)
_INVA = np.array([0.020, 0.030, 0.040, 0.050, 0.100, 0.200, 0.300, 0.400, 0.500, 0.600, 0.700,
                  0.800, 0.900, 1.000, 1.100, 1.200, 1.300, 1.400, 1.500, 1.600, 1.700, 1.800,
                  1.900, 2.000, 2.100, 2.200], dtype=np.float32)
_AVEC = np.array([0.446, 0.444, 0.442, 0.439, 0.425, 0.389, 0.346, 0.297, 0.246, 0.193, 0.142,
                  0.092, 0.044, 0.000, -0.040, -0.077, -0.109, -0.136, -0.160, -0.180, -0.196,
                  -0.208, -0.217, -0.224, -0.227, -0.229], dtype=np.float32)
_BVEC = np.array([40.005, 26.987, 20.481, 16.576, 8.737, 4.755, 3.370, 2.634, 2.159, 1.815, 1.549,
                  1.334, 1.154, 1.000, 0.867, 0.752, 0.652, 0.563, 0.486, 0.418, 0.359, 0.308,
                  0.263, 0.224, 0.190, 0.161], dtype=np.float32)


def _libm(func):
    """``func`` evaluated in double and rounded to the argument precision.

    That is what NCL does for float arguments; numpy's own float32 exp/log/pow
    are not correctly rounded, which flips e.g. the EVI CDF between 1 - 6e-8
    and exactly 1 (then clipped to 1 - TINY) and so changes tail values.
    """
    def wrapped(*args):
        dtype = np.result_type(*args)
        return func(*(np.asarray(a, dtype=np.float64) for a in args)).astype(dtype)
    return wrapped


_exp, _log, _pow = _libm(np.exp), _libm(np.log), _libm(np.power)


def weibul_params(skew):
    """Weibull (shape, location factor, bound factor) interpolated from skewness.

    As in NCL, the clip at the top of the table is overwritten by the clip at
    the bottom, so skewness above 8.214 gives NaN.
    """
    skw = np.where(skew < _SK[0], _SK[0], skew)
    k = np.clip(np.searchsorted(_SK, skw, side="right") - 1, 0, len(_SK) - 2)
    a = (_SK[k + 1] - skw) / (_SK[k + 1] - _SK[k])
    ok = skw <= _SK[-1]  # False for NaN
    shape = np.where(ok, 1.0 / (a * _INVA[k] + (1 - a) * _INVA[k + 1]), np.nan)
    loc = np.where(ok, a * _AVEC[k] + (1 - a) * _AVEC[k + 1], np.nan)
    bnd = np.where(ok, a * _BVEC[k] + (1 - a) * _BVEC[k + 1], np.nan)
    return shape, loc, bnd


def get_data_from_F_weibul(mean, stdev, skew, F):
    shape, loc, bnd = weibul_params(skew)
    b = loc * stdev + mean
    bound = b - bnd * stdev
    bound = np.where(bound < 0, 0.0, bound)
    # NCL's unary minus binds tighter than ^: (-log(1-F))^(1/shape)
    return _pow(-_log(1 - F), 1 / shape) * (b - bound) + bound


def get_F_from_data_weibul(mean, stdev, skew, x, tiny):
    shape, loc, bnd = weibul_params(skew)
    b = loc * stdev + mean
    bound = b - bnd * stdev
    bound = np.where((bound <= 0) | (bound > x), 0.0, bound)
    diff = b - bound
    diff = np.where(diff == 0, np.nan, diff)
    F = 1 - _exp(-_pow((x - bound) / diff, shape))
    F = np.where(F >= 1.0, F - tiny, F)
    return np.where(np.isnan(F) | (F < tiny), tiny, F)


def get_data_from_F_EVI(mean, stdev, F):
    std = np.where(stdev == 0, 1e-20, stdev)
    b = 3.14159 / (std * math.sqrt(6))
    a = mean - 0.5772 / b
    return a - (1 / b) * _log(-_log(F))


def get_F_from_data_EVI(mean, stdev, x, tiny):
    std = np.where(stdev == 0, 1e-20, stdev)
    b = 3.14159 / (std * math.sqrt(6))
    a = mean - 0.5772 / b
    F = _exp(-_exp(-b * (x - a)))
    F = np.where(F >= 1.0, F - tiny, F)
    return np.where(F < tiny, tiny, F)


def get_data_from_F_normal(mean, stdev, F):
    z = (_pow(F, 0.135) - _pow(1 - F, 0.135)) / 0.1975
    return z * stdev + mean


def get_F_from_data_normal(mean, stdev, x, tiny):
    fill = 1e-20
    std = np.where(stdev == 0, fill, stdev)
    z = (x - mean) / std
    neg = z < 0
    z = np.abs(z)
    z = np.where(z == 0.0, fill, z)
    F = 1 - 0.5 * _exp(-((83 * z + 351) * z + 562) / (165 + 703 / z))
    F = np.where(F >= 1.0, F - tiny, F)
    F = np.where(F < tiny, tiny, F)
    return np.where(neg, 1 - F, F)


def stat4(x):
    """Mean, sample variance and skewness along the last axis, ignoring NaN.

    Same definitions as NCL's dim_stat4_n: variance uses n-1, skewness is
    mean(((x - mean) / sd)**3) with that sample sd.  Like NCL, sums are done
    in double and the results returned in the input precision.
    """
    dtype = x.dtype
    valid = ~np.isnan(x)
    n = valid.sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(valid, x, 0.0).sum(axis=-1, dtype=np.float64) / n
        dev = np.where(valid, x - mean[..., None], 0.0)
        dev2 = dev * dev
        var = dev2.sum(axis=-1) / (n - 1)
        skew = (dev2 * dev).sum(axis=-1) / n / var**1.5
    return mean.astype(dtype), var.astype(dtype), skew.astype(dtype)


def _searchsorted(a, v, side):
    """Row-wise np.searchsorted: ``a`` (..., n) sorted along its last axis, ``v`` (..., m).

    NaN in ``a`` (which np.sort puts last) acts as +inf; NaN in ``v`` gives 0.
    """
    n = a.shape[-1]
    lo = np.zeros(v.shape, dtype=np.intp)
    hi = np.full(v.shape, n, dtype=np.intp)
    for _ in range(n.bit_length()):
        mid = (lo + hi) // 2
        am = np.take_along_axis(a, np.minimum(mid, n - 1), axis=-1)
        active = lo < hi
        go_right = active & ((am <= v) if side == "right" else (am < v))
        lo = np.where(go_right, mid + 1, lo)
        hi = np.where(active & ~go_right, mid, hi)
    return lo


def lookup(q, v1, v2, par, lu, mean, stdev, skew):
    """Map ``q`` through the piecewise-linear relation ``v1 -> v2`` (NCL ``lookup``).

    q:      (..., m) values to map.
    v1, v2: (..., n) or (n,), ``v1`` sorted ascending along the last axis.
    mean, stdev, skew: (...) parameters of the fitted distribution used where
            ``q`` falls outside the range of ``v1`` (Weibull below / EVI above
            for PRCP, normal for TEMP).
    """
    v1 = np.broadcast_to(v1, q.shape[:-1] + v1.shape[-1:])
    v2 = np.broadcast_to(v2, v1.shape)
    mean, stdev, skew = (np.asarray(p)[..., None] for p in (mean, stdev, skew))
    n = v1.shape[-1]
    tiny = 1.0 / (n + 1.0)

    def interp(k):
        a0 = np.take_along_axis(v1, k, axis=-1)
        a1 = np.take_along_axis(v1, k + 1, axis=-1)
        diff = a1 - a0
        diff = np.where(diff == 0, a1 - q, diff)
        diff = np.where(diff == 0, -9999.0, diff)
        A = (a1 - q) / diff
        return A * np.take_along_axis(v2, k, axis=-1) + (1 - A) * np.take_along_axis(v2, k + 1, axis=-1)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        # NCL loops over segments ndx and fills points where v1[ndx] <= q <= v1[ndx+1]
        # and val is still 0.  Those segments form the contiguous range [k_lo, k_hi].
        k_lo = np.maximum(_searchsorted(v1, q, "left") - 1, 0)
        k_hi = np.minimum(_searchsorted(v1, q, "right") - 1, n - 2)
        inside = k_lo <= k_hi
        k = np.minimum(k_lo, n - 2)
        val = np.where(inside, interp(k), 0.0)
        # An interpolated value of exactly 0 is overwritten by the next matching segment.
        redo = inside & (val == 0) & (k < k_hi)
        while redo.any():
            k = np.where(redo, k + 1, k)
            val = np.where(redo, interp(k), val)
            redo &= (val == 0) & (k < k_hi)

        below = q < v1[..., :1]
        above = q > v1[..., -1:]
        if below.any() or above.any():
            if lu == QUAN and par == PRCP:
                val = np.where(below, get_F_from_data_weibul(mean, stdev, skew, q, tiny), val)
                val = np.where(above, get_F_from_data_EVI(mean, stdev, q, tiny), val)
            elif lu == QUAN:
                val = np.where(below | above, get_F_from_data_normal(mean, stdev, q, tiny), val)
            elif par == PRCP:
                val = np.where(below, get_data_from_F_weibul(mean, stdev, skew, q), val)
                val = np.where(above, get_data_from_F_EVI(mean, stdev, q), val)
            else:
                val = np.where(below | above, get_data_from_F_normal(mean, stdev, q), val)

    # NCL's where() returns missing wherever its condition involves a missing value.
    return np.where(np.isnan(q) | np.isnan(v1).any(axis=-1, keepdims=True), np.nan, val)
