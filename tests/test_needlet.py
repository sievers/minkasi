import numpy as np
import pytest
from scipy import sparse

import minkasi.minkasi_all as minkasi
from minkasi.maps.mapset import Mapset
from minkasi.needlet.needlet import (
    Needlet,
    WavSkyMap,
    band2map,
    map2band,
    map2wav_real,
    wav2map_real,
)
from minkasi.maps.skymap import SkyMap

PIXSIZE = 30.0 / 3600 * np.pi / 180  # radians


def make_wmap(todvec, njs=4):
    """Build a WavSkyMap with njs filters covering todvec."""
    lims = todvec.lims()
    box = SkyMap(lims, PIXSIZE, square=True, multiple=2).map
    need = Needlet(
        np.arange(njs),
        lightcone=box,
        L=box.shape[0] * 30.0 / 60 * np.sqrt(2),
        pixsize=30.0,
    )
    need.get_needlet_filters_2d(
        need.lightcone_box.get_grid_dimless_2d(return_grid=True)
    )
    wmap = WavSkyMap(need, lims, PIXSIZE, square=True, multiple=2)
    for tod in todvec.tods:
        tod.info["ipix"] = wmap.get_pix(tod)
    return wmap


@pytest.fixture(scope="module")
def todvec():
    """Two small synthetic scanning TODs with white noise."""
    rng = np.random.default_rng(0)
    tv = minkasi.TodVec()
    ndet, nt = 6, 400
    t = np.linspace(0, 1, nt)
    for k in range(2):
        off = np.linspace(-1, 1, ndet)[:, None] * 0.7e-3
        dx = 1.5e-3 * np.sin(2 * np.pi * (3 + k) * t)[None] + off
        dy = 1.5e-3 * np.cos(2 * np.pi * (2 + k) * t)[None] + off[::-1]
        info = {
            "dx": dx + 1.0,
            "dy": dy + 0.5,
            "dat_calib": rng.normal(size=(ndet, nt)),
        }
        tod = minkasi.Tod(info)
        tod.set_noise(minkasi.NoiseWhite)
        tv.add_tod(tod)
    return tv


@pytest.fixture(scope="module")
def wmap(todvec):
    return make_wmap(todvec)


def test_band_helpers_match_full_transforms(wmap):
    rng = np.random.default_rng(1)
    m = rng.normal(size=(wmap.nx, wmap.ny))
    full = map2wav_real(m, wmap.needlet.filters)[0]
    back = wav2map_real(full, wmap.needlet.filters)[0]
    for i, f in enumerate(wmap.needlet.filters):
        assert np.allclose(map2band(m, f), full[i], atol=1e-10)
        only = np.zeros_like(full)
        only[i] = full[i]
        assert np.allclose(
            band2map(full[i], f),
            wav2map_real(only, wmap.needlet.filters)[0],
            atol=1e-10,
        )
    assert back.shape == m.shape


def brute_force(wmap, todvec, nfilt):
    """Reference ANA: one TOD pass per retained pixel through the original full transforms."""
    mapset = Mapset()
    mapset.add_map(wmap.real_map.copy())
    xs, ys = wmap.nx_space[nfilt], wmap.ny_space[nfilt]
    out = np.zeros((len(xs) * len(ys),) * 2)
    for r, (x, y) in enumerate((x, y) for x in xs for y in ys):
        imp = np.zeros((wmap.needlet.nfilt, wmap.nx, wmap.ny))
        imp[nfilt, x, y] = 1
        mapset.maps[0].map[:] = wav2map_real(imp[None], wmap.needlet.filters)
        acc = todvec.dot(mapset, cache_maps=False).maps[0].map
        z = map2wav_real(acc, wmap.needlet.filters)[0, nfilt]
        out[r] = z[np.ix_(xs, ys)].ravel()
    return out


def test_exact_response_matches_brute_force(wmap, todvec):
    down = np.array([1] * (wmap.needlet.nfilt - 1) + [2])
    resp = wmap.get_response_matrix(todvec, nfilts=[1, 3], downsamps=down)
    for i in (1, 3):
        ref = brute_force(wmap, todvec, i)
        assert np.allclose(
            resp[i].toarray(), ref, rtol=1e-6, atol=1e-8 * np.abs(ref).max()
        )
    assert resp[0] is None


def test_probed_response_close_to_exact(wmap, todvec):
    down = np.ones(wmap.needlet.nfilt, dtype=int)
    exact = wmap.get_response_matrix(todvec, nfilts=[2], downsamps=down)[2].toarray()
    approx = wmap.get_response_matrix(
        todvec, nfilts=[2], downsamps=down, radius=wmap.nx // 3
    )[2].toarray()
    assert np.abs(approx - exact).max() < 0.05 * np.abs(exact).max()


def test_map_response_diagonal_and_matrix_agree(wmap):
    rng = np.random.default_rng(2)
    ivar = rng.uniform(1, 2, size=(wmap.nx, wmap.ny))
    down = np.ones(wmap.needlet.nfilt, dtype=int)
    wmap.set_downsamps(down)
    a = wmap.get_response_matrix_map(ivar, nfilts=[2])[2]
    b = wmap.get_response_matrix_map(sparse.diags(ivar.ravel()), nfilts=[2])[2]
    assert np.allclose(a.toarray(), b.toarray())
    assert np.allclose(a.toarray(), a.toarray().T)


def test_solve_bands_recovers_input(wmap):
    """With N^-1 = identity and no TODs, G c = K^T K c must invert K^T K c_true."""
    down = np.array([1, 1, 1, 2])
    wmap.set_downsamps(down)
    rng = np.random.default_rng(3)
    ones = np.ones((wmap.nx, wmap.ny))
    resp = wmap.get_response_matrix_map(ones, nfilts=[1, 3])
    truth = np.zeros((wmap.needlet.nfilt, wmap.nx, wmap.ny))
    for i in (1, 3):
        xs, ys = wmap.nx_space[i], wmap.ny_space[i]
        truth[i][np.ix_(xs, ys)] = rng.normal(size=(len(xs), len(ys)))
    rhs = np.zeros_like(truth)
    for i in (1, 3):
        k = band2map(truth[i], wmap.needlet.filters[i])
        rhs[i] = map2band(k, wmap.needlet.filters[i])
    sol = wmap.solve_bands(rhs, [resp], ridge=1e-9)
    for i in (1, 3):
        xs, ys = wmap.nx_space[i], wmap.ny_space[i]
        got = band2map(sol[i], wmap.needlet.filters[i])
        want = band2map(truth[i], wmap.needlet.filters[i])
        assert np.abs(got - want).max() < 1e-3 * np.abs(want).max()
