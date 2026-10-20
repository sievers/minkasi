import minkasi.minkasi_all as minkasi
import numpy as np
import pytest
from minkasi.maps.mapset import Mapset
from minkasi.maps.skymap import SkyMap
from minkasi.needlet.needlet import (
    Needlet,
    WavSkyMap,
    apply_beam,
    band2map,
    color_groups,
    gaussian_beam_ft,
    map2band,
    map2wav_real,
    wav2map_real,
)
from scipy import sparse

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
        todvec, nfilts=[2], downsamps=down, radius=wmap.nx // 3, exact_below=0
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


def test_solve_bands_white_noise_is_identity(wmap):
    """With unit precision the filters partition unity, so the solve must return the rhs map."""
    wmap.set_downsamps(np.ones(wmap.needlet.nfilt, dtype=int))
    rng = np.random.default_rng(3)
    d = np.fft.irfft2(
        np.fft.rfft2(rng.normal(size=(wmap.nx, wmap.ny)))
        * (
            np.hypot(
                *np.meshgrid(
                    np.fft.fftfreq(wmap.nx), np.fft.rfftfreq(wmap.ny), indexing="ij"
                )
            )
            < 0.15
        ),
        s=(wmap.nx, wmap.ny),
    )  # band limited so no mode is lost below the highest filter
    resp = wmap.get_response_matrix_map(np.ones((wmap.nx, wmap.ny)))
    rhs = map2wav_real(d, wmap.needlet.filters)[0]
    out = wav2map_real(wmap.solve_bands(rhs, [resp], ridge=1e-6), wmap.needlet.filters)[
        0
    ]
    assert np.abs(out - d).max() < 0.02 * np.abs(d).max()


def test_solve_bands_handles_indefinite_precision(wmap):
    """A truncated, indefinite G must still give a finite solution of similar size."""
    wmap.set_downsamps(np.full(wmap.needlet.nfilt, 2))
    resp = wmap.get_response_matrix_map(np.ones((wmap.nx, wmap.ny)), nfilts=[2])
    G = resp[2].tolil()
    G.setdiag(G.diagonal() * 0.3)  # breaks positive definiteness
    rhs = np.random.default_rng(4).normal(size=(wmap.needlet.nfilt, wmap.nx, wmap.ny))
    out = wmap.solve_bands(rhs, [np.array([None, None, G.tocsr(), None], dtype=object)])
    assert np.isfinite(out).all()
    assert np.abs(out).max() < 100 * np.abs(rhs).max()


def test_map_response_with_beam_matches_dense(wmap):
    """Beam-observed map response equals K^T B D B K computed row by row."""
    rng = np.random.default_rng(5)
    ivar = rng.uniform(1, 2, size=(wmap.nx, wmap.ny))
    beam = gaussian_beam_ft((wmap.nx, wmap.ny), 6.0)
    wmap.set_downsamps(np.full(wmap.needlet.nfilt, 2))
    got = wmap.get_response_matrix_map(
        ivar, nfilts=[2], forward=lambda m: apply_beam(m, beam)
    )[2].toarray()
    filt = wmap.needlet.filters[2]
    xs, ys = wmap.nx_space[2], wmap.ny_space[2]
    for r in (0, 100, len(got) - 1):
        delta = np.zeros((wmap.nx, wmap.ny))
        delta[xs[r // len(ys)], ys[r % len(ys)]] = 1
        z = map2band(
            apply_beam(ivar * apply_beam(band2map(delta, filt), beam), beam), filt
        )
        assert np.allclose(got[r], z[np.ix_(xs, ys)].ravel(), atol=1e-10)
    assert np.allclose(got, got.T)


@pytest.mark.parametrize("n,width", [(55, 21), (110, 7), (484, 3), (10, 40)])
def test_color_groups_tile_and_separate(n, width):
    groups = color_groups(n, width)
    assert sorted(np.concatenate(groups)) == list(range(n))
    for g in groups:
        gaps = np.diff(np.concatenate([g, [g[0] + n]]))
        assert len(g) == 1 or gaps.min() >= width
