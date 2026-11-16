
import minkasi.minkasi_all as minkasi
import numpy as np
import pytest
from minkasi.maps.skymap import SkyMap
from minkasi.needlet.needlet import (
    Needlet,
    WavSkyMap,
    band2map,
    map2band,
    map2wav_real,
    wav2map_real,
)

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


def test_mpi_reduce_chunks_whole_wavelet_map(monkeypatch):
    """The chunked MPI reduction must cover all bands of a (nfilt, nx, ny) map, not only nx * ny."""
    from minkasi.maps import skymap

    class FakeComm:
        def allreduce(self, x):
            return 2 * x

    monkeypatch.setattr(skymap, "have_mpi", True)
    monkeypatch.setattr(skymap, "comm", FakeComm(), raising=False)
    lims = [0.0, 0.01, 0.0, 0.01]
    smap = SkyMap(lims, 30.0 / 3600 * np.pi / 180, square=True, multiple=2)
    smap.map = np.ones((3, *smap.map.shape[-2:]))
    smap.mpi_reduce(chunksize=smap.map.shape[-1] * smap.map.shape[-2] / 2)
    assert np.all(smap.map == 2)


def test_downsamps_from_filter_support_satisfy_nyquist(wmap):
    """With tol the lattice of every band samples above twice the highest wavenumber of its filter."""
    try:
        wmap.get_downsamps(1e-3)
        kx = np.fft.fftshift(np.fft.fftfreq(wmap.nx)) * wmap.nx
        ky = np.fft.fftshift(np.fft.fftfreq(wmap.ny)) * wmap.ny
        kmag = np.hypot(*np.meshgrid(kx, ky, indexing="ij"))
        for filt, down in zip(wmap.needlet.filters, wmap.downsamps):
            kmax = kmag[np.abs(filt) > 1e-3].max()
            assert down == 1 or 2 * kmax * down <= wmap.nx
    finally:
        wmap.get_downsamps()
