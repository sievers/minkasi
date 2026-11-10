# Code changes that fixed problems in the needlet code

Baseline: `3c435a3` (merge of master into `needlet`), the code as it was before the first fix commit `f92ca40`.
To see exactly what changed in the library since then:

```
git diff 3c435a3 HEAD -- minkasi/needlet minkasi/maps/skymap.py
git diff 3c435a3 HEAD -- examples/twores_ex.py tests/test_needlet.py tests/test_cmb.py
git log --oneline 3c435a3..HEAD -- minkasi examples tests
```

Files changed since the baseline: `minkasi/needlet/needlet.py` (about 1000 lines), `minkasi/needlet/cmb.py` (new),
`minkasi/maps/skymap.py` (46 lines), `minkasi/tools/noise_estimation.py` (73 lines, not examined for this note),
`examples/twores_ex.py` (about 1000 lines), new example and test scripts.

## Problems and fixes

| # | Problem | Root cause | Fix | Where | Commit |
|---|---|---|---|---|---|
| 1 | The solve lost about 15 times the source amplitude on multi-rank runs | `SkyMap.mpi_reduce` chunked a wavelet map `(nfilt, nx, ny)` by `nx*ny`, so only band 0 was reduced when `nx*ny > 1e5` | Chunk by `self.map.size` | `minkasi/maps/skymap.py` | `d6904a0` |
| 2 | ACT weighted 100 times too strongly against M2 | `read_act` returned the ACT inverse variance per ACT pixel, `coarse2fine` repeated it on the 10 x 10 fine pixels and the ACT term summed over them | Divide by `ACT_OSAMP**2` | `examples/twores_ex.py` | `1abee2d` |
| 3 | Residual stuck at about 1e-3 once ACT is in the system; ACT-only floor 9.3e-4, all of it in band 3 | Every band's lattice was undersampled (see below) | `get_downsamps(tol)` takes the band limit from the filter array and applies Nyquist; `twores_ex --lattice-tol 1e-3` | `minkasi/needlet/needlet.py`, `examples/twores_ex.py` | `5009a1d` |
| 4 | Edge ringing, band overlaps counted twice | The truncated probed ANA is indefinite; band overlaps were counted twice (from the commit message) | See the commit: it also pads the map | `minkasi/needlet/needlet.py`, `examples/twores_ex.py` | `dcf2377` |
| 5 | ACT term wrong in several ways | Units, residual, beam, pixel window, band selection (as listed in the commit message) | See the commit | `examples/twores_ex.py`, `minkasi/needlet/needlet.py` | `9545eb7` |
| 6 | A stale response cache from another map size crashed with an unclear error | The cache does not record the settings it was made with | Clear error naming the band and shapes (the cache itself still records nothing) | `minkasi/needlet/needlet.py` | `77a44eb` |
| 7 | ACT 1.86x too faint and 3.5x overweighted against M2 | ACT converted to Jy/beam, but the M2 TODs are in K_RJ (WITCH: y -> K_RJ prefactor, beam kernel summing to 1) | `act_conversion(unit="K_RJ")`, default `--m2-unit K_RJ`; not the cause of the residual floor | `examples/twores_ex.py` | `6c6b240` |
| 8 | M2 data and model inconsistent at large scales | `fit_cm_plus_poly` subtracts a per-detector quadratic from the data (the common mode is fitted but not subtracted), which the noise model does not know about | Optional `--deproject-poly`: noise Q N^-1 Q with Q projecting out Legendre orders 0-2; changes M2 information by 1-4% in bands 0-2 (`transfer_overlap.py`) | `examples/twores_ex.py` | `9f6332a`, `f42a467` |

Added without a bug behind them: the CMB-aware ACT weight (`--act-cmb`, `minkasi/needlet/cmb.py`, `1abee2d`), the block
preconditioner shared over MPI ranks and the cache of its smallest eigenvalues (`b9a17fa`), CG on the full operator with
the block solve as preconditioner (`4f49262`), the hits mask in `WavSkyMap` (`79551db`), and the diagnostics
`--resdiag`, `--act-only`, `--ridge` (`5009a1d`) and `--flexible` (Polak-Ribiere CG, added after it).

## Problem 3 in detail: the lattice of each band

`WavSkyMap.get_downsamps` chooses the coarsest lattice that does not undersample each band. It took the number of
samples across the map from `needlet.get_need_lims(i, real_space=True)`, which converts the highest wavenumber of the band
to a length with `pi / (k * delta_k)`, where `delta_k = 2 pi / L` and `L` is the size of the needlet box. `twores_ex.py`
(and `make_wmap` in `tests/test_needlet.py`) build the needlet with `L = n * pixsize * sqrt(2)`, but `get_downsamps` compares the
length with the map side `n * pixsize`. The number of samples came out too small by a factor of `sqrt(2)`, so every lattice
was `sqrt(2)` too coarse.

Measured on the 480 x 480 map, 3 arcsec pixels (highest wavenumber of the filter, samples needed for Nyquist, samples used):

| Band | kmax (cycles per map) | Samples for Nyquist | Samples used with `L = n*pixsize*sqrt(2)` | Ratio |
|---|---|---|---|---|
| 1 | 3.6 | 7.2 | 5.3 | 1.37 |
| 2 | 7.1 | 14.1 | 10.2 | 1.39 |
| 3 | 13.6 | 27.2 | 19.6 | 1.39 |
| 4 | 26.7 | 53.4 | 37.8 | 1.41 |
| 5 | 51.7 | 103.3 | 73.0 | 1.42 |

With `L = n * pixsize` the same formula gives a ratio of 0.97 to 1.00, which is Nyquist. The filters themselves do not depend
on `L` (kmax is the same in both cases), so `L` only enters through `get_need_lims`.

Effect: the part of a band above the Nyquist wavenumber of its lattice aliases onto lower wavenumbers. The block
preconditioner works through the lattice, so it cannot see that part and CG cannot remove the residual in it. The
ACT-only solve showed this directly: its whole residual lay at 8 to 12 cycles per map, above the band-3 Nyquist of 10.
With the lattice from the filter support the ACT-only residual fell to about 6e-7.

Fix: `get_downsamps(tol)` reads the highest wavenumber of each filter from the filter array (window above `tol`) and takes
the spacing from Nyquist, independent of `L`. The default (`tol=None`) is unchanged, so existing caches are still valid;
`twores_ex --lattice-tol 1e-3` uses it and tags its caches with `_nyq1e-3`. The one line alternative is to build the
needlet with `L = n * pixsize`, which I have not tried in a full run.

Not recorded anywhere before: the comment in `get_need_lims` ("Unsure about this factor of pi") is correct if `L` is the map side.

Test: `tests/test_needlet.py::test_downsamps_from_filter_support_satisfy_nyquist`.

## Still open

- Whether the full M2 + ACT solve on the corrected lattice removes the residual floor (40-TOD run in progress).
- The response caches do not record how they were made.
