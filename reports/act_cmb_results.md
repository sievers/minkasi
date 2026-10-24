# CMB weight on the ACT term: 40-TOD results

The CMB weight changes the 40-TOD map mostly at large scales, and the central decrement gets about 8% deeper. Both runs finished, and the code, the test, the report update and the summary are committed.

I compared the CMB run (S = 1) against the corrected white-noise run (S = 0). The table gives the rms of the difference over the rms of the white-noise map, per Gaussian band-pass sigma in pixels (3" each).

| Band sigma (px) | 0.01-1 | 1-2 | 2-4 | 4-8 | 8-16 | 16-32 | 32-64 |
|---|---|---|---|---|---|---|---|
| rms(CMB - white) / rms(white) | 0.00 | 0.00 | 0.00 | 0.02 | 0.06 | 0.15 | 0.40 |

| Run | Central decrement, 4-px smoothed (Jy/beam) |
|---|---|
| Old (100x ACT weight) | -3.23e-4 |
| Corrected, white noise | -3.31e-4 |
| CMB as noise | -3.59e-4 |

- **Small scales:** below about 24" (sigma up to 8 px) the map changes by 2% or less, so compact structure is unaffected.
- **Large scales:** the 40% change at the largest scales is confounded with convergence. The CG residual after 40 iterations is 2.0e-3, and those scales are the ones still converging.
- **Decrement:** the cluster decrement is 8% deeper with the CMB counted as noise. This fits the white model letting a CMB gradient bias it slightly, but I haven't shown that. The change is small against how unconverged the large scales are.

## How strong the weight is

I computed it from the real ACT inverse-variance maps.

- **Noise level:** the median total ACT inverse variance is 2.1e-3 uK^-2 per 0.5' pixel, which is 22 uK.
- **Weights:** they are 1.7e-4 at l = 500, 1.6e-3 at l = 1000, 3.5e-2 at l = 2000, 0.49 at l = 3000 and 1.0 at l = 5000.
- **What that means:** the CMB makes ACT nearly uninformative on the scales of the footprint. My earlier estimate of about 1/500 at l = 1000 was close. The field centre I used for that cut was approximate.

## What I committed

- **`1abee2d`:** the `--act-cmb` option, `minkasi/needlet/cmb.py`, the ACT weight normalization fix and the new tests.
- **`9845378`:** the `needlet.html` section and the summary file.
- **Left uncommitted, as before:** `CLAUDE.md`, `bands.png`, `summary.json` and the `nika2_*` files.

## Still open

- **Defaults:** `--act-cmb` stays at 0 by default, so existing commands now use the corrected normalization but still the white noise model. You'd decide whether the CMB weight should be the default.
- **Full sample:** I haven't re-run the 106 TODs with the corrected weight. The full-sample numbers in the report carry the 100x overweight.
