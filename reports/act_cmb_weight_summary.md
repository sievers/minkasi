# CMB-aware ACT weight and an ACT normalization error

I've implemented the CMB-aware ACT weight. While doing it I found a normalization error in the existing ACT term. Two validation runs were in progress when this was written.

## The normalization error

- **Cause:** `read_act` gives the ACT inverse variance per ACT pixel, but `coarse2fine` repeats it on the 10x10 = 100 fine pixels inside each ACT pixel. The ACT term sums over fine pixels, so every ACT pixel was counted 100 times. ACT was weighted 100x too strongly against M2 in every run so far.
- **Fix:** the weights are now divided by `ACT_OSAMP**2`.
- **Cache:** the ACT cache name now carries a normalization tag, so the old matrices can't be reused by accident.
- **Earlier test:** my `--act-ivar-scale 0.01` run is therefore close to the corrected weighting. It showed the cluster decrement and the compact source changing by about 2%, and scales below about 12" by under 7%. The extended structure above about 1' changed by 30-44%, but that was mixed up with the convergence problem.
- **Earlier maps:** results already in the reports carry the 100x overweight and are not recomputed. I haven't changed the reports yet.

## The CMB weight (`--act-cmb S`, default 0 = white noise only, like before)

- **Weight:** each Fourier mode of the ACT noise precision is multiplied by N_l / (N_l + S * C_l * B_l^2).
  - `C_l` is the lensed CAMB spectrum with Planck 2018 parameters.
  - `B_l` is the ACT beam.
  - `N_l` is the white noise power from the median ACT inverse variance.
- **Operator:** it enters as diag(sqrt(ivar)) . W . diag(sqrt(ivar)), a symmetric operator. It goes into the operator, the rhs and the probed ACT response, and with S = 0 it reduces exactly to the corrected diagonal weight.
- **Code and tests:** it is in `minkasi/needlet/cmb.py`, which has the CAMB helper with a disk cache and a rank-0 broadcast, and the weight function. The three unit tests (`tests/test_cmb.py`) pass.
- **Assumptions:**
  - Stationary spectrum.
  - Spatial variation of `ivar` kept only through the sqrt(ivar) factors.
  - No foregrounds.
  - No ACT pixel window.

## Scale

Back-of-envelope, with a typical ACT f090 depth of 15-20 uK-arcmin, the CMB power at l ~ 1000 is several hundred times the ACT white noise. The weight there is about 1/500, which is stronger than my factor-100 test. I haven't checked it against the real `ivar` maps, and the median in the run will give the real number.

## Running at the time

On 40 TODs with the `cg40` cache, in sequence: corrected normalization with S = 0, then S = 1. Each takes about 10 min. Outputs go to `/mnt/welch/USERS/jorlo/needlets/cmb/c0` and `c1`. When they finish I'll compare the maps against the old run in the same bands, and look at the cluster decrement.

Nothing from this is committed yet.
