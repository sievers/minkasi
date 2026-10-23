# Chat log: why the direct needlet map differs from the PCG maps

Dumped from the session. Starts at the diagnosis message, includes the follow-up question and the detailed explanation.

---

## Message 1: diagnosis

The two maps do differ by more than a scale factor. The needlet map's fine-scale amplitude is set by the solver's regularization, not by the data.

**What differs.** I compared the maps in the common pixels (hits >= 20).

| Quantity | Needlet | `minkasi_mask_example` | `minkasi_mpi_example` |
|---|---|---|---|
| Pixel rms (Jy/beam) | 5.0e-5 | 2.0e-4 | 7.3e-4 |
| Pixel-to-pixel noise | 3.4e-5 | 1.4e-4 | 1.5e-4 |

- **Brightest compact source:** the needlet peak is about 15x lower than in the mask map.
- **Power at 0.5-2':** the needlet map has 40-50x less power there, which is 6-7x in amplitude.
- **Fine scales:** the gap shrinks to about 1.5x in amplitude at 6".
- **ACT scales:** at the scales ACT constrains (bands 0-3), the needlet map agrees with ACT, with slope 1.09 and correlation 0.97 after a 2'-8' band-pass. This isn't an independent check, since ACT is part of the solve there.

**Cause.**
- **Indefinite probed matrices:** the probed response matrices of the fine bands (5-9) are not positive definite. Their smallest eigenvalue is -0.2 to -1.1 times the mean diagonal measured over the footprint pixels only. The eigenvectors sit entirely inside the footprint, not at the edges.
- **Shift in `solve_bands`:** `solve_bands` fixes this by adding a diagonal shift of that size, on top of the 1e-3 ridge. The shift multiplies the solution by d/(d+shift), where d is the local diagonal.
- **Resulting damping:** the median gain is 0.63-0.89 in bands 5-8 and 0.28 in band 9, the finest band. In shallow regions it falls to 0.03-0.4.
- **Why sources suffer most:** compact sources live in the finest band and shallow regions are damped most, which matches the 15x source suppression.

**Why exact matrices don't fix it.** I ran a control with 4 TODs, 12" pixels, exact ANA and ACT. Its pixel rms is 17x higher than the PCG map's and it is uncorrelated with it.
- **Why the exact solve blows up:** the block-diagonal solve treats each band on its own, so a band's rhs has components in modes that its block can't constrain, for example modes removed by the common-mode subtraction. The ridge amplifies those by up to 1/ridge. PCG never excites them.

So the approximation needs either the regularization or the band coupling changed. Options:
- Couple neighbouring bands, i.e. include the off-diagonal ANA blocks.
- Use a local shift instead of a global one.
- Make the probed matrices positive definite by construction.
- Use the probed ANA as a preconditioner for CG on the full system, which also matches the `ANA^-1 @ rhs` goal more closely.

No code was changed for this. Findings were added to `reports/needlet.html`. Nothing is committed.

---

## User

> Ah ok the regularization is a bit deal. Can you explain that a bit more to me in detail?

---

## Message 2: the regularization in detail

### What the solver computes

For each band i, `solve_bands` does the following:

1. It takes the band-filtered rhs, `b_i = F_i r`, sampled on that band's lattice.
2. It builds `G_i`, the probed (M2 + ACT) response matrix for that band, i.e. `A^T N^-1 A` restricted to band i.
3. It solves `(G_i + shift*I) c_i = b_i`.
4. The map is `sum_i F_i^2 (...) c_i`, with the bands summed back into real space.

An unregularized solve would be `c = G^-1 b`. Adding `shift*I` turns it into the ridge solution `c = (G + shift*I)^-1 b`.

### What a ridge does

Take any mode of `G` with eigenvalue lambda. The ridge multiplies the data in that mode by `lambda / (lambda + shift)` relative to the true inverse:
- **Well-constrained modes** (lambda >> shift): gain is about 1, so the solve is essentially unbiased.
- **Poorly constrained modes** (lambda << shift): gain is about 0, so the solve pulls them toward zero. This is a zero-mean Gaussian prior on those modes, as the docstring says.

The ridge trades bias for variance. It stops noise being divided by tiny eigenvalues, but it biases every mode with lambda comparable to or below the shift. Here the shift is large, so a lot of real signal sits in that regime.

### Two sources of shift, and the one that matters

The shift is `1e-3 * mean(diag G)` plus `max(0, -1.1 * lambda_min)`.

- **The ridge term** is 0.1% of the mean diagonal. That would be harmless.
- **The indefiniteness term** is the problem. The probed `G` for bands 5-9 has negative eigenvalues. The truncated and probed approximation of `A^T N^-1 A` is no longer positive semi-definite, even though the true one is. The code then shifts everything up by the most negative eigenvalue so the solve is stable.

That shift is one global number, set by a handful of bad modes. It is applied to all lattice points, including the well-measured ones. I measured it as 0.2-1.1 times the typical in-footprint diagonal. The gain `d/(d+shift)` at a lattice point with local diagonal `d` is:
- **Bands 5-8:** about 0.6-0.9 at the median, and as low as 0.1-0.4 at the 10th percentile.
- **Band 9 (finest):** 0.28 at the median and 0.03 at the 10th percentile.

### Why the damping is uneven

`d` scales roughly with local depth, meaning how many hits land on that part of the map. So the suppression depends on depth:
- **Shallow regions** have small `d`, so they are damped hard.
- **Deep regions** are damped only mildly.
- **Compact sources** live in the finest band, where the gain is lowest.

This is why the needlet map isn't just a rescaled PCG map. Large-scale bands are fine because ACT makes `G` well conditioned there and the shift is only the 1e-3 ridge. The loss of fine-scale amplitude and the 15x source suppression come from the bands where `G` is indefinite.

### Why not just remove the shift

I tried the exact response matrices in the 4-TOD control. They are positive semi-definite, so no indefiniteness shift is needed, and only the 1e-3 ridge applies. The result was much noisier than PCG, with 17x the pixel rms and no correlation with the PCG map. So the shift is hiding a second problem.

My explanation is the following. I haven't proven it:
- The true system couples all bands, but the solver only inverts the diagonal blocks.
- The full rhs `r` lives in the range of the full `A^T N^-1 A`. After projecting onto one band, `F_i r` has components in directions that `G_i` can't constrain. Common-mode subtraction removes some modes from M2 entirely, for example.
- With only the 1e-3 ridge, those directions get divided by a tiny number, so noise explodes.
- PCG avoids this because Krylov iterations only build the solution from the range of the operator.

### The dilemma

- **Small ridge:** noise blows up in unconstrained directions.
- **Large ridge, which is effectively what we have:** the map is stable, but the fine-scale amplitude is set by the regularization and varies with depth.
- **In both cases** the block-diagonal solve doesn't recover the true `ANA^-1 rhs`.

The fix has to make the system self-consistent, not retune the number. That means the cross-band blocks, or CG on the full system with the block solve as a preconditioner. In CG, null directions are never excited, and the preconditioner only needs to be approximately right.

### What I have and haven't verified

**Measured:**
- The eigenvalues of the probed matrices and their shifts.
- The per-band gains.
- The exact-matrix control being noisy.
- The amplitude differences between the maps.

**Inferred, not tested directly:**
- That the block-diagonal inconsistency (rather than something else in the control run) causes the noise blow-up.
- That the shift accounts for all of the fine-scale suppression. It clearly accounts for some, but I haven't re-solved with a different regularization to see how much of the 15x it explains.

The cheapest test of that second point is to re-run the production solve with a local, Jacobi-scaled shift, where the shift is proportional to the local diagonal. If the point-source peak recovers substantially, the shift is the main cause. The alternative is to go straight to the CG approach.

---

## Reference: where the numbers came from

- Scratch scripts: `/tmp/claude-1001/-home-jorlo-dev-minkasi/13f39bd6-5e81-40c5-9f34-f45692849e2d/scratchpad/` (`scale.py`, `ctrl.py`, `shift.py`, `eig.py`, `gain.py`).
- Production caches (probed response matrices): `/mnt/welch/USERS/jorlo/needlets/direct_all/cache/`.
- Control runs (4 TODs, 12" pixels, exact ANA; `d/` without ACT, `da/` with ACT): `/tmp/claude-1001/-home-jorlo-dev-minkasi/13f39bd6-5e81-40c5-9f34-f45692849e2d/scratchpad/ctrl/`.
- Code: `WavSkyMap.solve_bands` in `minkasi/needlet/needlet.py`, `min_eigenvalue` in the same file.
- Report section: "Why the direct needlet map has a different amplitude from the PCG maps" in `reports/needlet.html`.

---

## Follow-up: results of the two tests (`examples/ridge_test.py`)

Setup: 4 TODs, 12" pixels, 8 filters, padded map (84x84), M2 only, exact response matrices, noise model cleaned with 50 PCG steps (as `--direct`).
Reference = CG on the full operator, block solve with the masked exact G (ridge 1e-3) as preconditioner, 40 iterations (residual 7e-6).

| map | rms | pixel noise | band-pass corr | band-pass slope |
|---|---|---|---|---|
| reference | 3.34e-4 | 2.75e-4 | 1.000 | 1.000 |
| PCG 50 | 2.93e-4 | 2.47e-4 | 0.994 | 0.846 |
| PCG 100 | 3.33e-4 | 2.75e-4 | 1.000 | 0.998 |
| block solve, ridge 0.1 | 3.6e-4 | 2.6e-4 | 0.88 | 1.08 |
| block solve, ridge 1e-3 | 2.2e-3 | 1.2e-3 | 0.79 | 7.2 |
| block solve, ridge 1e-6 | 5.5e-1 | 5.4e-1 | 0.60 | 939 |

(masked-G block solves; unmasked-G ones are similar.)

- Block-preconditioned CG on the full operator converges and reproduces PCG: the exact ANA is fine, the block-diagonal *solve* is the problem.
- The block solve's noise scales as 1/ridge, as predicted by the unconstrained-mode mechanism. No ridge makes it match (best correlation 0.88).
- The mask mismatch of `get_response_matrix` (probes the unmasked real-space map) is real (1.6e-3 vs 2.9e-15) but secondary.
- A plain 300-step PCG of the same problem diverged; not investigated.

---

## Update: the amplitude difference was mostly an MPI reduction bug

`SkyMap.mpi_reduce` chunked a (nfilt, nx, ny) wavelet map by nx*ny, so for nx*ny > 1e5 (the 480x480 production maps) only band 0 was summed over ranks.
The fine bands of the rhs and A x stayed rank-local, so 16-rank runs solved with ~1/16 of the data in the fine bands against a response from all TODs (about 1/15 amplitude).
Fixed (chunks by map.size) with a regression test. After the fix, CG on the full operator for 40 TODs gives the same compact source amplitude as the PCG examples
(3 px mean around the brightest source: 1.5e-3 needlet, 1.4e-3 mask PCG, 1.2e-3 plain PCG), and 13% more power than the unconverged 50 step PCG at 0.3-1.8'.
The block-solve conclusions above (ridge noise amplification, shift damping) still hold for the block solve but are not the explanation of the 15x.
