# Handoff: needlet mapmaking work (WITCH / minkasi, branch `needlet`)

Written 2026-11-09. Read this first if you are picking the work up in a new session.

## Rules for working here (from `CLAUDE.md` and the user)

- Write reports as html with figures, and update the relevant report as the work progresses.
- Scripts need an argparse CLI and a `--debug` option. Run ruff (check and format) before running code. Type hints and full docstrings.
- Reduce duplication, write tight code. Use `~/venvs/general` (uv for packages).
- Commits end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
- Use they/them for anyone whose pronouns are not stated. Confirm before outward-facing or hard-to-reverse actions.
- Never send the user's email address to external services.

## Where things stand

The project combines MUSTANG-2 (M2) timestreams and an ACT map in a needlet (wavelet) basis. The main script is
`examples/twores_ex.py`, the library code is `minkasi/needlet/needlet.py` (`WavSkyMap`, `probe_band`, `block_solver`, `get_downsamps`)
and `minkasi/needlet/cmb.py`. Recommended solve: `--direct --cg N` (CG on the full M2 + ACT operator, block solve as preconditioner).

Code changes made since the baseline `3c435a3` are logged in `reports/code_changes.md` (table of problem, root cause, fix, commit).
Main fixes: `d6904a0` (MPI reduce chunking, a 15x amplitude loss), `1abee2d` (ACT 100x overweight, CMB-aware ACT weight `--act-cmb`),
`5009a1d` (lattice of each band from the filter support, `--lattice-tol`; also `--resdiag`, `--act-only`, `--ridge`),
`42e3b2c` (`--flexible` CG, residual curve plot, investigation pages).

### The lattice bug (most recent finding)

`get_downsamps` took the band limit from `get_need_lims`, which uses `delta_k` from the needlet box size `L`. `twores_ex.py` and the
test helper build the needlet with `L = n * pixsize * sqrt(2)`, but `get_downsamps` compares with the map side `n * pixsize`, so every band's
lattice was sqrt(2) too coarse and undersampled the top of the band. The block preconditioner works through the lattice, so it could not
see that part of each band and CG could not reduce the residual there. Fix: `get_downsamps(tol)` reads the highest wavenumber of each filter
from the filter array and applies Nyquist (`twores_ex.py --lattice-tol 1e-3`; caches get an `_nyq1e-3` tag). The default (`tol=None`) is unchanged.
Test: `tests/test_needlet.py::test_downsamps_from_filter_support_satisfy_nyquist`. An untried alternative: build the needlet with `L = n * pixsize`.

### What the investigation established

With M2 alone the CG converges (|r|/|b| about 1e-10 on 106 TODs, 1e-12 on 40 TODs). With ACT in the operator it stalls:
about 1e-3 (106 TODs) and 5e-4 (40 TODs) for hundreds of iterations, and the large scales keep moving.

- Ruled out: operator or preconditioner asymmetry, the ridge shift (1e-5 to 1e-1), the sharp ACT cutoff at band 3, nonlinear preconditioner (flexible CG), true against recursive residual.
- Found: the lattice undersampling above. ACT-only solve (`--act-only`) with the old lattice floors at 9.3e-4 entirely in band 3 at 8-12 cycles/map, above the band-3 lattice Nyquist of 10. With the corrected lattice it reaches 5.7e-7 (then drifts up after about 200 iterations, not investigated).
- Mode counting: the old lattice had fewer points than Fourier modes in bands 1-6 (band 3, 106-TOD map: 400 points for 548 modes). The corrected one has more points than modes in every band.
- 40 TODs, 600 iterations, old against corrected lattice: residual 3.8e-4 against 1.6e-4 at iteration 600 (old stays 3.8-6.6e-4 after iteration 200; corrected plateaus until about 250 then falls in steps).
  Map change between saved iterations at the largest scale (32-64 px): old 0.38, 0.42, 0.50 (not settling); corrected 0.39, 0.39, 0.16.
- Still unexplained: with ACT in the system the residual ends far above the M2-only solve (1.6e-4 against 1e-12), and the floor does not scale with the ACT weight
  (200 iterations, 40 TODs: weight 1: 4.7e-4; 1e-2: 2.7e-4; 1e-4: 2.1e-4; 1e-6: 3.5e-5; 0: 1.4e-12). Untested guess: M2 leaves weakly constrained modes (large scales removed by the common-mode subtraction), any ACT term turns them from exact nulls into tiny nonzero eigenvalues and puts signal into them, so CG needs many iterations.

### Other results worth knowing

- Cluster decrement, 106 TODs: white ACT noise -3.14e-4 Jy/beam, CMB counted as noise (`--act-cmb 1`) -3.75e-4 (19% deeper at every saved iteration; not a convergence effect).
- Bowling test (`examples/bowling_test.py`): the quadratic-surface amplitude of the large-scale disagreement with ACT is about 1.1e-4 Jy/beam without ACT (converged) and 2-5e-5 with ACT at iterations 40-200, growing with iteration. Caveats: a quadratic fit is a crude measure, and the ACT map contains the CMB.
- Unmasked solve (`--min-hits 0`, 106 TODs, CMB weight, 80 iterations): same compact structure and decrement (-3.68e-4) inside hits >= 20; edge pixels are noisy, no instability; pixels with no hits must be blanked in plots.

## Latest: edge modes and the outside-footprint prior (2026-11-09 evening)

The user decided we only care about convergence inside the M2 footprint (outside it, use the ACT map itself).
Small test problem (`--debug`, 1 TOD, 180x180, 6 bands, ACT in bands 0-1, corrected lattice), new `twores_ex.py` options:
`--ritz` (Ritz values/vectors of the preconditioned operator from the CG Lanczos data, and an exact split of the CG map into
eigenvector components, checked to 1e-12), `--restrict tol` (drop coefficients whose needlet does not reach the footprint),
`--outside-prior c` (prior eps*|map|^2 outside the footprint, eps = c times the M2 pixel weight), and `--resdiag` now prints
the real-space residual inside the footprint.

- The slowest modes (eigenvalues 1.7e-4 to 6e-4) have about 0% of their power inside the footprint but, because the solution
  weights them by 1/eigenvalue, carry 9% (lambda < 1e-3) to 13% (lambda < 1e-2) of the map inside it. They are combinations of
  coefficients near the edge (bands 1-2, about 7-9 cycles/map) that the masked data barely see.
- Dropping coefficients makes it worse: `--restrict 0.3` and `0.99` give smaller minimum eigenvalues (5.7e-5, 6.6e-6),
  residual 6.5e-4 and 2.6e-3 at 400 iterations against 1.2e-6, and in-footprint maps 3-4% off the reference. New slow edge modes appear.
- Running at handoff: `--outside-prior` 1e-4, 1e-3, 1e-2, 1e-1, 400 iterations each, in `/mnt/welch/USERS/jorlo/needlets/ritz/p*/`.
  A summary (residuals, spectrum, in-footprint map difference from the no-prior reference `ritz/ref/`, edge 10 px and interior)
  is written to `/mnt/welch/USERS/jorlo/needlets/ritz/prior_summary.txt` when they finish (script `ritz/summarize_prior.py`).
  What to look for: smallest eigenvalues raised, faster convergence, and a small in-footprint bias against the reference.
- If the prior works, next: test it on 40 TODs with the corrected lattice and judge convergence inside the footprint only.

## Open items

1. Explain why ACT in the system stalls the residual (see above). Ideas: dense analysis of the small `--debug` problem (180 x 180 map, 6 bands); deflation or a better preconditioner for the weakly constrained modes.
2. Run the 106-TOD solve on the corrected lattice (`--lattice-tol 1e-3`, new cache directory, M2 response must be re-probed; the old probing peaked at about 13 GB per rank, do not run two 16-rank probes at once).
3. Decide whether `--lattice-tol` (or `L = n * pixsize`) should become the default, and whether `--act-cmb 1` should be.
4. The earlier 40-TOD and first 106-TOD sections of `reports/needlet.html` still carry the old ACT weight (100x) and the old lattice. They have not been recomputed.
5. Speedups left: wavelet transforms once per apply instead of once per TOD per apply (rfft2 would be faster), probe assembly memory, uncompressed cache.
6. `reports/needlet.html` has not been updated with the lattice finding or the 600-iteration comparison (they are in the investigation artifact and `reports/code_changes.md`).

## Uncommitted in the working tree

`CLAUDE.md` and `reports/figs/summary.json` (user's own, leave alone), `reports/act_in_operator_vs_preconditioner.md`, `reports/figs/bowling_vs_act*.png`,
`reports/figs/map_nomask106_cmb.png`, `nika2_*` files, and this file. `reports/figs/bands.png` was blank in the working tree after a benchmark run and was restored with `git restore`.

## Artifacts (republished 2026-11-09 under the upenn account, private; the earlier copies belong to the previous account)

- Main page: https://claude.ai/artifact/83yLdSf6RiKEEh5Z6yDwkU, source `reports/needlet_mapmaking.html`, figures `bands.png`, `probe_band.gif`, `cg_convergence200_c1.png`, `final_cmb106.png` in `reports/figs/`.
- Investigation log: https://claude.ai/artifact/8XxEaS5g2cTvvs8tgzTGRt, source `reports/needlet_investigation.html`, figures `residual_curves.png`, `cg_convergence600_ref40.png`, `cg_convergence600_nyq40.png`.
- The user is signed in to the browser as a different account and cannot open these. To republish under another account: log in, publish the investigation first (`Artifact` publish with `file_path`, `icon: "log"`, and `files` mapping `figs/<name>.png` to the files above), then publish the main page with its link updated to the new investigation URL (icon `map`, `files` for its four figures), then update the back-link on the investigation page.
  The two pages link to each other by artifact URL. Source files are the truth; the published copies match them as of commit `42e3b2c`.

## Data and runs (all under `/mnt/welch/USERS/jorlo/needlets/`)

- `all106/cache`: 106-TOD M2 response (old lattice), ACT matrices, lmin JSON, CAMB spectrum. `all106_cmb_200/{c0,c1}`: 106 TODs, 200 iterations, white and CMB weight. `all106_noact_200`, `all106_cmb_nomask`, `all106_resdiag*`, `all106_actonly*`, `all106_noact_cut`: diagnostics.
- `cg40/cache`: 40-TOD cache (old lattice). `lat40/`: 40-TOD runs (`ref`, `ref600` old lattice; `nyq`, `nyq600` corrected lattice; `flex`, `noact`, `s1e-2`, `s1e-4`, `s1e-6`). `dense_dbg`, `dense_dbg_nyq`: small test problem.
- Response caches do not record settings: use a new cache directory when the TODs, map size, mask, radius or lattice change (the `--lattice-tol` runs tag their files).

## Practical notes

- Machine `psd`, 64 logical CPUs, 251 GB. Launch: `mpirun -n 16 --cpu-set 0-15 --bind-to core -x OMP_NUM_THREADS=1 -x OPENBLAS_NUM_THREADS=1 -x NUMBA_NUM_THREADS=1 python twores_ex.py --name MOOJ1142 --ntods 7 --direct --cg 200 --cg-tol 0 ...` (106 TODs: `--ntods 7`; 40 TODs: `--ntods 3 --ntods-total 40`). `--debug` gives a small one-TOD problem (run it without mpirun).
- The X display (`DISPLAY=localhost:11.0`) is dead: matplotlib hangs on a GUI backend. Use `MPLBACKEND=Agg` (the newer scripts force it). `ds9` hangs too. aplpy plots use `~/venvs/aplpy/bin/python` and `convention="calabretta"`.
- Do not use `pkill -f` or `pgrep -f` patterns that match their own command line (use the bracket trick `'twores_ex.py --nam[e]'`); kill your own jobs by PID.
- Run the tests with `~/venvs/general/bin/python -m pytest tests/test_needlet.py tests/test_cmb.py`.
- Useful scripts in `examples/`: `plot_residual_curves.py`, `cg_convergence.py`, `compare_maps_bands.py`, `compare_gif.py`, `plot_final_aplpy.py`, `plot_map_aplpy.py`, `bowling_test.py`, `probe_band_gif.py`.
