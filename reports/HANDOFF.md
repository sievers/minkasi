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
- Result (full table in `ritz/prior_summary.txt`): the prior makes the in-footprint map converge fast (at c = 1e-2 and 1e-1 it stops
  changing by iteration 50) but moves it away from the no-prior solution by 3-5% (c = 1e-4, not converged) to 10-14% (c >= 1e-3),
  including 8-11% more than 10 px from the edge. So on this toy problem the in-footprint map depends on what is assumed outside at
  the 10% level. A zero-mean prior outside is not physical (the sky there is not zero). Proposed next test, not yet run: let ACT
  constrain the outside (ACT inverse variance not masked to the M2 hits), i.e. a prior whose mean is the ACT map.
- Correction: ACT's inverse variance is NOT masked to the M2 hits (`coarse2fine` only upsamples; ivar covers the whole map). The outside
  is free only in the bands without ACT. Follow-up tests (small problem, 400 iterations, `ritz/`): prior only on non-ACT bands (`pna`)
  = same as the prior on all bands; ACT also in band 2 (`t001`) converges slower. New `--pixel-cg {jacobi,needlet}` solves for the
  real-space map: residual at 400 iterations 1.5e-3 (Jacobi, stalls) and 7.1e-4 (needlet preconditioner), against 1.2e-6 for the
  needlet-coefficient solve. All solutions except the coefficient reference agree within 3-9% in the footprint interior; each
  differs from the coefficient reference by 11-12%. So the in-footprint map has a component of about 11% that the data barely
  determine, and the coefficient (overcomplete) formulation fills it differently from all the others. Next proposed: a simulation
  with a known input sky to decide which is closer to the truth.

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

## 2026-11-10: units, simulation, transfer functions, deprojection

Summary page (current status, read first): https://claude.ai/artifact/G5BMVxggqE7KRSRYHUsFxj, source `reports/convergence_summary.html`,
figures in `reports/figs/summary/` (made by `examples/summary_figs.py`, `plot_residual_curves.py`, `transfer_overlap.py`).

- Units (`6c6b240`): M2 TODs are K_RJ; ACT was converted to Jy/beam (4.37e-7 instead of 8.15e-7 per uK_CMB). Not the cause of the
  residual floor (krj600 and nyq600 residual curves are the same), but large scales settle faster (32-64 px change 0.23/0.13/0.09 vs 0.41/0.40/0.16).
- Simulation (`examples/make_sim_sky.py` uses WITCH; `twores_ex --sim-sky --sim-noise flip`), runs in `sim/`, table in `sim/summary.txt`.
  Joint solves get the decrement to 1% (truth -6.49e-4). Interior 32-64 px error: coef 0.14 at iteration 40 -> 0.32 at 600 (drifts, at the edge),
  pixn the same, pixj slower, prior (c=1e-2, non-ACT bands) 0.12 and stable = best. m2only: transfer 0.27 (the bowl).
- Transfer functions (`examples/transfer_overlap.py`, `overlap/tf40*`): M2 information per mode (Fourier diagonal of A^T N^-1 A with the
  SVD noise model) against the ACT term. M2 share: band 0 0.3%, band 1 3.8%, band 2 37%, band 3 92%; crossover at l ~ 5000 (2.5' half wavelength).
  The SVD noise model already down-weights large scales. The common mode is NOT removed from the data: fit_cm_plus_poly subtracts only the
  per-detector quadratic, whose data transfer is 0.55/0.72/0.93 in bands 0/1/2.
- `--deproject-poly` (`9f6332a`, fixed in `f42a467`): noise Q N^-1 Q, with Q projecting out Legendre orders 0-2 per detector, so the model matches the
  filtered data. The first version also projected out the common mode, which was wrong because it is still in the data; removed. Cache tag `_dpoly`.
  M2 information changes by 1/2/4% in bands 0/1/2 (polynomial only), so expect a small effect.
- Q was removed in `72ae36f` (no effect; the noise model already does the down-weighting). The coefdp/m2onlydp runs died with a session disconnect and are not needed.
- The residual on the plots is |b - H x_k| / |b| over all needlet coefficients (whole padded map, not preconditioned); small-eigenvalue directions barely show in it, so judge convergence against the simulated truth.

## 2026-11-10 afternoon: the mask, display maps, the real-data tilt

- **The hits mask caused the large-scale edge drift.** With `--min-hits 20` the M2 model is zero at low-hit pixels while their samples stay
  in the data; the SVD noise model carries that sky into the footprint edge. 40-TOD simulation, interior largest-scale error at iteration 600:
  masked 0.32, unmasked 0.10 (transfer 0.98), unmasked + outside prior 0.09. **Baseline decision (user): no mask, no prior**, M2 and ACT
  combined only through the noise models (`--direct --cg 600 --lattice-tol 1e-3 --min-hits 0`). The prior stays as a diagnostic.
- Prior options now: `--outside-prior c`, `--prior-bands`/`--prior-non-act`, `--prior-min-hits`, `--prior-mask FILE`, `--prior-mean FILE`
  (+ `--prior-mean-scale`). Checked on the debug problem (mean reproduced to 0.03 uK; mask file reproduces the hits region exactly).
  Fixed in the same commit (`3d020e9`): `sample_sky` lost the last row/column to round-off when resampling an image on the same grid.
- Display maps (for viewing only): `examples/multiscale_map.py` (real data) and `examples/sim_multiscale.py` (truth / data / residual).
  ACT 30" pixels split 8x8 where M2 has data; settings used on the pages: `--smooth 10 --smooth-ref-hits 3000 --show-hits 120
  --feather-hits 300 --outline` (sim figures also `--vmax 150`). Default field 18'.
- **Open: the real joint map tilts against ACT** by ~81 uK across the M2 region (2.57x ACT noise; the simulation gives 0.97x).
  Not the mask, not convergence (same at iteration 40), not ACT (pa5 and pa6 agree, split null white). So it is a sky-coherent M2
  systematic or an M2/ACT calibration/transfer difference. Tool: `examples/act_consistency.py --root <outroot>`.
- Running, detached (setsid), in order:
  1. `all106_nomask/run.sh`: 106 TODs, no mask, 600 iterations -> `all106_nomask/run600/`, then a multiscale map
     `reports/figs/summary/best_real106_multiscale.png` (made with the older display settings; rerender with the ones above).
  2. `jack40/run.sh`: jackknife of the 40 TODs (`--tod-list`, lists from `examples/tod_splits.py`): time1/time2 (2018 vs 2020+2023) and
     elev_lo/elev_hi, 200 iterations each; summary with the tilt per half in `jack40/summary.txt`.
- Pages: status https://claude.ai/artifact/G5BMVxggqE7KRSRYHUsFxj (source `reports/convergence_summary.html`, section 8 is current),
  mapmaking https://claude.ai/artifact/83yLdSf6RiKEEh5Z6yDwkU (source `reports/needlet_mapmaking.html`, "Current best status").
  When the 106-TOD map is in, swap it into Figure 10 / the mapmaking page, and add the jackknife result to section 8.

## 2026-11-10 evening: ACT was mirrored in RA in every real-data solve (fixed, 809620f)

- The real-data tilt (~80 uK, same in every jackknife half) was a bug: `read_act` copied ACT pixels onto the fine grid by array index.
  pixell returns the cutout with RA increasing along the array; the fine grid has RA decreasing. Evidence: joint map follows ACT in Dec,
  opposite gradient in RA; the solve's ACT map correlates +0.63 with ACT mirrored in RA, -0.35 as is. Fixed: each 30" block takes the
  ACT pixel at its centre's sky position (correlation with independently read ACT 0.999, ivar ratio 1.0000). ACT response caches now tagged `_wcs`.
- All real-data results with ACT before the fix are superseded (krj600, nomask600, all106_nomask/run600, jack40, the 106-TOD numbers in
  reports/needlet.html and the mapmaking page's older table). Simulations are unaffected (sim ACT built on the fine grid from the sky).
- Running (detached, `run_wcs.sh`): `lat40/nomask600_wcs` then `all106_nomask/run600_wcs`, each followed by `act_consistency.py`
  (`consistency.txt`) and a multiscale map (written to reports/figs/summary/best_real_multiscale.png and best_real106_multiscale.png;
  rerender without the --title argument, which got underscores). Expect the consistency rms near 1 if the fix explains the tilt.
- Result of the reruns: the fix explains the tilt. Joint vs ACT at ACT resolution, rms / ACT noise: 40 TODs 0.96 (tilt 4 uK; was 2.57, 82 uK),
  106 TODs 1.15 (tilt 11 uK, bowl +26 uK; was 3.29, 104 uK); simulation 0.97. Display maps rerendered; the 106-TOD map is Figure 10 /
  the mapmaking page's real map. Possible follow-up: the +26 uK bowl in the 106-TOD map (leftover M2 bowling?).

## 2026-11-10 late: twores_ex.py refactored into minkasi (bit-identical)

- `examples/twores_ex.py` is now 19 lines: `JointMapper(parse_args(__doc__)).run()` from `minkasi/needlet/joint.py`. Same options.
- Moved: `pcg` (general CG; `run_pcg_wprior` now wraps it) -> `minkasi/mapmaking/pcg.py`; `read_tods` -> `minkasi/tods/io.py`;
  `sample_sky` -> `minkasi.maps.utils.sample_image`; `setup_wmap` -> `minkasi.needlet.needlet.wavskymap_for_tods`, `cached` ->
  `cached_responses`; ACT (`act_conversion`, `read_act`, `sim_act`, `edge_taper`, `ActTerm`) -> `minkasi/needlet/act.py`; `inject_sim` ->
  `minkasi/needlet/sim.py`; diagnostics (`symmetry_check`, `coefficient_support`, `ritz_analysis`, `residual_diagnostics`, `pixel_solve`)
  -> `minkasi/needlet/diagnostics.py`. pcg callbacks now get `(i, x, info)` with `info["relres"]`.
- Verified: old vs new on 4 debug configurations (baseline CG; mask+prior+mean+restrict+ritz+resdiag; sim+pixel Jacobi; block solve):
  all outputs identical (max difference 0). tests/test_needlet.py, test_cmb.py pass. Other scripts now import from minkasi.
- Next (user): the same for the other analysis scripts (multiscale_map, sim_multiscale, act_consistency, sim_compare, band_error,
  transfer_overlap, make_sim_sky, tod_splits, summary_figs).
- Queued after the M2-only 106-TOD run (all106_nomask/m2only300, for the bowling): sim pixj/pixn unmasked (`sim/run_pix_nomask.sh`,
  summary `sim/summary_pix_nomask.txt`), using the refactored script.
