"""
Diagnostics of the needlet CG solve: operator symmetry, which coefficients reach the footprint, the Ritz spectrum of
the preconditioned operator, the residual per band, and a solve for the real-space map as a cross-check.
"""

import time
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh_tridiagonal
from scipy.ndimage import distance_transform_edt

from ..mapmaking.pcg import pcg
from ..maps.mapset import Mapset
from ..parallel import myrank
from .needlet import WavSkyMap, map2wav_real, wav2map_real


def footprint_of(mask: NDArray | float, shape: tuple[int, int]) -> NDArray[np.bool_]:
    """
    Boolean footprint from a hits mask that may be the scalar 1.0 (no mask).

    Parameters
    ----------
    mask : NDArray | float
        Mask in real space; nonzero is the footprint.
    shape : tuple[int, int]
        Map shape.

    Returns
    -------
    NDArray[np.bool_]
        The footprint.
    """
    return np.asarray(mask) + np.zeros(shape) > 0


def symmetry_check(
    m2_dot: Callable[[Mapset], Mapset],
    act_dot: Callable[[NDArray], NDArray] | None,
    like: Mapset,
    solve: Callable[[NDArray], NDArray] | None = None,
) -> None:
    """
    Print the relative asymmetry |<y, A x> - <A y, x>| / (|y| |A x|) of the M2 operator, the ACT term and their sum
    for random vectors, as a CG on a non symmetric operator stalls. With a preconditioner, also the asymmetry of that
    and the sign of <x, P x> for several vectors.

    Parameters
    ----------
    m2_dot : Callable[[Mapset], Mapset]
        The M2 operator, e.g. TodVec.dot.
    act_dot : Callable[[NDArray], NDArray] | None
        The ACT term acting on a wavelet array, or None without ACT.
    like : Mapset
        Mapset to take the shape from.
    solve : Callable[[NDArray], NDArray] | None, default: None
        The preconditioner acting on wavelet arrays.
    """
    rng = np.random.default_rng(0)  # the same vectors on every rank
    x, y = like.copy(), like.copy()
    x.maps[0].map[:] = rng.normal(size=x.maps[0].map.shape)
    y.maps[0].map[:] = rng.normal(size=y.maps[0].map.shape)
    ops = {"M2": lambda v: m2_dot(v).maps[0].map}
    if act_dot is not None:
        ops["ACT"] = lambda v: act_dot(v.maps[0].map)
        ops["M2+ACT"] = lambda v: ops["M2"](v) + ops["ACT"](v)
    if solve is not None:
        ops["precon"] = lambda v: solve(v.maps[0].map)
    for name, op in ops.items():
        ax, ay = op(x), op(y)
        a, b = np.sum(y.maps[0].map * ax), np.sum(ay * x.maps[0].map)
        err = abs(a - b) / (np.linalg.norm(y.maps[0].map) * np.linalg.norm(ax))
        if myrank == 0:
            print(f"{name}: <y,Ax> = {a:.6e}, <Ay,x> = {b:.6e}, asymmetry {err:.2e}")
    if solve is not None:
        xs = [x.maps[0].map, y.maps[0].map] + [
            rng.normal(size=x.maps[0].map.shape) for _ in range(3)
        ]
        # solve is collective over the MPI ranks, so every rank calls it
        vals = [f"{np.sum(v * solve(v)) / np.sum(v * v):.2e}" for v in xs]
        if myrank == 0:
            print("<x,Px> / |x|^2:", vals)


def coefficient_support(
    filt: NDArray, footprint: NDArray[np.bool_], tol: float
) -> NDArray[np.bool_]:
    """
    Wavelet coefficients whose needlet reaches the footprint: for each band, the radius beyond which its real space
    kernel stays below tol times its peak, and the footprint dilated by that radius.

    Parameters
    ----------
    filt : NDArray
        Needlet filters, shape (nfilt, nx, ny), centred in Fourier space.
    footprint : NDArray[np.bool_]
        Boolean map of the pixels the solution is wanted in.
    tol : float
        Relative kernel amplitude that defines its extent.

    Returns
    -------
    keep : NDArray[np.bool_]
        Boolean array of shape (nfilt, nx, ny).
    """
    nx, ny = footprint.shape
    dist = distance_transform_edt(~footprint)
    yy, xx = np.meshgrid(
        np.arange(nx) - nx // 2, np.arange(ny) - ny // 2, indexing="ij"
    )
    keep = np.empty(filt.shape, dtype=bool)
    for i, f in enumerate(filt):
        kern = np.abs(
            np.fft.fftshift(
                np.fft.irfft2(np.fft.fftshift(f)[:, : ny // 2 + 1], s=(nx, ny))
            )
        )
        radius = np.hypot(xx, yy)[kern > tol * kern.max()].max()
        keep[i] = dist <= radius
    return keep


def ritz_analysis(
    lanczos: dict,
    filt: NDArray,
    mask: NDArray | float,
    outroot: str,
    x: NDArray | None = None,
    nshow: int = 10,
) -> None:
    """
    Ritz values and vectors of the preconditioned operator P A from a CG run started at zero. The CG coefficients
    give the Lanczos tridiagonal matrix T = S diag(lambda) S^T of P^(1/2) A P^(1/2), and the stored normalised
    preconditioned residuals q_k (sign (-1)^k) give the eigenvectors of P A as u_j = sum_k S_kj q_k. The CG solution
    is then exactly x = sqrt(z_0 . r_0) sum_j S_0j / lambda_j u_j, so every Ritz pair gets its contribution to the map
    inside the footprint. Prints the spectrum, the smallest pairs and the pairs that contribute most inside the
    footprint, and saves {outroot}_ritz.npz.

    Parameters
    ----------
    lanczos : dict
        "alpha", "beta", "q" and "zr0" as filled by minkasi.mapmaking.pcg.pcg.
    filt : NDArray
        Needlet filters, shape (nfilt, nx, ny).
    mask : NDArray | float
        Hits mask in real space; nonzero is the footprint.
    outroot : str
        Output prefix.
    x : NDArray | None, default: None
        The CG solution (wavelet array), to check the decomposition.
    nshow : int, default: 10
        Number of pairs listed in each table.
    """
    a, b = np.array(lanczos["alpha"]), np.array(lanczos["beta"])
    n = len(a)
    diag = 1 / a
    diag[1:] += b[:-1] / a[:-1]
    lam, s = eigh_tridiagonal(diag, np.sqrt(b[:-1]) / a[:-1])
    q = np.array(lanczos["q"])[:n] * ((-1.0) ** np.arange(n))[:, None, None, None]
    u = np.tensordot(s.T, q, axes=1)  # eigenvectors of P A, wavelet arrays
    coef = np.sqrt(lanczos["zr0"]) * s[0] / lam
    nx, ny = filt.shape[1:]
    foot = footprint_of(mask, (nx, ny))
    kx = np.fft.fftshift(np.fft.fftfreq(nx)) * nx
    ky = np.fft.fftshift(np.fft.fftfreq(ny)) * ny
    kmag = np.hypot(*np.meshgrid(kx, ky, indexing="ij"))
    maps = np.array([wav2map_real(v, filt)[0] for v in u])
    infoot = (maps[:, foot] ** 2).sum(1) / np.maximum((maps**2).sum((1, 2)), 1e-300)
    p2 = np.abs(np.fft.fftshift(np.fft.fft2(maps), axes=(1, 2))) ** 2
    kbar = (p2 * kmag).sum((1, 2)) / p2.sum((1, 2))
    contrib = np.abs(coef) * np.sqrt(
        (maps[:, foot] ** 2).sum(1)
    )  # |component| in footprint
    xmap = np.tensordot(coef, maps, axes=1)
    print(f"Ritz values of P A: min {lam[0]:.3e} max {lam[-1]:.3e} ({n})")
    print(
        "number below 1e-1, 1e-2, 1e-3, 1e-4:",
        [int((lam < t).sum()) for t in (1e-1, 1e-2, 1e-3, 1e-4)],
    )
    if x is not None:
        ref = wav2map_real(x, filt)[0]
        err = np.linalg.norm((xmap - ref)[foot]) / np.linalg.norm(ref[foot])
        print(
            f"decomposition check: |sum - CG map| / |CG map| in footprint = {err:.1e}"
        )
    tot = np.linalg.norm(xmap[foot])
    order = np.argsort(contrib)[::-1]
    for title, idx in (
        ("smallest", range(min(nshow, n))),
        ("largest in footprint", order[:nshow]),
    ):
        print(
            f"Ritz pairs, {title}: lambda, |part in footprint| / |map in footprint|, power in footprint, mean k"
        )
        for j in idx:
            print(
                f"  {lam[j]:.3e}  {contrib[j] / tot:.3f}  {infoot[j]:.2f}  {kbar[j]:.1f}"
            )
    for t in (1e-3, 1e-2, 1e-1):
        sel = lam < t
        part = np.linalg.norm(np.tensordot(coef[sel], maps[sel], axes=1)[foot]) / tot
        print(
            f"modes with lambda < {t:g}: {sel.sum()}, their part of the map in footprint {part:.3f}"
        )
    if myrank == 0:
        np.savez(
            f"{outroot}_ritz.npz",
            lam=lam,
            coef=coef,
            contrib=contrib / tot,
            infoot=infoot,
            kbar=kbar,
            small=maps[: min(nshow, n)],
            big=maps[order[:nshow]],
        )


def residual_diagnostics(
    apply: Callable[[Mapset], Mapset],
    b: Mapset,
    x: Mapset,
    act_rhs: NDArray | float,
    act_bands: Sequence[int],
    outroot: str,
    filt: NDArray,
    mask: NDArray | float,
) -> None:
    """
    Print the true residual b - A x per band (norm relative to the norm of b in that band) and save it as a real
    space map, to see where the CG residual that does not go away lives.

    Parameters
    ----------
    apply : Callable[[Mapset], Mapset]
        The operator.
    b : Mapset
        Right hand side, M2 plus ACT.
    x : Mapset
        Solution.
    act_rhs : NDArray | float
        The ACT part of b (wavelet array), 0 without ACT.
    act_bands : Sequence[int]
        Bands in which ACT is used.
    outroot : str
        Prefix of the output map.
    filt : NDArray
        Needlet filters.
    mask : NDArray | float
        Hits mask.
    """
    r = b.copy()
    r.axpy(apply(x), -1)
    rb, bb = r.maps[0].map, b.maps[0].map
    ab = act_rhs + np.zeros_like(bb)
    nrm = lambda a: np.sqrt((a**2).sum(axis=tuple(range(1, a.ndim))))
    print(f"true |r|/|b| = {np.sqrt(r.dot(r) / b.dot(b)):.2e}")
    print(
        f"band  |r_i|/|b|   |b_i|/|b|   |act rhs_i|/|b_i|   (act bands: {list(act_bands)})"
    )
    bn = np.sqrt(b.dot(b))
    foot = footprint_of(mask, rb.shape[1:])
    rmap, bmap = wav2map_real(rb, filt)[0], wav2map_real(bb, filt)[0]
    print(
        "real space residual inside the footprint: |r| / |b| = "
        f"{np.linalg.norm(rmap[foot]) / np.linalg.norm(bmap[foot]):.2e}"
        f" (outside: {np.linalg.norm(rmap[~foot]) / np.linalg.norm(bmap[foot]):.2e})"
    )
    for i, (ri, bi, ai) in enumerate(zip(nrm(rb), nrm(bb), nrm(ab))):
        print(f"{i:4d}  {ri / bn:.2e}   {bi / bn:.2e}   {ai / max(bi, 1e-300):.2e}")
    if myrank == 0:
        np.save(f"{outroot}_resid.npy", rmap)
        np.save(f"{outroot}_resid_mask.npy", np.asarray(mask) + np.zeros(rb.shape[1:]))


def pixel_solve(
    kind: str,
    apply: Callable[[Mapset], Mapset],
    m2_dot: Callable[[Mapset], Mapset],
    act_dot: Callable[[NDArray], NDArray] | None,
    solve: Callable[[NDArray], NDArray],
    b: Mapset,
    wmap: WavSkyMap,
    hits: NDArray,
    ivar: NDArray | None,
    mask: NDArray | float,
    niter: int,
    tol: float,
    callback: Callable[[int, Mapset, dict], None] | None = None,
) -> NDArray:
    """
    CG for the real space map instead of the needlet coefficients. With W = wav2map_real and sum_i F_i^2 = 1, the map
    space operator is W apply W^T and the right hand side W b. The preconditioner is either the needlet block solve in
    map space, W solve W^T ("needlet"), or the inverse of an estimate of the operator diagonal ("jacobi"): M2
    proportional to the hits and ACT proportional to its inverse variance, each scaled by the operator applied to one
    pixel.

    Parameters
    ----------
    kind : str
        "needlet" or "jacobi".
    apply : Callable[[Mapset], Mapset]
        Operator on needlet coefficients.
    m2_dot : Callable[[Mapset], Mapset]
        M2 part of the operator on needlet coefficients.
    act_dot : Callable[[NDArray], NDArray] | None
        ACT part of the operator on wavelet arrays.
    solve : Callable[[NDArray], NDArray]
        Block solve on wavelet arrays.
    b : Mapset
        Right hand side in needlet coefficients.
    wmap : WavSkyMap
        The map.
    hits : NDArray
        Hits map.
    ivar : NDArray | None
        ACT inverse variance as used in the operator.
    mask : NDArray | float
        Hits mask.
    niter : int
        CG iterations.
    tol : float
        CG tolerance.
    callback : Callable | None, default: None
        Passed to pcg; x holds the real space map.

    Returns
    -------
    wsol : NDArray
        The solution as needlet coefficients, W^T m, so that wav2map_real gives back m.
    """
    filt = wmap.needlet.filters
    pix = Mapset()
    pix.add_map(wmap.real_map.copy())
    wav = b.copy()
    foot = footprint_of(mask, hits.shape)

    def lift(op, m, wavelet_op=False):  # W op W^T on a real map
        wav.maps[0].map[:] = map2wav_real(m, filt)[0]
        out = op(wav.maps[0].map) if wavelet_op else op(wav).maps[0].map
        return wav2map_real(out, filt)[0]

    def papply(m):
        y = m.copy()
        y.maps[0].map[:] = lift(apply, m.maps[0].map)
        return y

    if kind == "needlet":
        pre = lambda m: wav2map_real(solve(map2wav_real(m, filt)[0]), filt)[0]
    else:
        delta = np.zeros(hits.shape)
        p0 = np.unravel_index(np.argmax(hits), hits.shape)
        delta[p0] = 1
        diag = lift(m2_dot, delta)[p0] / hits[p0] * hits
        if act_dot is not None:
            diag = diag + lift(act_dot, delta, True)[p0] / ivar[p0] * ivar
        print(f"jacobi: diagonal from {diag.min():.3e} to {diag.max():.3e}")
        pre = lambda m: np.divide(m, diag, out=np.zeros_like(m), where=diag > 0)

    def pprecon(r):
        z = r.copy()
        z.maps[0].map[:] = pre(r.maps[0].map)
        return z

    rng = np.random.default_rng(
        1
    )  # symmetry and positivity of the operator and preconditioner
    u, v = rng.normal(size=(2, *hits.shape))
    au, av = lift(apply, u), lift(apply, v)
    pu, pv = pre(u), pre(v)
    print(
        f"pixel operator asymmetry {abs((u * av).sum() - (v * au).sum()) / abs((u * av).sum()):.1e}, "
        f"preconditioner asymmetry {abs((u * pv).sum() - (v * pu).sum()) / abs((u * pv).sum()):.1e}, "
        f"u.Au {(u * au).sum():.3e}, u.Pu {(u * pu).sum():.3e}, |P A u| / |u| "
        f"{np.linalg.norm(pre(au)) / np.linalg.norm(u):.3e}"
    )
    rhs = pix.copy()
    rhs.maps[0].map[:] = wav2map_real(b.maps[0].map, filt)[0]
    x0 = rhs.copy()
    x0.clear()
    t0 = time.time()
    x = pcg(papply, rhs, x0, pprecon, niter, tol, callback)
    r = rhs.copy()
    r.axpy(papply(x), -1)
    rm, bm = r.maps[0].map, rhs.maps[0].map
    print(
        "real space residual inside the footprint: |r| / |b| = "
        f"{np.linalg.norm(rm[foot]) / np.linalg.norm(bm[foot]):.2e}"
        f" (outside: {np.linalg.norm(rm[~foot]) / np.linalg.norm(bm[foot]):.2e}), {time.time() - t0:.0f} s"
    )
    return map2wav_real(x.maps[0].map, filt)[0]
