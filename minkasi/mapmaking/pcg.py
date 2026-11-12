import sys
import time
from typing import TYPE_CHECKING, Any, Callable, List, Optional, cast

import numpy as np

from ..parallel import myrank

if TYPE_CHECKING:
    from ..maps import Mapset, SkyMap
    from ..tods import TodVec

if sys.version_info >= (3, 8):
    from typing import Protocol, runtime_checkable
else:
    from typing_extensions import Protocol, runtime_checkable


@runtime_checkable
class HasPrior(Protocol):
    def apply_prior(self, x, Ax): ...


def pcg(
    apply: Callable[[Any], Any],
    b: Any,
    x0: Any,
    precon: Callable[[Any], Any] | Any | None = None,
    maxiter: int = 25,
    tol: float = 0.0,
    callback: Callable[[int, Any, dict], None] | None = None,
    flexible: bool = False,
    lanczos: dict | None = None,
) -> Any:
    """
    Preconditioned conjugate gradient for A x = b with a general operator, on any vector type with
    copy, axpy and dot (e.g. Mapset).

    Parameters
    ----------
    apply : Callable
        The operator A, returning a new vector.
    b : Mapset
        Right hand side.
    x0 : Mapset
        Starting guess.
    precon : Callable | Mapset | None, default: None
        Preconditioner, a symmetric positive definite approximation of A^-1: a callable returning a new
        vector, or a vector that multiplies the residual elementwise (e.g. 1 / hits). None for no
        preconditioning.
    maxiter : int, default: 25
        Maximum number of iterations.
    tol : float, default: 0
        Stop when |r| / |b| is below this.
    callback : Callable | None, default: None
        Called as callback(i, x, info) after every iteration i (from 0), info holding "relres"
        (|r| / |b|), "zr" (z . r) and "alpha".
    flexible : bool, default: False
        Use the Polak-Ribiere form of beta, which tolerates a preconditioner that is not exactly a
        fixed linear operator (e.g. one with inner iterations).
    lanczos : dict | None, default: None
        If given, filled with the CG coefficients "alpha" and "beta", "zr0" and the normalised
        preconditioned residuals "q" (the arrays of the first map), from which the Lanczos matrix of
        the preconditioned operator can be built (see minkasi.needlet.diagnostics.ritz_analysis).

    Returns
    -------
    x : Mapset
        The solution after the last iteration.
    """
    if precon is None:
        pre = lambda r: r.copy()
    elif callable(precon):
        pre = precon
    else:
        pre = lambda r: precon * r
    bnorm = np.sqrt(b.dot(b))
    x, r = x0.copy(), b.copy()
    r.axpy(apply(x), -1)
    z = pre(r)
    p, zr = z.copy(), r.dot(z)
    for i in range(maxiter):
        if lanczos is not None:
            lanczos.setdefault("q", []).append(z.maps[0].map / np.sqrt(zr))
            lanczos.setdefault("zr0", zr)
        ap = apply(p)
        alpha = zr / p.dot(ap)
        x.axpy(p, alpha)
        r_old = r.copy() if flexible else None
        r.axpy(ap, -alpha)
        z = pre(r)
        zr_new = r.dot(z)
        beta = (zr_new - z.dot(r_old)) / zr if flexible else zr_new / zr
        p_new = z.copy()
        p_new.axpy(p, beta)
        p, zr = p_new, zr_new
        if lanczos is not None:
            lanczos.setdefault("alpha", []).append(alpha)
            lanczos.setdefault("beta", []).append(beta)
        relres = np.sqrt(r.dot(r)) / bnorm
        if callback is not None:
            callback(i, x, {"relres": relres, "zr": zr, "alpha": alpha})
        if relres < tol:
            break
    return x


def run_pcg_wprior(
    b: "Mapset",
    x0: "Mapset",
    tods: "TodVec",
    prior: Optional[HasPrior] = None,
    precon: Optional["Mapset"] = None,
    mask: Optional["Mapset"] = None,
    maxiter: int = 25,
    outroot: str = "map",
    save_iters: List[int] = [-1],
    save_ind: int = 0,
    save_tail: str = ".fits",
    plot_iters: List[int] = [],
    plot_info: Optional[dict] = None,
    plot_ind: int = 0,
) -> "Mapset":
    """
    Function which runs preconditioned conjugate gradient on a bundle of tods to generate a map.
    PCG iteratively approximates the solution to the linear equation Ax = b,
    where A is a matrix and x and b are vectors.

    In the ML map making equation,  A = P'N"P and b = P'N"d.
    Where d the vector of TODs, N the noise matrix,  P the tod to map pointing matrix
    (i.e. a matrix that specifies which pixel in the map was observed by each TOD data point).
    In this case x is the map we are solving for.

    In the presence of a prior the least squares equation becomes:
    chi^2 = (d-Am)^T N^-1 (d-Ax) + (p-x)^T Q^-1 (p-x).
    Where p is the prior target for parameters, and Q is the variance.
    So now the equation we want to solve is:
    (A^T N-1 A + Q^-1)x = A^T N^-1 d + Q^-1 p.
    For non-zero p, it is assumed you have done this already and that:
    b=A^T N^-1 d + Q^-1 p.
    So to have a prior apply a Q^-1 x to Ax every time we call Ax.

    This is pcg with the operator tods.dot (plus the prior). Note that the prior is now also applied to
    the starting guess when the initial residual is formed (before, only tods.dot was), which only
    matters for a nonzero x0 with a prior.

    Parameters
    ----------
    b : Mapset
        The rhs of the equation. In our case this is P'N''d.
        The TOD class has a built in method for computing this.
    x0 : Mapset
        The initial guess. Generally set to zero for the first iteration
        and then to the output of the previous itteration for subsequent iterations.
    tods : TodVec
        The input TODs we want to make into maps.
        Note the noise should already been estimated and stored within each TOD object.
    prior : HasPrior | None, default: None
        Prior on the solution to the linear equation.
        If not None then needs to implement the apply_prior method.
    precon : Mapset | None, default: None
        The preconditioner matrix applied to A to ensure faster convergence.
        1/hitsmap is a frequent selection.
        Set to None to not use.
    mask : Mapset | None, default: None
        If mask is not none, apply the (map-space) mask to the tods.
    maxiter : int, default: 25
        Maximum number of iterations to perform.
    outroot : str, default: 'map'
        Root at which to save maps.
        Each saved iter will be saved at: outroot_{iter}{save_tail}
    save_iters : list, default: [-1]
        The iterations at which to save the result map.
        Default is to save only the last.
    save_ind : int, default: 0
        Index of Map in the MapSet x that should be saved.
    save_tail : str, default: '.fits'
        Extention for saving the output maps.
    plot_iters : list, default: []
        Which iterations to plot.
    plot_info : dict | None, default: None
        Plotting settings.
        See maps.SkyMap.plot for details.
    plot_ind : int, default: 0
        Index of Map in the MapSet x that should be plotted.

    Outputs:
        x: best guess for x after the conversion criteria has been reached (either max iter or
        Ax = b close enough to 0
    """

    def apply(x):
        ax = tods.dot(x, mask=mask)
        if prior is not None:
            prior.apply_prior(x, ax)
        return ax

    nsamp = tods.get_nsamp()
    clock = {"start": time.time(), "last": time.time()}

    def report(i, x, info):
        if myrank == 0:
            dt = time.time() - clock["last"]
            print(i, info["zr"], info["alpha"], dt, nsamp / dt / 1e6)
            sys.stdout.flush()
        clock["last"] = time.time()
        if i in save_iters and myrank == 0:
            x.maps[save_ind].write(outroot + "_" + str(i) + save_tail)
        if i in plot_iters and myrank == 0:
            to_plot = x.maps[plot_ind]
            if hasattr(to_plot, "plot"):
                to_plot = cast("SkyMap", to_plot)
                print("plotting on iteration ", i)
                to_plot.plot(plot_info)
            else:
                print("Warning: Can't plot as PolMap has no plot function implemented")

    x = pcg(apply, b, x0, precon, maxiter, callback=report)
    tave = (time.time() - clock["start"]) / maxiter
    print(
        "average time per iteration was ",
        tave,
        " with effective throughput ",
        nsamp / tave / 1e6,
        " Msamp/s",
    )
    return x


def run_pcg(
    b: "Mapset",
    x0: "Mapset",
    tods: "TodVec",
    precon: Optional["Mapset"] = None,
    mask: Optional["Mapset"] = None,
    maxiter: int = 25,
    outroot: str = "map",
    save_iters: List[int] = [-1],
    save_ind: int = 0,
    save_tail: str = ".fits",
    plot_iters: List[int] = [],
    plot_info: Optional[dict] = None,
    plot_ind: int = 0,
) -> "Mapset":
    """
    Wrapper function that just calls run_pcg_wprior with no prior.
    This exists for convenience/compatibility reasons,
    see the docstring of run_pcg_wprior for details on the params.
    """
    return run_pcg_wprior(
        b=b,
        x0=x0,
        tods=tods,
        prior=None,
        precon=precon,
        mask=mask,
        maxiter=maxiter,
        outroot=outroot,
        save_iters=save_iters,
        save_ind=save_ind,
        save_tail=save_tail,
        plot_iters=plot_iters,
        plot_info=plot_info,
        plot_ind=plot_ind,
    )


class null_precon:
    """
    Seems to be a dummy class to store a preconditioner
    that does nothing, no real reason to use this since you
    can just pass in precon=None if you don't want to precondition.
    """

    def __init__(self):
        self.isnull = True

    def __add__(self, val):
        return val

    def __mul__(self, val):
        return val
