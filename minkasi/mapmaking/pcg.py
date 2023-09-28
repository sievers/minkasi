import sys
import time
from typing import TYPE_CHECKING, Optional, cast

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
    def apply_prior(self, x, Ax):
        ...


def run_pcg_wprior(
    b: "Mapset",
    x0: "Mapset",
    tods: "TodVec",
    prior: Optional[HasPrior] = None,
    precon: Optional["Mapset"] = None,
    maxiter: int = 25,
    outroot: str = "map",
    save_iters: list = [-1],
    save_ind: int = 0,
    save_tail: str = ".fits",
    plot_iters: list = [],
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
    t1 = time.time()
    Ax = tods.dot(x0)

    # compute the remainder r_0
    try:
        r = b.copy()
        r.axpy(Ax, -1)
    except:
        r = b - Ax
    if precon is not None:
        z = precon * r
        # key = tods.tods[0].info["fname"]
    else:
        z = r.copy()

    # Initial p_0 = z_0 = M*r_0
    p = z.copy()
    # k = 0.0

    # compute z*r, which is used for computing alpha
    zr = r.dot(z)
    # make a copy of our initial guess
    x = x0.copy()
    t2 = time.time()
    t3 = 0
    alpha = 0
    nsamp = tods.get_nsamp()
    tloop = time.time()
    for i in range(maxiter):
        if myrank == 0:
            if i > 0:
                print(i, zr, alpha, t2 - t1, t3 - t2, t3 - t1, nsamp / (t2 - t1) / 1e6)
            else:
                print(i, zr, t2 - t1)
            sys.stdout.flush()
        t1 = time.time()

        # Compute pAp
        Ap = tods.dot(p)
        if prior is not None:
            prior.apply_prior(p, Ap)
        t2 = time.time()
        pAp = p.dot(Ap)

        # Compute alpha_k
        alpha = zr / pAp

        # Update guess using alpha
        x_new = x.copy()
        x_new.axpy(p, alpha)

        # Write down next remainder r_k+1
        r_new = r.copy()
        r_new.axpy(Ap, -alpha)

        # Apply preconditioner
        if not (precon is None):
            z_new = precon * r_new
        else:
            z_new = r_new.copy()

        # compute new z_k+1
        zr_new = r_new.dot(z_new)

        # compute beta_k, which is used to compute p_k+1
        beta = zr_new / zr

        # compute new p_k+1
        p_new = z_new.copy()
        p_new.axpy(p, beta)

        # Update values
        p = p_new
        z = z_new
        r = r_new
        zr = zr_new
        x = x_new
        t3 = time.time()

        if i in save_iters and myrank == 0:
            x.maps[save_ind].write(outroot + "_" + repr(iter) + save_tail)
        if i in plot_iters and myrank == 0:
            to_plot = x.maps[plot_ind]
            if hasattr(to_plot, "plot"):
                to_plot = cast("SkyMap", to_plot)
                print("plotting on iteration ", i)
                to_plot.plot(plot_info)
            else:
                print("Warning: Can't plot as PolMap has no plot function implemented")

    tave = (time.time() - tloop) / maxiter
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
    maxiter: int = 25,
    outroot: str = "map",
    save_iters: list = [-1],
    save_ind: int = 0,
    save_tail: str = ".fits",
    plot_iters: list = [],
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
