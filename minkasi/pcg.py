import time
from .maps import Mapset, SkyMap, PolMap
from .tods import TodVec
from .parallel import myrank


def run_pcg(
    b: Mapset,
    x0: Mapset,
    tods: TodVec,
    precon: Mapset | None = None,
    maxiter: int = 25,
    outroot: str = "map",
    save_iters: list = [-1],
    save_ind: int = 0,
    save_tail: str = ".fits",
    plot_iters: list = [],
    plot_info: dict | None = None,
    plot_ind: int = 0,
) -> Mapset:
    """
    Function which runs preconditioned conjugate gradient on a bundle of tods to generate a map.
    PCG iteratively approximates the solution to the linear equation Ax = b,
    where A is a matrix and x and b are vectors.

    In the ML map making equation,  A = P'N"P and b = P'N"d.
    Where d the vector of TODs, N the noise matrix,  P the tod to map pointing matrix
    (i.e. a matrix that specifies which pixel in the map was observed by each TOD data point).
    In this case x is the map we are solving for.

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
    precon : Mapset | None, default: None
        The preconditioner matrix applied to A to ensure faster convergence.
        1/hitsmap is a frequent selection.
        Set to None to not use.
    maxiter : int, default: 25
        Maximum number of iterations to perform.
    outroot : str, default: 'map'
        Root at which to save maps.
        Each saved iter will be saved at: outroot_{iter}{save_tail}
    save_iters : list, deault: [-1]
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
        t1 = time.time()

        # Compute pAp
        Ap = tods.dot(p)
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
            if isinstance(to_plot, SkyMap):
                print("plotting on iteration ", i)
                to_plot.plot(plot_info)
            elif isinstance(to_plot, PolMap):
                print("Warning: Can't plot as SkyMap has no plot function implemented")

    tave = (time.time() - tloop) / maxiter
    print(
        "average time per iteration was ",
        tave,
        " with effective throughput ",
        nsamp / tave / 1e6,
        " Msamp/s",
    )

    return x


def run_pcg_wprior(
    b,
    x0,
    tods,
    prior=None,
    precon=None,
    maxiter=25,
    outroot="map",
    save_iters=[-1],
    save_ind=0,
    save_tail=".fits",
):
    # least squares equations in the presence of a prior - chi^2 = (d-Am)^T N^-1 (d-Am) + (p-m)^T Q^-1 (p-m)
    # where p is the prior target for parameters, and Q is the variance.  The ensuing equations are
    # (A^T N-1 A + Q^-1)m = A^T N^-1 d + Q^-1 p.  For non-zero p, it is assumed you have done this already and that
    # b=A^T N^-1 d + Q^-1 p
    # to have a prior then, whenever we call Ax, just a Q^-1 x to Ax.
    t1 = time.time()
    Ax = tods.dot(x0)
    if not (prior is None):
        # print('applying prior')
        prior.apply_prior(x0, Ax)
    try:
        r = b.copy()
        r.axpy(Ax, -1)
    except:
        r = b - Ax
    if not (precon is None):
        z = precon * r
    else:
        z = r.copy()
    p = z.copy()
    k = 0.0

    zr = r.dot(z)
    x = x0.copy()
    t2 = time.time()
    for iter in range(maxiter):
        if myrank == 0:
            if iter > 0:
                print(iter, zr, alpha, t2 - t1, t3 - t2, t3 - t1)
            else:
                print(iter, zr, t2 - t1)
            sys.stdout.flush()
        t1 = time.time()
        Ap = tods.dot(p)
        if not (prior is None):
            # print('applying prior')
            prior.apply_prior(p, Ap)
        t2 = time.time()
        pAp = p.dot(Ap)
        alpha = zr / pAp
        try:
            x_new = x.copy()
            x_new.axpy(p, alpha)
        except:
            x_new = x + p * alpha

        try:
            r_new = r.copy()
            r_new.axpy(Ap, -alpha)
        except:
            r_new = r - Ap * alpha
        if not (precon is None):
            z_new = precon * r_new
        else:
            z_new = r_new.copy()
        zr_new = r_new.dot(z_new)
        beta = zr_new / zr
        try:
            p_new = z_new.copy()
            p_new.axpy(p, beta)
        except:
            p_new = z_new + p * beta

        p = p_new
        z = z_new
        r = r_new
        zr = zr_new
        x = x_new
        t3 = time.time()
        if iter in save_iters:
            if myrank == 0:
                x.maps[save_ind].write(outroot + "_" + repr(iter) + save_tail)

    return x


class null_precon:
    def __init__(self):
        self.isnull = True

    def __add__(self, val):
        return val

    def __mul__(self, val):
        return val
