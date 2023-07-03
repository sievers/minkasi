
def run_pcg(b,x0,tods,precon=None,maxiter=25,outroot='map',save_iters=[-1],save_ind=0,save_tail='.fits',plot_iters=[],plot_info=None,plot_ind=0):
    """
    Function which runs preconditioned conjugate gradient on a bundle of tods to generate a map.
    PCG itteratively approximates the solution to the linear equation Ax = b for A a matrix, x
    and b vectors. In the map making equation,  A = P'N"P and b = P'N"d for d the vector of TODs,
    N the noise matrix,  P the tod to map pointing matrix, i.e. a matrix that specifies which 
    pixel in the map was observed by each TOD data point. Futher x is the map. 
    Arguments:
        b: The rhs of the equation. In our case this is P'N''d. The tod class has a built in 
        method for computing this. 

        x0: The initial guess. Generally set to for the first itteration and then to the output
        of the previous itteration.
        
        tods: the input tods we want to make into maps. Note the noise has already been estimated        and is within the tod object. 

        precon: The preconditioner. A matrix applied to A to ensure faster convergence. 1/hitsmap
        is a frequent selection.

        maxiter: Maximum number of iterations to perform. 
 
        outroot: location at which to save the output map

        save_iters: The iterations at which to save the result map. Default is to save only the 
        last

        save_ind:

        save_tail: Extention for saving the output maps

        plot_iters: Which iterations to plot

        plot_info: 

        plot_ind: 

    Outputs:
        x: best guess for x after the conversion criteria has been reached (either max iter or
        Ax = b close enough to 0
    """
    t1=time.time()
    Ax=tods.dot(x0)
    
    try:
        #compute the remainder r_0
        r=b.copy()
        r.axpy(Ax,-1)
    except:
        r=b-Ax
    if not(precon is None):
        #print('applying precon')
        # z_0 = M*r_0
        z=precon*r
        key = tods.tods[0].info['fname']
    
    else:
        z=r.copy()

    #Initial p_0 = z_0 = M*r_0
    p=z.copy()
    k=0.0

    #compute z*r, which is used for computing alpha
    zr=r.dot(z)
    #make a copy of our initial guess
    x=x0.copy()
    t2=time.time()
    nsamp=tods.get_nsamp()
    tloop=time.time()
    for iter in range(maxiter):
        if myrank==0:
            if iter>0:
                print(iter,zr,alpha,t2-t1,t3-t2,t3-t1,nsamp/(t2-t1)/1e6)
            else:
                print(iter,zr,t2-t1)
        t1=time.time()
        #Compute pAp
        Ap=tods.dot(p)
        t2=time.time()
        pAp=p.dot(Ap)
        #Compute alpha_k
        alpha=zr/pAp
        #print('alpha,pAp, and zr  are ' + repr(alpha) + '  ' + repr(pAp) + '  ' + repr(zr))
        try:
            #Update guess using alpha
            x_new=x.copy()
            x_new.axpy(p,alpha)
        except:
            x_new=x+p*alpha

        try:
            #Write down next remainder r_k+1
            r_new=r.copy()
            r_new.axpy(Ap,-alpha)
        except:
            r_new=r-Ap*alpha
        if not(precon is None):
            #print('applying precon')
            z_new=precon*r_new
        else:
            z_new=r_new.copy()
        #compute new z_k+1
        zr_new=r_new.dot(z_new)
        #compute beta_k, which is used to compute p_k+1
        beta=zr_new/zr
        try:
            #compute new p_k+1
            p_new=z_new.copy()
            p_new.axpy(p,beta)
        except:
            p_new=z_new+p*beta
        #Update values
        p=p_new
        z=z_new
        r=r_new
        zr=zr_new
        x=x_new
        t3=time.time()
        if iter in save_iters:
            if myrank==0:
                x.maps[save_ind].write(outroot+'_'+repr(iter)+save_tail)
        if iter in plot_iters:
            print('plotting on iteration ',iter)
            x.maps[plot_ind].plot(plot_info)

    tave=(time.time()-tloop)/maxiter
    print('average time per iteration was ',tave,' with effective throughput ',nsamp/tave/1e6,' Msamp/s')
    if iter in plot_iters:
        print('plotting on iteration ',iter)
        x.maps[plot_ind].plot(plot_info)
    else:
        print('skipping plotting on iter ',iter)
    return x

def run_pcg_wprior(b,x0,tods,prior=None,precon=None,maxiter=25,outroot='map',save_iters=[-1],save_ind=0,save_tail='.fits'):
    #least squares equations in the presence of a prior - chi^2 = (d-Am)^T N^-1 (d-Am) + (p-m)^T Q^-1 (p-m)
    #where p is the prior target for parameters, and Q is the variance.  The ensuing equations are
    #(A^T N-1 A + Q^-1)m = A^T N^-1 d + Q^-1 p.  For non-zero p, it is assumed you have done this already and that 
    #b=A^T N^-1 d + Q^-1 p
    #to have a prior then, whenever we call Ax, just a Q^-1 x to Ax.
    t1=time.time()
    Ax=tods.dot(x0)    
    if not(prior is None):
        #print('applying prior')
        prior.apply_prior(x0,Ax) 
    try:
        r=b.copy()
        r.axpy(Ax,-1)
    except:
        r=b-Ax
    if not(precon is None):
        z=precon*r
    else:
        z=r.copy()
    p=z.copy()
    k=0.0

    zr=r.dot(z)
    x=x0.copy()
    t2=time.time()
    for iter in range(maxiter):
        if myrank==0:
            if iter>0:
                print(iter,zr,alpha,t2-t1,t3-t2,t3-t1)
            else:
                print(iter,zr,t2-t1)
            sys.stdout.flush()
        t1=time.time()
        Ap=tods.dot(p)
        if not(prior is None):
            #print('applying prior')
            prior.apply_prior(p,Ap)
        t2=time.time()
        pAp=p.dot(Ap)
        alpha=zr/pAp
        try:
            x_new=x.copy()
            x_new.axpy(p,alpha)
        except:
            x_new=x+p*alpha

        try:
            r_new=r.copy()
            r_new.axpy(Ap,-alpha)
        except:
            r_new=r-Ap*alpha
        if not(precon is None):
            z_new=precon*r_new
        else:
            z_new=r_new.copy()
        zr_new=r_new.dot(z_new)
        beta=zr_new/zr
        try:
            p_new=z_new.copy()
            p_new.axpy(p,beta)
        except:
            p_new=z_new+p*beta
        
        p=p_new
        z=z_new
        r=r_new
        zr=zr_new
        x=x_new
        t3=time.time()
        if iter in save_iters:
            if myrank==0:
                x.maps[save_ind].write(outroot+'_'+repr(iter)+save_tail)

    return x

class null_precon:
    def __init__(self):
        self.isnull=True
    def __add__(self,val):
        return val
    def __mul__(self,val):
        return val

