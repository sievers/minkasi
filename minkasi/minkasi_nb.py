import numpy as np
import numba as nb

@nb.njit(parallel=True)
def map2tod_destriped(mat,pars,lims,do_add=True):
    ndet=mat.shape[0]
    nseg=len(lims)-1
    for seg in nb.prange(nseg):
        for det in range(ndet):
            if do_add:
                for i in range(lims[seg],lims[seg+1]):
                    mat[det,i]=mat[det,i]+pars[det,seg]
            else:
                for i in range(lims[seg],lims[seg+1]):
                    mat[det,i]=pars[det,seg]

@nb.njit(parallel=True)
def tod2map_destriped(mat,pars,lims,do_add=True):
    ndet=mat.shape[0]
    nseg=len(lims)-1
    for seg in nb.prange(nseg):
        for det in range(ndet):
            if do_add==False:
                pars[det,seg]=0
            for i in range(lims[seg],lims[seg+1]):
                pars[det,seg]=pars[det,seg]+mat[det,i]
                

@nb.njit(parallel=True)
def __map2tod_binned_det_loop(pars,inds,mat,ndet,n):
    for det in nb.prange(ndet):
        for i in range(n):
            mat[det][i]=mat[det][i]+pars[det][inds[i]]
            #pars[det][inds[i]]=pars[det][inds[i]]+mat[det][i]


def map2tod_binned_det(mat,pars,vec,lims,nbin,do_add=True):
    n=mat.shape[1]
    #print('range is ',pars.min(),pars.max())
    #inds=np.empty(n,dtype='int64')
    fac=nbin/(lims[1]-lims[0]) 
    inds=np.asarray((vec-lims[0])*fac,dtype='int64')
    #print('ind range is ',inds.min(),inds.max())
    #for i in nb.prange(n):
    #    inds[i]=(vec[i]-lims[0])*fac
    ndet=mat.shape[0]
    if do_add==False:
        mat[:]=0
    __map2tod_binned_det_loop(pars,inds,mat,ndet,n)
    #for det in nb.prange(ndet):
    #    for i in np.arange(n):
    #        pars[det][inds[i]]=pars[det][inds[i]]+mat[det][i]



@nb.njit(parallel=True)
def __tod2map_binned_det_loop(pars,inds,mat,ndet,n):
    for det in nb.prange(ndet):
        for i in range(n):
            pars[det][inds[i]]=pars[det][inds[i]]+mat[det][i]
            
def tod2map_binned_det(mat,pars,vec,lims,nbin,do_add=True):
    #print('dims are ',mat.shape,pars.shape,vec.shape)
    #print('lims are ',lims,nbin,vec.min(),vec.max())
    n=mat.shape[1]
    
    fac=nbin/(lims[1]-lims[0]) 
    #inds=np.empty(n,dtype='int64')
    #for i in nb.prange(n):
    #    inds[i]=(vec[i]-lims[0])*fac
    inds=np.asarray((vec-lims[0])*fac,dtype='int64')

    #print('max is ',inds.max())
    ndet=mat.shape[0]
    if do_add==False:
        mat[:]=0
    __tod2map_binned_det_loop(pars,inds,mat,ndet,n)
    #for det in nb.prange(ndet):
    #    for i in np.arange(n):
    #        pars[det][inds[i]]=pars[det][inds[i]]+mat[det][i]
    return 0
            


#@nb.njit(parallel=True)
#def map2tod_binned_det(mat,pars,vec,lims,nbin,do_add=True):
#    n=mat.shape[1]
#    inds=np.empty(n,dtype='int')
#    fac=nbin/(lims[1]-lims[0]) 
#    for i in nb.prange(n):
#        inds[i]=(vec[i]-lims[0])*fac
#    ndet=mat.shape[0]
#    if do_add==False:
#        mat[:]=0
#    for det in np.arange(ndet):
#        for i in nb.prange(n):
#            mat[det][i]=mat[det][i]+pars[det][inds[i]]


@nb.njit(parallel=True)
def fill_elliptical_isobeta(params,dx,dy,pred):
    ndet=dx.shape[0]
    n=dx.shape[1]
    x0=params[0]
    y0=params[1]
    theta1=params[2]
    theta2=params[3]
    theta1_inv=1/theta1
    theta2_inv=1/theta2
    theta1_inv_sqr=theta1_inv**2
    theta2_inv_sqr=theta2_inv**2
    psi=params[4]
    beta=params[5]
    amp=params[6]
    cosdec=np.cos(y0)
    cospsi=np.cos(psi)
    sinpsi=np.sin(psi)
    mypow=0.5-1.5*beta
    for det in nb.prange(ndet):
        for j in np.arange(n):
            delx=(dx[det,j]-x0)*cosdec
            dely=dy[det,j]-y0
            xx=delx*cospsi+dely*sinpsi
            yy=dely*cospsi-delx*sinpsi
            rr=1+theta1_inv_sqr*xx*xx+theta2_inv_sqr*yy*yy
            pred[det,j]=amp*(rr**mypow)



@nb.njit(parallel=True)
def fill_elliptical_isobeta_derivs(params,dx,dy,pred,derivs):
    """Fill model/derivatives for an isothermal beta model.  
    Parameters should be [ra,dec,theta axis 1,theta axis 2,angle,beta,amplitude.
    Beta should be positive (i.e. 0.7, not -0.7)."""

    ndet=dx.shape[0]
    n=dx.shape[1]
    x0=params[0]
    y0=params[1]
    theta1=params[2]
    theta2=params[3]
    theta1_inv=1/theta1
    theta2_inv=1/theta2
    theta1_inv_sqr=theta1_inv**2
    theta2_inv_sqr=theta2_inv**2
    psi=params[4]
    beta=params[5]
    amp=params[6]
    cosdec=np.cos(y0)
    sindec=np.sin(y0)/np.cos(y0)
    #cosdec=np.cos(dy[0,0])
    #cosdec=1.0
    cospsi=np.cos(psi)
    cc=cospsi**2
    sinpsi=np.sin(psi)
    ss=sinpsi**2
    cs=cospsi*sinpsi
    mypow=0.5-1.5*beta
    for det in nb.prange(ndet):
        for j in np.arange(n):
            delx=(dx[det,j]-x0)*cosdec
            dely=dy[det,j]-y0
            xx=delx*cospsi+dely*sinpsi
            yy=dely*cospsi-delx*sinpsi
            xfac=theta1_inv_sqr*xx*xx
            yfac=theta2_inv_sqr*yy*yy
            #rr=1+theta1_inv_sqr*xx*xx+theta2_inv_sqr*yy*yy
            rr=1+xfac+yfac
            rrpow=rr**mypow
            
            pred[det,j]=amp*rrpow
            dfdrr=rrpow/rr*mypow
            drdx=-2*delx*(cc*theta1_inv_sqr+ss*theta2_inv_sqr)-2*dely*(theta1_inv_sqr-theta2_inv_sqr)*cs
            #drdy=-2*dely*(cc*theta2_inv_sqr+ss*theta1_inv_sqr)-2*delx*(theta1_inv_sqr-theta2_inv_sqr)*cs
            drdy=-(2*xx*theta1_inv_sqr*(cospsi*sindec*delx+sinpsi)+2*yy*theta2_inv_sqr*(-sinpsi*sindec*delx+cospsi))
            drdtheta=2*(theta1_inv_sqr-theta2_inv_sqr)*(cs*(dely**2-delx**2)+delx*dely*(cc-ss))
            #drdtheta=-2*delx**2*cs*(theta_1_inv_sqr-theta_2_inv_sqr)+2*dely*delx*(theta_1_inv_sqr-theta_2_inv_sqr)*(cc-ss)+2*dely**2*cs*(

            
            derivs[0,det,j]=dfdrr*drdx*cosdec
            derivs[1,det,j]=dfdrr*drdy
            derivs[2,det,j]=dfdrr*xfac*(-2*theta1_inv)
            derivs[3,det,j]=dfdrr*yfac*(-2*theta2_inv)
            derivs[4,det,j]=dfdrr*drdtheta
            derivs[5,det,j]=-1.5*np.log(rr)*amp*rrpow
            derivs[6,det,j]=rrpow


@nb.njit(parallel=True)
def fill_elliptical_gauss_derivs(params,dx,dy,pred,derivs):
    """Fill model/derivatives for an elliptical gaussian model.  
    Parameters should be [ra,dec,sigma axis 1,sigmaaxis 2,angle,amplitude."""


    ndet=dx.shape[0]
    n=dx.shape[1]
    x0=params[0]
    y0=params[1]
    theta1=params[2]
    theta2=params[3]
    theta1_inv=1/theta1
    theta2_inv=1/theta2
    theta1_inv_sqr=theta1_inv**2
    theta2_inv_sqr=theta2_inv**2
    psi=params[4]
    amp=params[5]
    cosdec=np.cos(y0)
    sindec=np.sin(y0)/np.cos(y0)
    cospsi=np.cos(psi)
    cc=cospsi**2
    sinpsi=np.sin(psi)
    ss=sinpsi**2
    cs=cospsi*sinpsi

    for det in nb.prange(ndet):
        for j in np.arange(n):
            delx=(dx[det,j]-x0)*cosdec
            dely=dy[det,j]-y0
            xx=delx*cospsi+dely*sinpsi
            yy=dely*cospsi-delx*sinpsi
            xfac=theta1_inv_sqr*xx*xx
            yfac=theta2_inv_sqr*yy*yy
            #rr=1+theta1_inv_sqr*xx*xx+theta2_inv_sqr*yy*yy
            rr=xfac+yfac
            rrpow=np.exp(-0.5*rr)
            
            pred[det,j]=amp*rrpow
            dfdrr=-0.5*rrpow
            drdx=-2*delx*(cc*theta1_inv_sqr+ss*theta2_inv_sqr)-2*dely*(theta1_inv_sqr-theta2_inv_sqr)*cs
            #drdy=-2*dely*(cc*theta2_inv_sqr+ss*theta1_inv_sqr)-2*delx*(theta1_inv_sqr-theta2_inv_sqr)*cs
            drdy=-(2*xx*theta1_inv_sqr*(cospsi*sindec*delx+sinpsi)+2*yy*theta2_inv_sqr*(-sinpsi*sindec*delx+cospsi))

            drdtheta=2*(theta1_inv_sqr-theta2_inv_sqr)*(cs*(dely**2-delx**2)+delx*dely*(cc-ss))
            #drdtheta=-2*delx**2*cs*(theta_1_inv_sqr-theta_2_inv_sqr)+2*dely*delx*(theta_1_inv_sqr-theta_2_inv_sqr)*(cc-ss)+2*dely**2*cs*(

            
            derivs[0,det,j]=dfdrr*drdx*cosdec
            #derivs[1,det,j]=dfdrr*(drdy-2*sindec*delx**2*theta1_inv_sqr)
            derivs[1,det,j]=dfdrr*drdy
            derivs[2,det,j]=dfdrr*xfac*(-2*theta1_inv)
            derivs[3,det,j]=dfdrr*yfac*(-2*theta2_inv)
            derivs[4,det,j]=dfdrr*drdtheta
            derivs[5,det,j]=rrpow


@nb.njit(parallel=True)
def radec2pix_car(ra,dec,ipix,lims,pixsize,cosdec,ny):
    ra=np.ravel(ra)
    dec=np.ravel(dec)
    ipix=np.ravel(ipix)
    n=len(ipix)
    for i in nb.prange(n):
        xpix=int((ra[i]-lims[0])*cosdec/pixsize+0.5)
        ypix=int((dec[i]-lims[2])/pixsize+0.5)
        ipix[i]=xpix*ny+ypix

@nb.njit(parallel=True)
def axpy_in_place(y,x,a=1.0):
    #add b into a
    n=x.shape[0]
    m=x.shape[1]
    assert(n==y.shape[0])
    assert(m==y.shape[1])
    #Numba has a bug, as of at least 0.53.1 (an 0.52.0) where
    #both parts of a conditional can get executed, so don't 
    #try to be fancy.  Lower-down code can be used in the future.
    for i in nb.prange(n):
        for j in np.arange(m):
            y[i,j]=y[i,j]+x[i,j]*a
    
    #isone=(a==1.0)
    #if isone:
    #    for  i in nb.prange(n):
    #        for j in np.arange(m):
    #            y[i,j]=y[i,j]+x[i,j]
    #else:
    #    for i in nb.prange(n):
    #        for j in np.arange(m):
    #            y[i,j]=y[i,j]+x[i,j]*a

@nb.njit(parallel=True)
def scale_matrix_by_vector(mat,vec,axis=1):
    n=mat.shape[0]
    m=mat.shape[1]
    if axis==1:
        assert(len(vec)==n)
        for i in nb.prange(n):
            for j in np.arange(m):
                mat[i,j]=mat[i,j]*vec[i]
    elif axis==0:
        assert(len(vec)==m)
        for i in nb.prange(n):
            for j in np.arange(m):
                mat[i,j]=mat[i,j]*vec[j]
    else:
        print('unsupported number of dimensions in scale_matrix_by_vector')
