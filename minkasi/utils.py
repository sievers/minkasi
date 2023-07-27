
def y2rj(freq=90):
    """conversion to multiply a y map by to get a Rayleigh-Jeans normalized map
    note that it doesn't have the T_cmb at the end, so the value for low frequencies
    is -2."""
    kb=1.38064852e-16
    h=6.62607004e-27
    T=2.725
    x=freq*1e9*h/kb/T
    
    ex=np.exp(x)
    f=x**2*ex/(ex-1)**2*( x*(ex+1)/(ex-1)-4)
    return f

def planck_g(freq=90):
    """conversion between T_CMB and T_RJ as a function of frequency."""
    kb=1.38064852e-16
    h=6.62607004e-27
    T=2.725
    x=freq*1e9*h/kb/T
    ex=np.exp(x)
    return x**2*ex/( (ex-1)**2)


def make_rings_wSlope(edges,cent,vals,map,pixsize=2.0,fwhm=10.0,amps=None,aa=1.0,bb=1.0,rot=0.0):
    xvec=np.arange(map.nx)
    yvec=np.arange(map.ny)
    xvec[map.nx//2:]=xvec[map.nx//2:]-map.nx
    yvec[map.ny//2:]=yvec[map.ny//2:]-map.ny

    xmat=np.repeat([xvec],map.ny,axis=0).transpose()
    ymat=np.repeat([yvec],map.nx,axis=0)

    rmat=np.sqrt(xmat**2+ymat**2)*pixsize
    if isinstance(fwhm,int)|isinstance(fwhm,float):
        sig=fwhm/np.sqrt(8*np.log(2.))
        src_map=np.exp(-0.5*rmat**2./sig**2)
        src_map=src_map/src_map.sum()
    else:
        sig=fwhm[0]/np.sqrt(8*np.log(2))
        src_map=np.exp(-0.5*rmat**2/sig**2)*amps[0]
        for i in range(1,len(fwhm)):
            sig=fwhm[i]/np.sqrt(8*np.log(2))
            src_map=src_map+np.exp(-0.5*rmat**2/sig**2)*amps[i]

        src_map=src_map/src_map.sum()
        beam_area=pixsize**2/src_map.max()
        beam_area=beam_area/3600**2/(360**2/np.pi)
        print('beam_area is ',beam_area*1e9,' nsr')
    nring=len(edges)-1
    rings=np.zeros([nring,map.nx,map.ny])
    mypix=map.wcs.wcs_world2pix(cent[0],cent[1],1)
    print('mypix is ',mypix)

    xvec=np.arange(map.nx)
    yvec=np.arange(map.ny)
    xmat=np.repeat([xvec],map.ny,axis=0).transpose()
    ymat=np.repeat([yvec],map.nx,axis=0)

    srcft=np.fft.fft2(src_map)
    xtr  = (xmat-mypix[0])*np.cos(rot) + (ymat-mypix[1])*np.sin(rot) # Rotate and translate x coords
    ytr  = (ymat-mypix[1])*np.cos(rot) - (xmat-mypix[0])*np.sin(rot) # Rotate and translate y coords
    rmat = np.sqrt( (xtr/aa)**2 + (ytr/bb)**2 ) * pixsize            # Elliptically scale x,y
    myvals = vals[:nring]*1.0   # Get just the values that correspond to rings
    myvals -= np.max(myvals) # Set it such that the maximum value approaches 0
    pk2pk = np.max(myvals) - np.min(myvals)
    myvals -= pk2pk/50.0       # Let's assume we're down about a factor of 50 at the outskirts.

    for i in range(nring):
        #rings[i,(rmat>=edges[i])&(rmat<edges[i+1]=1.0
        if i == nring-1:
            slope=0.0
        else:
            slope = (myvals[i]-myvals[i+1])/(edges[i+1]-edges[i]) # expect positve slope; want negative one.
        rgtinedge = (rmat>=edges[i])
        rfromin   = (rmat-edges[i])
        initline  = rfromin[rgtinedge]*slope
        if vals[i] != 0:
            rings[i,rgtinedge] = (myvals[i] - initline)/myvals[i]  # Should be normalized to 1 now.
        else:
            rings[i,rgtinedge] = 1.0
        rgtoutedge = (rmat>=edges[i+1])
        rings[i,rgtoutedge]=0.0
        myannul = [ c1 and not(c2) for c1,c2 in zip(rgtinedge.ravel(),rgtoutedge.ravel())]
        rannul  = rmat.ravel()[myannul]
        rmin    = (rmat == np.min(rannul))
        rmout   = (rmat == np.max(rannul))
        rings[i,:,:]=np.real(np.fft.ifft2(np.fft.fft2(rings[i,:,:])*srcft))
    return rings

def make_rings(edges,cent,map,pixsize=2.0,fwhm=10.0,amps=None,iswcs=True):
    xvec=np.arange(map.nx)
    yvec=np.arange(map.ny)
    ix=int(map.nx/2)
    iy=int(map.ny/2)
    xvec[ix:]=xvec[ix:]-map.nx
    yvec[iy:]=yvec[iy:]-map.ny
    #xvec[map.nx/2:]=xvec[map.nx/2:]-map.nx
    #yvec[map.ny/2:]=yvec[map.ny/2:]-map.ny

    xmat=np.repeat([xvec],map.ny,axis=0).transpose()
    ymat=np.repeat([yvec],map.nx,axis=0)

    rmat=np.sqrt(xmat**2+ymat**2)*pixsize
    if isinstance(fwhm,int)|isinstance(fwhm,float):
        sig=fwhm/np.sqrt(8*np.log(2))
        src_map=np.exp(-0.5*rmat**2/sig**2)
        src_map=src_map/src_map.sum()
    else:
        sig=fwhm[0]/np.sqrt(8*np.log(2))
        src_map=np.exp(-0.5*rmat**2/sig**2)*amps[0]
        for i in range(1,len(fwhm)):
            sig=fwhm[i]/np.sqrt(8*np.log(2))
            src_map=src_map+np.exp(-0.5*rmat**2/sig**2)*amps[i]

        src_map=src_map/src_map.sum()
        beam_area=pixsize**2/src_map.max()
        beam_area=beam_area/3600**2/(360**2/np.pi)
        print('beam_area is ',beam_area*1e9,' nsr')
    nring=len(edges)-1
    rings=np.zeros([nring,map.nx,map.ny])
    if iswcs:
        mypix=map.wcs.wcs_world2pix(cent[0],cent[1],1)
    else:
        mypix=cent

    print('mypix is ',mypix)

    xvec=np.arange(map.nx)
    yvec=np.arange(map.ny)
    xmat=np.repeat([xvec],map.ny,axis=0).transpose()
    ymat=np.repeat([yvec],map.nx,axis=0)


    srcft=np.fft.fft2(src_map)
    rmat=np.sqrt( (xmat-mypix[0])**2+(ymat-mypix[1])**2)*pixsize
    for i in range(nring):
        #rings[i,(rmat>=edges[i])&(rmat<edges[i+1]=1.0
        rings[i,(rmat>=edges[i])]=1.0
        rings[i,(rmat>=edges[i+1])]=0.0
        rings[i,:,:]=np.real(np.fft.ifft2(np.fft.fft2(rings[i,:,:])*srcft))
    return rings
        





def nsphere_vol(npp):
    iseven=(npp%2)==0
    if iseven:
        nn=npp/2
        vol=(np.pi**nn)/np.prod(np.arange(1,nn+1))
    else:
        nn=(npp-1)/2
        vol=2**(nn+1)*np.pi**nn/np.prod(np.arange(1,npp+1,2))
    return vol


def _prime_loop(ln,lp,icur,lcur,vals):
    facs=np.arange(lcur,ln+1e-3,lp[0])
    if len(lp)==1:
        nfac=len(facs)
        if (nfac>0):
            vals[icur:(icur+nfac)]=facs
            icur=icur+nfac
            #print 2**vals[:icur]
        else:
            print('bad facs came from ' + repr([2**lcur,2**ln,2**lp[0]]))
        #print icur
        return icur
    else:
        facs=np.arange(lcur,ln,lp[0])
        for fac in facs:
            icur=_prime_loop(ln,lp[1:],icur,fac,vals)
        return icur
    print('I don''t think I should have gotten here.')
    return icur
                             
        

def find_good_fft_lens(n,primes=[2,3,5,7]):
    lmax=np.log(n+0.5)
    npr=len(primes)
    vol=nsphere_vol(npr)

    r=np.log2(n+0.5)
    lp=np.log2(primes)
    int_max=(vol/2**npr)*np.prod(r/lp)+30 #add a bit just to make sure we don't act up for small n
    #print 'int max is ',int max
    int_max=int(int_max)

    #vals=np.zeros(int_max,dtype='int')
    vals=np.zeros(int_max)
    icur=0
    icur=_prime_loop(r,lp,icur,0.0,vals)
    assert(icur<=int_max)
    myvals=np.asarray(np.round(2**vals[:icur]),dtype='int')
    myvals=np.sort(myvals)
    return myvals
def plot_ps(vec,downsamp=0):
    vecft=mkfftw.fft_r2r(vec)
