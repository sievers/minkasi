import numpy

def zernike_column(m,nmax,rmat):
    """Generate the radial part of zernike polynomials for all n from m up to nmax"""

    if ((m-nmax)%2!=0):
        #print 'm an n must have same parity'
        #return None
        #if parity is wrong, then drop nmax by one.  makes external loop to generate all zns much simpler
        nmax=nmax-1
    if (m>nmax):
        print 'm may not be larger than n'
        return None
    nm=(nmax-m)/2+1

    mask=rmat>1

    zn=[None]*nm
    nn=numpy.zeros(nm,dtype='int')

    zn[0]=rmat**m
    zn[0][mask]=0
    nn[0]=m
    if nm==1:
        return zn,nn
    rsqr=rmat**2
    zn[1]=((m+2)*rsqr-m-1)*zn[0]
    zn[1][mask]=0
    nn[1]=m+2
    if nm==2:
        return zn,nn

    ii=2
    for n in range(m+4,nmax+1,2):

        f1=2*(n-1)*(2*n*(n-2)*rsqr-m*m-n*(n-2))*zn[ii-1]
        f2=n*(n+m-2)*(n-m-2)*zn[ii-2]
        f3=1.0/((n+m)*(n-m)*(n-2))    
        zn[ii]=(f1-f2)*f3
        nn[ii]=n
        ii=ii+1

    return zn,nn


def all_zernike(n,r,th):
    znvec=[None]*(n+1)
    nvec=[None]*(n+1)
    mvec=[None]*(n+1)
    nzer=0
    for m in range(0,n+1):
        znvec[m],nvec[m]=zernike_column(m,n,r)
        mvec[m]=0*nvec[m]+m
        if m==0:
            nzer=nzer+len(znvec[m])
        else:
            nzer=nzer+2*len(znvec[m])
    #print nzer
    
    shp=r.shape
    shp=numpy.append(nzer,shp)
    zns=numpy.zeros(shp)
    #print shp

    icur=0
    #print n,len(znvec)
    for m in range(0,n+1):
        #print icur
        ss=numpy.sin(m*th)
        cc=numpy.cos(m*th)
        zz=znvec[m]
        
        nn=len(zz)
        if m==0:
            for i in range(nn):
                zns[icur,:]=zz[i]
                icur=icur+1

        else:
            for i in range(nn):
                zns[icur,:]=zz[i]*cc
                icur=icur+1
                zns[icur,:]=zz[i]*ss
                icur=icur+1

    return zns,znvec

