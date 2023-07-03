
def find_spikes(dat,inner=1,outer=10,rad=0.25,thresh=8,pad=2):
    #find spikes in a block of timestreams
    n=dat.shape[1];
    ndet=dat.shape[0]
    x=np.arange(n);
    filt1=np.exp(-0.5*x**2/inner**2)
    filt1=filt1+np.exp(-0.5*(x-n)**2/inner**2);
    filt1=filt1/filt1.sum()

    filt2=np.exp(-0.5*x**2/outer**2)
    filt2=filt2+np.exp(-0.5*(x-n)**2/outer**2);
    filt2=filt2/filt2.sum()
    
    filt=filt1-filt2 #make a filter that is the difference of two Gaussians, one narrow, one wide
    filtft=np.fft.rfft(filt)
    datft=np.fft.rfft(dat,axis=1)
    datfilt=np.fft.irfft(filtft*datft,axis=1,n=n)
    jumps=[None]*ndet
    mystd=np.median(np.abs(datfilt),axis=1)
    for i in range(ndet):
        while np.max(np.abs(datfilt[i,:]))>thresh*mystd[i]:
            ind=np.argmax(np.abs(datfilt[i,:]))
            if jumps[i] is None:
                jumps[i]=[ind]
            else:
                jumps[i].append(ind)
            datfilt[i,ind]=0
    return jumps,datfilt
    return mystd
def find_jumps(dat,width=10,pad=2,thresh=10,rat=0.5):
    #find jumps in a block of timestreams, preferably with the common mode removed
    #width is width in pixels to average over when looking for a jump
    #pad is the length in units of width to mask at beginning/end of timestream
    #thresh is threshold in units of filtered data median absolute deviation to qualify as a jump
    #rat is the ratio of largest neighboring opposite-sign jump to the found jump.  If
    #  there is an opposite-sign jump nearby, the jump finder has probably just picked up a spike.
    n=dat.shape[1]
    ndet=dat.shape[0]

    #make a filter template that is a gaussian with sigma with, sign-flipped in the center
    #so, positive half-gaussian starting from zero, and negative half-gaussian at the end
    x=np.arange(n)
    myfilt=np.exp(-0.5*x**2/width**2)
    myfilt=myfilt-np.exp( (-0.5*(x-n)**2/width**2))
    fac=np.abs(myfilt).sum()/2.0
    myfilt=myfilt/fac

    dat_filt=np.fft.rfft(dat,axis=1)

    myfilt_ft=np.fft.rfft(myfilt)
    dat_filt=dat_filt*np.repeat([myfilt_ft],ndet,axis=0)
    dat_filt=np.fft.irfft(dat_filt,axis=1,n=n)
    dat_filt_org=dat_filt.copy()

    print(dat_filt.shape)
    dat_filt[:,0:pad*width]=0
    dat_filt[:,-pad*width:]=0
    det_thresh=thresh*np.median(np.abs(dat_filt),axis=1)
    dat_dejump=dat.copy()
    jumps=[None]*ndet
    print('have filtered data, now searching for jumps')
    for i in range(ndet):
        while np.max(np.abs(dat_filt[i,:]))>det_thresh[i]:            
            ind=np.argmax(np.abs(dat_filt[i,:]))+1 #+1 seems to be the right index to use
            imin=ind-width
            if imin<0:
                imin=0
            imax=ind+width
            if imax>n:
                imax=n
            val=dat_filt[i,ind]
            if val>0:
                val2=np.min(dat_filt[i,imin:imax])
            else:
                val2=np.max(dat_filt[i,imin:imax])
            
            
            print('found jump on detector ',i,' at sample ',ind)
            if np.abs(val2/val)>rat:
                print('I think this is a spike due to ratio ',np.abs(val2/val))
            else:
                if jumps[i] is None:
                    jumps[i]=[ind]
                else:
                    jumps[i].append(ind)
            #independent of if we think it is a spike or a jump, zap that stretch of the data
            dat_dejump[i,ind:]=dat_dejump[i,ind:]+dat_filt[i,ind]
            dat_filt[i,ind-pad*width:ind+pad*width]=0
        if not(jumps[i] is None):
            jumps[i]=np.sort(jumps[i])
    #return dat_dejump,jumps,dat_filt_org
    return jumps

def fit_jumps_from_cm(dat,jumps,cm,cm_order=1,poly_order=1):
    jump_vals=jumps[:]
    ndet=len(jumps)
    n=dat.shape[1]
    x=np.linspace(-1,1,n)
    m1=np.polynomial.legendre.legvander(x,poly_order)
    m2=np.polynomial.legendre.legvander(x,cm_order-1)
    for i in range(cm_order):
        m2[:,i]=m2[:,i]*cm
    mat=np.append(m1,m2,axis=1)
    npp=mat.shape[1]

    dat_dejump=dat.copy()
    for i in range(ndet):
        if not(jumps[i] is None):
            njump=len(jumps[i])
            segs=np.append(jumps[i],n)
            print('working on detector ',i,' who has ', len(jumps[i]),' jumps with segments ',segs)
            mm=np.zeros([n,npp+njump])
            mm[:,:npp]=mat
            for j in range(njump):
                mm[segs[j]:segs[j+1],j+npp]=1.0
            lhs=np.dot(mm.transpose(),mm)
            #print lhs
            rhs=np.dot(mm.transpose(),dat[i,:].transpose())
            lhs_inv=np.linalg.inv(lhs)
            fitp=np.dot(lhs_inv,rhs)
            jump_vals[i]=fitp[npp:]
            jump_pred=np.dot(mm[:,npp:],fitp[npp:])
            dat_dejump[i,:]=dat_dejump[i,:]-jump_pred


    return dat_dejump
            

    #for i in range(ndet):
def gapfill_eig(dat,cuts,tod=None,thresh=5.0, niter_eig=3, niter_inner=3, insert_cuts=False):
    ndat=dat.shape[1]
    cuts_empty=cuts.copy() #use this to clear out cut samples
    cuts_empty.clear() 
    cuts_cur=cuts.copy()
    cuts_cur.clear()
    for eig_ctr in range(niter_eig):
        tmp=dat.copy()
        cuts_cur.map2tod(tod,tmp,do_add=False)
        mycov=np.dot(tmp,tmp.T)
        ee,vv=np.linalg.eig(mycov)
        mask=ee>thresh*thresh*np.median(ee)
        neig=np.sum(mask)
        print('working with ' + repr(neig) + ' eigenvectors.')
        ee=ee[mask]
        vv=vv[:,mask]
        uu=np.dot(vv.T,tmp)
        lhs=np.dot(uu,uu.T)
        lhs_inv=np.linalg.inv(lhs)
        for iter_ctr in range(niter_inner):
            #in this inner loop, we fit the data 
            rhs=np.dot(tmp,uu.T)
            fitp=np.dot(lhs_inv,rhs.T)
            pred=np.dot(fitp.T,uu)
            cuts_cur.tod2map(tod,pred,do_add=False)
            cuts_cur.map2tod(tod,tmp,do_add=False)
    if insert_cuts:
        cuts_cur.map2tod(dat)
    return cuts_cur
        

def __gapfill_eig_poly(dat,cuts,tod=None,npoly=2, thresh=5.0, niter_eig=3, niter_inner=3):
    assert(1==0) #this code is not yet working.  regular gapfill_eig should work since the polys could
                 #be described by SVD, so SVD modes should look like polys iff they would have been important
    ndat=dat.shape[1]
    if npoly>0:
        xvec=np.linspace(-1,1,ndat)
        polymat=np.polynomial.legendre.legvander(x,npoly-1)
    old_coeffs=None
    cuts_cur=cuts.copy()    
    cuts_cur.clear()
    cuts_empty.cuts.copy()
    cuts_empty.clear()
    for eig_ctr in range(niter_eig):
        tmp=dat.copy()
        cuts_cur.map2tod(tod,tmp,do_add=False) #insert current best-guess solution for the cuts
        if npoly>1:  #if we're fitting polynomials as well as eigenmodes, subtract them off before re-estimating the covariance
            if not(old_coeffs is None):
                tmp=tmp-np.dot(polymat,old_coeffs[neig:,:]).T
        mycov=np.dot(tmp,tmp.T)
        mycov=0.5*(mycov+mycov.T)
        ee,vv=np.linalg.eig(mycov)
        mode_map=ee>thresh*thresh*np.median(ee)
        neig=mode_map.sum()
        mat=np.zeros([ndat,neig+npoly])
        eigs=vv[:,mode_map]
        ts_vecs=np.dot(eigs.T,tmp)
        mat[:,:neig]=ts_vecs.T
        if npoly>0:
            mat[:,neig:]=polymat
        lhs=np.dot(mat.T,mat)
        lhs_inv=np.linalg.inv(lhs)
        #now that we have the vectors we expect to describe our data, do a few rounds
        #of fitting amplitudes to timestream models, subtract that off, assign cuts to zero,
        #and restore the model.  
        tmp=dat.copy()
        for inner_ctr in range(niter_inner):
            cuts_cur.map2tod(tod,tmp)
            rhs=np.dot(tmp,mat)
            fitp=np.dot(lhs_inv,rhs.T)
            pred=np.dot(mat,fitp).T
            
def fit_cm_plus_poly(dat,ord=2,cm_ord=1,niter=2,medsub=False,full_out=False):
    n=dat.shape[1]
    ndet=dat.shape[0]
    if medsub:
        med=np.median(dat,axis=1)        
        dat=dat-np.repeat([med],n,axis=0).transpose()
        
        

    xx=np.arange(n)+0.0
    xx=xx-xx.mean()
    xx=xx/xx.max()

    pmat=np.polynomial.legendre.legvander(xx,ord)
    cm_pmat=np.polynomial.legendre.legvander(xx,cm_ord-1)
    calfacs=np.ones(ndet)*1.0
    dd=dat.copy()
    for i in range(1,niter):
        for j in range(ndet):
            dd[j,:]/=calfacs[j]
            
        cm=np.median(dd,axis=0)
        cm_mat=np.zeros(cm_pmat.shape)
        for i in range(cm_mat.shape[1]):
            cm_mat[:,i]=cm_pmat[:,i]*cm
        fitp_p,fitp_cm=_linfit_2mat(dat.transpose(),pmat,cm_mat)
        pred1=np.dot(pmat,fitp_p).transpose()
        pred2=np.dot(cm_mat,fitp_cm).transpose()
        pred=pred1+pred2
        dd=dat-pred1
        
    if full_out:
        return dd,pred2,cm #if requested, return the modelled CM as well
    return dd

def find_bad_skew_kurt(dat,skew_thresh=6.0,kurt_thresh=5.0):
    ndet=dat.shape[0]
    isgood=np.ones(ndet,dtype='bool')
    skew=np.mean(dat**3,axis=1)
    mystd=np.std(dat,axis=1)
    skew=skew/mystd**1.5
    mykurt=np.mean(dat**4,axis=1)
    kurt=mykurt/mystd**4-3
    
    isgood[np.abs(skew)>skew_thresh*np.median(np.abs(skew))]=False
    isgood[np.abs(kurt)>kurt_thresh*np.median(np.abs(kurt))]=False
    


    return skew,kurt,isgood

def downsample_array_r2r(arr,fac):

    n=arr.shape[1]
    nn=int(n/fac)
    arr_ft=mkfftw.fft_r2r(arr)
    arr_ft=arr_ft[:,0:nn].copy()
    arr=mkfftw.fft_r2r(arr_ft)/(2*(n-1))
    return arr

def downsample_vec_r2r(vec,fac):

    n=len(vec)
    nn=int(n/fac)
    vec_ft=mkfftw.fft_r2r(vec)
    vec_ft=vec_ft[0:nn].copy()
    vec=mkfftw.fft_r2r(vec_ft)/(2*(n-1))
    return vec

def downsample_tod(dat,fac=10):
    ndata=dat['dat_calib'].shape[1]
    keys=dat.keys()
    for key in dat.keys():
        try:
            if len(dat[key].shape)==1:
                #print('working on downsampling ' + key)
                #print('shape is ' + repr(dat[key].shape[0])+'  '+repr(n))
                if len(dat[key]):
                    #print('working on downsampling ' + key)
                    dat[key]=downsample_vec_r2r(dat[key],fac)
            else:
                if dat[key].shape[1]==ndata:
                #print 'downsampling ' + key
                    dat[key]=downsample_array_r2r(dat[key],fac)
        except:
            #print 'not downsampling ' + key
            pass
    

def truncate_tod(dat,primes=[2,3,5,7,11]):
    n=dat['dat_calib'].shape[1]
    lens=find_good_fft_lens(n-1,primes)
    n_new=lens.max()+1
    if n_new<n:
        print('truncating from ',n,' to ',n_new)
        for key in dat.keys():
            try:
                #print('working on key ' + key)
                if len(dat[key].shape)==1:
                    if dat[key].shape[0]==n:
                        dat[key]=dat[key][:n_new].copy()
                else:
                    if dat[key].shape[1]==n:
                        dat[key]=dat[key][:,0:n_new].copy()
            except:
                #print('skipping key ' + key)
                pass

def fit_mat_vecs_poly_nonoise(dat,mat,order,cm_order=None):
    if cm_order is None:
        cm_order=order
    n=dat.shape[1]
    x=np.linspace(-1,1,n)
    polys=np.polynomial.legendre.legvander(x,order).transpose()
    cm_polys=np.polynomial.legendre.legvander(x,cm_order).transpose()
    v1=np.sum(dat,axis=0)
    v2=np.sum(dat*mat,axis=0)
    rhs1=np.dot(cm_polys,v1)
    rhs2=np.dot(polys,v2)
    ndet=dat.shape[0]
    A1=cm_polys*ndet
    vv=np.sum(mat,axis=0)
    A2=polys*np.repeat([vv],order+1,axis=0)
    A=np.append(A1,A2,axis=0)
    rhs=np.append(rhs1,rhs2)
    lhs=np.dot(A,A.transpose())
    fitp=np.dot(np.linalg.inv(lhs),rhs)
    cm_fitp=fitp[:cm_order+1]
    mat_fitp=fitp[cm_order+1:]
    assert(len(mat_fitp)==(order+1))
    cm_pred=np.dot(cm_fitp,cm_polys)
    tmp=np.dot(mat_fitp,polys)
    mat_pred=np.repeat([tmp],ndet,axis=0)*mat
    pred=cm_pred+mat_pred
    return pred,cm_fitp,mat_fitp,polys
