
def fit_linear_ps_uncorr(dat,vecs,tol=1e-3,guess=None,max_iter=15):
    if guess is None:
        lhs=np.dot(vecs,vecs.transpose())
        rhs=np.dot(vecs,dat**2)
        guess=np.dot(np.linalg.inv(lhs),rhs) 
        guess=0.5*guess #scale down since we're less likely to run into convergence issues if we start low
        #print guess
    fitp=guess.copy()
    converged=False
    npp=len(fitp)
    iter=0
    
    grad_tr=np.zeros(npp)
    grad_chi=np.zeros(npp)
    curve=np.zeros([npp,npp])
    datsqr=dat*dat
    while (converged==False):
        iter=iter+1
        C=np.dot(fitp,vecs)
        Cinv=1.0/C
        for i in range(npp):
            grad_chi[i]=0.5*np.sum(datsqr*vecs[i,:]*Cinv*Cinv)
            grad_tr[i]=-0.5*np.sum(vecs[i,:]*Cinv)
            for j in range(i,npp):
                #curve[i,j]=-0.5*np.sum(datsqr*Cinv*Cinv*Cinv*vecs[i,:]*vecs[j,:]) #data-only curvature
                #curve[i,j]=-0.5*np.sum(Cinv*Cinv*vecs[i,:]*vecs[j,:]) #Fisher curvature
                curve[i,j]=0.5*np.sum(Cinv*Cinv*vecs[i,:]*vecs[j,:])-np.sum(datsqr*Cinv*Cinv*Cinv*vecs[i,:]*vecs[j,:]) #exact
                curve[j,i]=curve[i,j]
        grad=grad_chi+grad_tr
        curve_inv=np.linalg.inv(curve)
        errs=np.diag(curve_inv)
        dp=np.dot(grad,curve_inv)
        fitp=fitp-dp
        frac_shift=dp/errs
        #print dp,errs,frac_shift
        if np.max(np.abs(frac_shift))<tol:
            print('successful convergence after ',iter,' iterations with error estimate ',np.max(np.abs(frac_shift)))
            converged=True
            print(C[0],C[-1])
        if iter==max_iter:
            print('not converging after ',iter,' iterations in fit_linear_ps_uncorr with current convergence parameter ',np.max(np.abs(frac_shift)))
            converged=True
            
    return fitp

def get_curve_deriv_powspec(fitp,nu_scale,lognu,datsqr,vecs):
    vec=nu_scale**fitp[2]
    C=fitp[0]+fitp[1]*vec
    Cinv=1.0/C
    vecs[1,:]=vec
    vecs[2,:]=fitp[1]*lognu*vec
    grad_chi=0.5*np.dot(vecs,datsqr*Cinv*Cinv)
    grad_tr=-0.5*np.dot(vecs,Cinv)
    grad=grad_chi+grad_tr
    np=len(grad_chi)
    curve=np.zeros([np,np])
    for i in range(np):
        for j in range(i,np):
            curve[i,j]=0.5*np.sum(Cinv*Cinv*vecs[i,:]*vecs[j,:])-np.sum(datsqr*Cinv*Cinv*Cinv*vecs[i,:]*vecs[j,:])
            curve[j,i]=curve[i,j]
    like=-0.5*sum(datsqr*Cinv)-0.5*sum(np.log(C))
    return like,grad,curve,C

def fit_ts_ps(dat,dt=1.0,ind=-1.5,nu_min=0.0,nu_max=np.inf,scale_fac=1.0,tol=0.01):

    datft=mkfftw.fft_r2r(dat)
    n=len(datft)

    dnu=0.5/(len(dat)*dt) #coefficient should reflect the type of fft you did...
    nu=dnu*np.arange(n)
    isgood=(nu>nu_min)&(nu<nu_max)
    datft=datft[isgood]
    nu=nu[isgood]
    n=len(nu)
    vecs=np.zeros([2,n])
    vecs[0,:]=1.0 #white noise
    vecs[1,:]=nu**ind
    guess=fit_linear_ps_uncorr(datft,vecs)
    pred=np.dot(guess,vecs)
    #pred=guess[0]*vecs[0]+guess[1]*vecs[1]
    #return pred

    rat=vecs[1,:]*guess[1]/(vecs[0,:]*guess[0])
    #print 'rat lims are ',rat.max(),rat.min()
    my_ind=np.max(np.where(rat>1)[0])
    nu_ref=np.sqrt(nu[my_ind]*nu[0]) #WAG as to a sensible frequency pivot point

    #nu_ref=0.2*nu[my_ind] #WAG as to a sensible frequency pivot point
    #print 'knee is roughly at ',nu[my_ind],nu_ref

    #model = guess[1]*nu^ind+guess[0]
    #      = guess[1]*(nu/nu_ref*nu_ref)^ind+guess[0]
    #      = guess[1]*(nu_ref)^in*(nu/nu_ref)^ind+guess[0]

    nu_scale=nu/nu_ref
    guess_scale=guess.copy()
    guess_scale[1]=guess[1]*(nu_ref**ind)
    #print 'guess is ',guess
    #print 'guess_scale is ',guess_scale
    C_scale=guess_scale[0]+guess_scale[1]*(nu_scale**ind)
    

    fitp=np.zeros(3)
    fitp[0:2]=guess_scale
    fitp[2]=ind

    npp=3
    vecs=np.zeros([npp,n])
    vecs[0,:]=1.0
    lognu=np.log(nu_scale)
    curve=np.zeros([npp,npp])
    grad_chi=np.zeros(npp)
    grad_tr=np.zeros(npp)
    datsqr=datft**2
    #for robustness, start with downscaling 1/f part
    fitp[1]=0.5*fitp[1]
    like,grad,curve,C=get_curve_deriv_powspec(fitp,nu_scale,lognu,datsqr,vecs)
    lamda=0.0
    #print 'starting likelihood is',like
    for iter in range(50):
        tmp=curve+lamda*np.diag(np.diag(curve))
        curve_inv=np.linalg.inv(tmp)
        dp=np.dot(grad,curve_inv)
        trial_fitp=fitp-dp
        errs=np.sqrt(-np.diag(curve_inv))
        frac=dp/errs
        new_like,new_grad,new_curve,C=get_curve_deriv_powspec(trial_fitp,nu_scale,lognu,datsqr,vecs)

        if (new_like>like):
        #if True:
            like=new_like
            grad=new_grad
            curve=new_curve
            fitp=trial_fitp
            lamda=update_lamda(lamda,True)
        else:
            lamda=update_lamda(lamda,False)
        if (lamda==0)&(np.max(np.abs(frac))<tol):
            converged=True
        else:
            converged=False
        if False:
            vec=nu_scale**fitp[2]
            C=fitp[0]+fitp[1]*vec
            Cinv=1.0/C
            vecs[1,:]=vec
            vecs[2,:]=fitp[1]*lognu*vec
            like=-0.5*np.sum(datsqr*Cinv)-0.5*np.sum(np.log(C))
            for i in range(np):
                grad_chi[i]=0.5*np.sum(datsqr*vecs[i,:]*Cinv*Cinv)
                grad_tr[i]=-0.5*np.sum(vecs[i,:]*Cinv)
                for j in range(i,np):
                    curve[i,j]=0.5*np.sum(Cinv*Cinv*vecs[i,:]*vecs[j,:])-np.sum(datsqr*Cinv*Cinv*Cinv*vecs[i,:]*vecs[j,:])
                    curve[j,i]=curve[i,j]
            grad=grad_chi+grad_tr
            curve_inv=np.linalg.inv(curve)
            errs=np.diag(curve_inv)
            dp=np.dot(grad,curve_inv)
            fitp=fitp-dp*scale_fac
            frac_shift=dp/errs

        #print fitp,errs,frac_shift,np.mean(np.abs(new_grad-grad))
        #print fitp,grad,frac,lamda
        if converged:
            print('converged after ',iter,' iterations')
            break



    #C=np.dot(guess,vecs)
    print('mean diff is ',np.mean(np.abs(C_scale-C)))
    #return datft,vecs,nu,C
    return fitp,datsqr,C
    
