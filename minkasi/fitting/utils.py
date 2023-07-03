
def update_lamda(lamda,success):
    if success:
        if lamda<0.2:
            return 0
        else:
            return lamda/np.sqrt(2)
    else:
        if lamda==0.0:
            return 1.0
        else:
            return 2.0*lamda
        
def invsafe(mat,thresh=1e-14):
    u,s,v=np.linalg.svd(mat,0)
    ii=np.abs(s)<thresh*s.max()
    #print ii
    s_inv=1/s
    s_inv[ii]=0
    tmp=np.dot(np.diag(s_inv),u.transpose())
    return np.dot(v.transpose(),tmp)

def invscale(mat,do_invsafe=False):
    vec=1/np.sqrt(abs(np.diag(mat)))
    vec[np.where(vec == np.inf)[0]] = 1e-10
    mm=np.outer(vec,vec)
    mat=mm*mat
    #ee,vv=np.linalg.eig(mat)
    #print 'rcond is ',ee.max()/ee.min(),vv[:,np.argmin(ee)]
    if do_invsafe:
        return mm*invsafe(mat)
    else:
        try:
            return mm*np.linalg.inv(mat)
        except:
            return mm*np.linalg.pinv(mat) 

