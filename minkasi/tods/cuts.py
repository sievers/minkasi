
def segs_from_vec(vec,pad=True):
    """ segs_from_vec(vec,pad=True)
    return the starting/stopping points of regions marked False in vec.  For use in e.g. generating
    cuts from a vector/array.  If pad is False, assume vector is already True-padded"""
    #insert input vector into a True-padded vector do make reasoning about starting/stopping points
    #of False regions easier.
    if pad:
        vv=np.ones(len(vec)+2,dtype='bool')
        vv[1:-1]=vec
    else:
        if vec.dtype=='bool':
            vv=vec
        else:
            vv=np.ones(len(vec),dtype='bool')
            vv[:]=vec
    if vv.min()==True:
        nseg=0
        istart=[]
        istop=[]
    else:
        inds=np.where(np.diff(vv))[0]
        assert(len(inds)%2==0)
        nseg=len(inds)//2
        istart=[]
        istop=[]
        for i in range(nseg):
            istart.append(inds[2*i])
            istop.append(inds[2*i+1])
    return nseg,istart,istop


class Cuts:
    def __init__(self,tod,do_add=True):
        #if class(tod)==Cuts: #for use in copy
        if isinstance(tod,Cuts):
            self.map=tod.map.copy()
            self.bad_inds=tod.bad_inds.copy()
            self.namps=tod.nsamp
            self.do_add=tod.do_add
            return
        bad_inds=np.where(tod.info['bad_samples'])
        #dims=tod.info['dat_calib'].shape
        dims=tod.get_data_dims()
        bad_inds=np.ravel_multi_index(bad_inds,dims)
        self.nsamp=len(bad_inds)
        self.inds=bad_inds
        self.map=np.zeros(self.nsamp)
        self.do_add=do_add
    def clear(self):
        self.map[:]=0
    def axpy(self,cuts,a):
        self.map[:]=self.map[:]+a*cuts.map[:]
    def map2tod(self,tod,dat):
        dd=np.ravel(dat)
        if self.do_add:
            dd[self.inds]=self.map
        else:
            dd[self.inds]+=self.map
    def tod2map(self,tod,dat):
        dd=np.ravel(dat)
        self.map[:]=dd[self.inds]
    def dot(self,cuts):
        tot=np.dot(self.map,cuts.map)
        return tot
    def copy(self):
        return Cuts(self)
class CutsCompact:
    def __init__(self,tod):
        if isinstance(tod,CutsCompact):
            self.ndet=tod.ndet
            self.nseg=tod.nseg
            self.istart=tod.istart
            self.istop=tod.istop
        else:
            #ndet=tod.info['dat_calib'].shape[0]
            ndet=tod.get_ndet()
            self.ndet=ndet
            self.nseg=np.zeros(ndet,dtype='int')
            self.istart=[None]*ndet
            self.istop=[None]*ndet
            #self.imax=tod.info['dat_calib'].shape[1]
            self.imax=tod.get_ndata()

        self.imap=None
        self.map=None
        
    def copy(self,deep=True):
        copy=CutsCompact(self)
        if deep:
            if not(self.imap is None):
                copy.imap=self.imap.copy()
            if not(self.map is None):
                copy.map=self.map.copy()
        else:
            copy.imap=self.imap
            copy.map=self.map
        return copy
    def add_cut(self,det,istart,istop):
        if istart>=self.imax:
            #this is asking to add a cut past the end of the data.
            return
        if istop>self.imax: #don't have a cut run past the end of the timestream
            istop=self.imax
            
        self.nseg[det]=self.nseg[det]+1
        if self.istart[det] is None:
            self.istart[det]=[istart]
        else:
            self.istart[det].append(istart)
        if self.istop[det] is None:
            self.istop[det]=[istop]
        else:
            self.istop[det].append(istop)
    def get_imap(self):
        ncut=0
        for det in range(self.ndet):
            for i in range(self.nseg[det]):
                ncut=ncut+(self.istop[det][i]-self.istart[det][i])
        print('ncut is ' + repr(ncut))
        self.imap=np.zeros(ncut,dtype='int64')
        icur=0
        for det in range(self.ndet):
            for i in range(self.nseg[det]):
                istart=det*self.imax+self.istart[det][i]
                istop=det*self.imax+self.istop[det][i]
                nn=istop-istart
                self.imap[icur:icur+nn]=np.arange(istart,istop)
                icur=icur+nn
        self.map=np.zeros(len(self.imap))
    def cuts_from_array(self,cutmat):
        for det in range(cutmat.shape[0]):
            nseg,istart,istop=segs_from_vec(cutmat[det,:])
            self.nseg[det]=nseg
            self.istart[det]=istart
            self.istop[det]=istop
    def merge_cuts(self):
        tmp=np.ones(self.imax+2,dtype='bool')
        for det in range(self.ndet):
            if self.nseg[det]>1:  #if we only have one segment, don't have to worry about strange overlaps
                tmp[:]=True
                for i in range(self.nseg[det]):
                    tmp[(self.istart[det][i]+1):(self.istop[det][i]+1)]=False
                nseg,istart,istop=segs_from_vec(tmp,pad=False)
                self.nseg[det]=nseg
                self.istart[det]=istart
                self.istop[det]=istop
    
    def tod2map(self,tod,mat=None,do_add=True,do_omp=False):
        if mat is None:
            #mat=tod.info['dat_calib']
            mat=tod.get_data()
        tod2cuts_c(self.map.ctypes.data,mat.ctypes.data,self.imap.ctypes.data,len(self.imap),do_add)

    def map2tod(self,tod,mat=None,do_add=True,do_omp=False):
        if mat is None:
            #mat=tod.info['dat_calib']
            mat=tod.get_data()
        #print('first element is ' + repr(mat[0,self.imap[0]]))
        cuts2tod_c(mat.ctypes.data,self.map.ctypes.data,self.imap.ctypes.data,len(self.imap),do_add)
        #print('first element is now ' + repr(mat[0,self.imap[0]]))
        #return mat
    def clear(self):
        if not(self.map is None):
            self.map[:]=0
    def dot(self,other=None):
        if self.map is None:
            return None
        if other is None:
            return np.dot(self.map,self.map)
        else:
            if other.map is None:
                return None
            return np.dot(self.map,other.map)
    def axpy(self,common,a):
        self.map=self.map+a*common.map
    def write(self,fname=None):
        pass
    def apply_prior(self,x,Ax):
        Ax.map=Ax.map+self.map*x.map
    def __mul__(self,to_mul):
        tt=self.copy()
        tt.map=self.map*to_mul.map
        return tt                
            
