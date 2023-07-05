
def tod2map_simple(map,dat,ipix):
    ndet=dat.shape[0]
    ndata=dat.shape[1]
    if not(ipix.dtype=='int32'):
        print("Warning - ipix is not int32 in tod2map_simple.  this is likely to produce garbage results.")
    tod2map_simple_c(map.ctypes.data,dat.ctypes.data,ndet,ndata,ipix.ctypes.data)

def tod2map_everyone(map,dat,ipix,edges):
    assert(len(edges)==get_nthread()+1)
    tod2map_everyone_c(map.ctypes.data,dat.ctypes.data,dat.shape[0],dat.shape[1],ipix.ctypes.data,map.size,edges.ctypes.data,len(edges))

def tod2map_omp(map,dat,ipix,atomic=False):
    ndet=dat.shape[0]
    ndata=dat.shape[1]
    if not(ipix.dtype=='int32'):
        print("Warning - ipix is not int32 in tod2map_omp.  this is likely to produce garbage results.")
    if atomic:
        tod2map_atomic_c(map.ctypes.data,dat.ctypes.data,ndet,ndata,ipix.ctypes.data,map.size)
    else:
        tod2map_omp_c(map.ctypes.data,dat.ctypes.data,ndet,ndata,ipix.ctypes.data,map.size)

def tod2map_cached(map,dat,ipix):
    ndet=dat.shape[0]
    ndata=dat.shape[1]
    if not(ipix.dtype=='int32'):
        print("Warning - ipix is not int32 in tod2map_cached.  this is likely to produce garbage results.")
    tod2map_cached_c(map.ctypes.data,dat.ctypes.data,ndet,ndata,ipix.ctypes.data,map.shape[1])
    
def tod2polmap(map,dat,poltag,twogamma,ipix):
    ndet=dat.shape[0]
    ndata=dat.shape[1]
    fun=None
    if poltag=='QU':
        fun=tod2map_qu_simple_c
    if poltag=='IQU':
        fun=tod2map_iqu_simple_c
    if poltag=='QU_PRECON':
        fun=tod2map_qu_precon_simple_c
    if poltag=='IQU_PRECON':
        fun=tod2map_iqu_precon_simple_c
    if fun is None:
        print('unrecognized poltag ' + repr(poltag) + ' in tod2polmap.')
    #print('calling ' + repr(fun))
    fun(map.ctypes.data,dat.ctypes.data,twogamma.ctypes.data,ndet,ndata,ipix.ctypes.data)

@jit(nopython=True)
def tod2mapbowl(vecs, mat):
    """
    transpose of map2tod for bowling 
 
    Parameters
    ----------
    vecs: np.array(ndet, ndata, order)
        pseudo-Vandermonde matrix
    mat: np.array(ndet, ndata)
        tod data 
    """
     
    #Return tod should have shape ndet x ndata
    to_return = np.zeros((vecs.shape[0], vecs.shape[-1]))
    for i in range(vecs.shape[0]):
         to_return[i] = np.dot(vecs[i,...].T, mat[i,...])
  
