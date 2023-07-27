from .minkasi import set_nthread_c, get_nthread_c
try:
    import mpi4py.rc
    mpi4py.rc.threads = False
    from mpi4py import MPI
    print('mpi4py imported')
    comm=MPI.COMM_WORLD
    myrank = comm.Get_rank()
    nproc=comm.Get_size()
    print('nproc:, ', nproc)
    if nproc>1:
        have_mpi=True
    else:
        have_mpi=False
except:
    MPI = None
    have_mpi=False
    myrank=0
    nproc=1

def report_mpi():
    if have_mpi:
        print('myrank is ',myrank,' out of ',nproc)
    else:
        print('mpi not found')

def barrier():
    if have_mpi:
        comm.barrier()
    else:
        pass



def set_nthread(nthread):
    set_nthread_c(nthread)

def get_nthread():
    nthread=np.zeros([1,1],dtype='int32')
    get_nthread_c(nthread.ctypes.data)
    return nthread[0,0]

