import numpy 
import ctypes
import time

mylib=ctypes.cdll.LoadLibrary("libmkfftw.so")

many_fft_r2c_1d_c=mylib.many_fft_r2c_1d
many_fft_r2c_1d_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int]


many_fftf_r2c_1d_c=mylib.many_fftf_r2c_1d
many_fftf_r2c_1d_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int]


many_fft_c2r_1d_c=mylib.many_fft_c2r_1d
many_fft_c2r_1d_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int]

many_fftf_c2r_1d_c=mylib.many_fftf_c2r_1d
many_fftf_c2r_1d_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int]

fft_r2r_1d_c=mylib.fft_r2r_1d
fft_r2r_1d_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int]

many_fft_r2r_1d_c=mylib.many_fft_r2r_1d
many_fft_r2r_1d_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int]

many_fftf_r2r_1d_c=mylib.many_fftf_r2r_1d
many_fftf_r2r_1d_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int]

fft_r2c_n_c=mylib.fft_r2c_n
fft_r2c_n_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p]

fft_c2r_n_c=mylib.fft_c2r_n
fft_c2r_n_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p]


fft_r2c_3d_c=mylib.fft_r2c_3d
fft_r2c_3d_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]

fft_c2r_3d_c=mylib.fft_c2r_3d
fft_c2r_3d_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]




set_threaded_c=mylib.set_threaded
set_threaded_c.argtypes=[ctypes.c_int]

read_wisdom_c=mylib.read_wisdom
read_wisdom_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p]

write_wisdom_c=mylib.write_wisdom
write_wisdom_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p]


def set_threaded(n=-1):
    set_threaded_c(n)

def rfftn(dat):
    myshape=dat.shape
    myshape=numpy.asarray(myshape,dtype='int32')
    myshape2=myshape.copy()
    myshape2[-1]=(myshape2[-1]//2+1)
    datft=numpy.zeros(myshape2,dtype='complex')
    fft_r2c_n_c(dat.ctypes.data,datft.ctypes.data,len(myshape),myshape.ctypes.data)
    return datft

def irfftn(datft,iseven=True,preserve_input=True):
    #the c2r transforms destroy input.  if you want to keep the input
    #around, then we need to copy the incoming data.
    if preserve_input:
        datft=datft.copy()
    myshape=datft.shape
    myshape=numpy.asarray(myshape,dtype='int32')
    myshape2=myshape.copy()
    if iseven:
        myshape2[-1]=2*(myshape2[-1]-1)
    else:
        myshape2[-1]=2*myshape2[-1]-1
    #print(myshape2)
    dat=numpy.empty(myshape2,dtype='float64')
    fft_c2r_n_c(datft.ctypes.data,dat.ctypes.data,len(myshape2),myshape2.ctypes.data)
    return dat


def fft_r2c_3d(dat):
    myshape=dat.shape
    assert(len(myshape)==3)
    myshape=numpy.asarray(myshape,dtype='int')
    myshape2=myshape.copy()
    myshape2[-1]=(myshape2[-1]//2+1)
    datft=numpy.zeros(myshape2,dtype='complex')
    fft_r2c_3d_c(dat.ctypes.data,datft.ctypes.data,myshape.ctypes.data)
    return datft

def fft_c2r_3d(datft,iseven=True,preserve_input=True):
    #the c2r transforms destroy input.  if you want to keep the input
    #around, then we need to copy the incoming data.
    if preserve_input:
        datft=datft.copy()
    myshape=datft.shape
    assert(len(myshape)==3)
    myshape=numpy.asarray(myshape,dtype='int')
    myshape2=myshape.copy()
    if iseven:
        myshape2[-1]=2*(myshape2[-1]-1)
    else:
        myshape2[-1]=2*myshape2[-1]-1
    #print(myshape2)
    dat=numpy.empty(myshape2,dtype='float64')
    fft_c2r_3d_c(datft.ctypes.data,dat.ctypes.data,myshape2.ctypes.data)
    return dat


def fft_r2c(dat):
    ndat=dat.shape[1]
    ntrans=dat.shape[0]
    
    if dat.dtype==numpy.dtype('float64'):
        #datft=numpy.zeros(dat.shape,dtype=complex)
        datft=numpy.empty(dat.shape,dtype=complex)
        many_fft_r2c_1d_c(dat.ctypes.data,datft.ctypes.data,ntrans,ndat,ndat,ndat)
    else:
        assert(dat.dtype==numpy.dtype('float32'))
        datft=numpy.empty(dat.shape,dtype='complex64')
        many_fftf_r2c_1d_c(dat.ctypes.data,datft.ctypes.data,ntrans,ndat,ndat,ndat)
    return datft


def fft_c2r(datft):
    ndat=datft.shape[1]
    ntrans=datft.shape[0]
    if datft.dtype==numpy.dtype('complex128'):
        dat=numpy.zeros(datft.shape)
        many_fft_c2r_1d_c(datft.ctypes.data,dat.ctypes.data,ntrans,ndat,ndat,ndat)
        dat=dat/ndat
    else:
        assert(datft.dtype==numpy.dtype('complex64'))
        dat=numpy.zeros(datft.shape,dtype='float32')
        many_fftf_c2r_1d_c(datft.ctypes.data,dat.ctypes.data,ntrans,ndat,ndat,ndat)
        dat=dat/numpy.float32(ndat)
    return dat


def fft_r2r_1d(dat,kind=1):
    nn=dat.size
    trans=numpy.zeros(nn)
    fft_r2r_1d_c(dat.ctypes.data,trans.ctypes.data,nn,kind)
    return trans

def fft_r2r(dat,trans=None,kind=1):
    if len(dat.shape)==1:
        return fft_r2r_1d(dat,kind)
    ntrans=dat.shape[0]
    n=dat.shape[1]
    #trans=numpy.zeros([ntrans,n],dtype=type(dat[0,0]))
    if trans is None:
        trans=numpy.empty([ntrans,n],dtype=type(dat[0,0]))
    

    if type(dat[0,0])==numpy.dtype('float32'):
        #print 'first two element in python are ',dat[0,0],dat[0,1]
        many_fftf_r2r_1d_c(dat.ctypes.data,trans.ctypes.data,n,kind,ntrans)
    else:
        many_fft_r2r_1d_c(dat.ctypes.data,trans.ctypes.data,n,kind,ntrans)
    return trans


def read_wisdom(double_file='.fftw_wisdom',single_file='.fftwf_wisdom'):

    df=numpy.zeros(len(double_file)+1,dtype='int8')
    df[0:-1]=[ord(c) for c in double_file]

    sf=numpy.zeros(len(single_file)+1,dtype='int8')
    sf[0:-1]=[ord(c) for c in single_file]

    read_wisdom_c(df.ctypes.data,sf.ctypes.data)

def write_wisdom(double_file='.fftw_wisdom',single_file='.fftwf_wisdom'):

    df=numpy.zeros(len(double_file)+1,dtype='int8')
    df[0:-1]=[ord(c) for c in double_file]

    sf=numpy.zeros(len(single_file)+1,dtype='int8')
    sf[0:-1]=[ord(c) for c in single_file]

    write_wisdom_c(df.ctypes.data,sf.ctypes.data)
