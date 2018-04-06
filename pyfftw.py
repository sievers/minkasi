import numpy 
import ctypes
import time

mylib=ctypes.cdll.LoadLibrary("libpyfftw.so")

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


set_threaded_c=mylib.set_threaded
set_threaded_c.argtypes=[]

read_wisdom_c=mylib.read_wisdom
read_wisdom_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p]

write_wisdom_c=mylib.write_wisdom
write_wisdom_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p]


def set_threaded():
    set_threaded_c()

def fft_r2c(dat):
    ndat=dat.shape[1]
    ntrans=dat.shape[0]
    
    if dat.dtype==numpy.dtype('float64'):
        datft=numpy.zeros(dat.shape,dtype=complex)
        many_fft_r2c_1d_c(dat.ctypes.data,datft.ctypes.data,ntrans,ndat,ndat,ndat)
    else:
        assert(dat.dtype==numpy.dtype('float32'))
        datft=numpy.zeros(dat.shape,dtype='complex64')
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


def fft_r2r_1d(dat,type=1):
    nn=dat.size
    trans=numpy.zeros(nn)
    fft_r2r_1d_c(dat.ctypes.data,trans.ctypes.data,nn,type)
    return trans

def fft_r2r(dat,type=1):
    ntrans=dat.shape[0]
    n=dat.shape[1]
    trans=numpy.zeros([ntrans,n])
    many_fft_r2r_1d_c(dat.ctypes.data,trans.ctypes.data,n,type,ntrans)
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
