import ctypes
import os

from ._minkasi_init import libmkfftw as mylib


many_fft_r2c_1d_c = mylib.many_fft_r2c_1d
many_fft_r2c_1d_c.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]


many_fftf_r2c_1d_c = mylib.many_fftf_r2c_1d
many_fftf_r2c_1d_c.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]


many_fft_c2r_1d_c = mylib.many_fft_c2r_1d
many_fft_c2r_1d_c.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]

many_fftf_c2r_1d_c = mylib.many_fftf_c2r_1d
many_fftf_c2r_1d_c.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]

fft_r2r_1d_c = mylib.fft_r2r_1d
fft_r2r_1d_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]

many_fft_r2r_1d_c = mylib.many_fft_r2r_1d
many_fft_r2r_1d_c.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]

many_fftf_r2r_1d_c = mylib.many_fftf_r2r_1d
many_fftf_r2r_1d_c.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]

fft_r2c_n_c = mylib.fft_r2c_n
fft_r2c_n_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]

fft_c2r_n_c = mylib.fft_c2r_n
fft_c2r_n_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]


fft_r2c_3d_c = mylib.fft_r2c_3d
fft_r2c_3d_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

fft_c2r_3d_c = mylib.fft_c2r_3d
fft_c2r_3d_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]


set_threaded_c = mylib.set_threaded
set_threaded_c.argtypes = [ctypes.c_int]

read_wisdom_c = mylib.read_wisdom
read_wisdom_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

write_wisdom_c = mylib.write_wisdom
write_wisdom_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
