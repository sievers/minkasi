"""
Python wrapper for the libminkasi C library.
"""
import ctypes
import os

from ._minkasi_init import libminkasi as mylib

# ------------------------------ tod2map functions --------------------------- #
tod2map_simple_c = mylib.tod2map_simple
tod2map_simple_c.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
]

tod2map_atomic_c = mylib.tod2map_atomic
tod2map_atomic_c.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
]

tod2map_everyone_c = mylib.tod2map_everyone
tod2map_everyone_c.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
]

tod2map_omp_c = mylib.tod2map_omp
tod2map_omp_c.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
]

tod2map_cached_c = mylib.tod2map_cached
tod2map_cached_c.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
]

tod2map_iqu_simple_c = mylib.tod2map_iqu_simple
tod2map_iqu_simple_c.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
]

tod2map_qu_simple_c = mylib.tod2map_qu_simple
tod2map_qu_simple_c.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
]

tod2map_iqu_precon_simple_c = mylib.tod2map_iqu_precon_simple
tod2map_iqu_precon_simple_c.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
]

tod2map_qu_precon_simple_c = mylib.tod2map_qu_precon_simple
tod2map_qu_precon_simple_c.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
]

tod2cuts_c = mylib.tod2cuts
tod2cuts_c.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
]
# ---------------------------------------------------------------------------- #

# ---------------------------- map2tod functions ------------------------------ #
map2tod_simple_c = mylib.map2tod_simple
map2tod_simple_c.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
]

map2tod_omp_c = mylib.map2tod_omp
map2tod_omp_c.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
]

map2tod_iqu_omp_c = mylib.map2tod_iqu_omp
map2tod_iqu_omp_c.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
]

map2tod_qu_omp_c = mylib.map2tod_qu_omp
map2tod_qu_omp_c.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
]

cuts2tod_c = mylib.cuts2tod
cuts2tod_c.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
]
# ---------------------------------------------------------------------------- #

# --------------------------------- threads ---------------------------------- #
set_nthread_c = mylib.set_nthread
set_nthread_c.argtypes = [ctypes.c_int]

get_nthread_c = mylib.get_nthread
get_nthread_c.argtypes = [ctypes.c_void_p]
# ---------------------------------------------------------------------------- #

# --------------------------------- models ----------------------------------- #
fill_isobeta_c = mylib.fill_isobeta
fill_isobeta_c.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]

fill_isobeta_derivs_c = mylib.fill_isobeta_derivs
fill_isobeta_derivs_c.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]

fill_gauss_derivs_c = mylib.fill_gauss_derivs
fill_gauss_derivs_c.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]

fill_gauss_src_c = mylib.fill_gauss_src
fill_gauss_src_c.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]
# ---------------------------------------------------------------------------- #

# ---------------------------------- other ----------------------------------- #
scan_map_c = mylib.scan_map
scan_map_c.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]

outer_c = mylib.outer_block
outer_c.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
# ---------------------------------------------------------------------------- #
