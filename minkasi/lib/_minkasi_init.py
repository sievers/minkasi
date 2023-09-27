import ctypes
from pathlib import Path


__all__ = ['libminkasi', 'libmkfftw']


_libminkasi_path = next(iter(Path(__file__).parent.glob("_libminkasi.*.so")))
libminkasi = ctypes.cdll.LoadLibrary(_libminkasi_path.as_posix())

_libmkfftw_path = next(iter(Path(__file__).parent.glob("_libmkfftw.*.so")))
libmkfftw = ctypes.cdll.LoadLibrary(_libmkfftw_path.as_posix())
