import ctypes
import sysconfig
from pathlib import Path

__all__ = ["libminkasi", "libmkfftw"]

suffix = sysconfig.get_config_var("EXT_SUFFIX")

try:
    _libminkasi_path = Path(__file__).parent.joinpath("_libminkasi" + suffix)
    libminkasi = ctypes.cdll.LoadLibrary(_libminkasi_path.as_posix())
except OSError:
    print(
        "Can't find pip installed version of libminkasi, checking for a manually compiled one"
    )
    libminkasi = ctypes.cdll.LoadLibrary("libminkasi.so")

try:
    _libmkfftw_path = Path(__file__).parent.joinpath("_libmkfftw" + suffix)
    libmkfftw = ctypes.cdll.LoadLibrary(_libmkfftw_path.as_posix())
except OSError:
    print(
        "Can't find pip installed version of libfftw, checking for a manually compiled one"
    )
    try:
        libmkfftw = ctypes.cdll.LoadLibrary("libmkfftw.so")
    except OSError:
        print(
            "Can't find any version of libfftw, if you need it install it following the instructions on the readme."
        )
