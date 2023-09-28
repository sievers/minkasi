import ctypes
import sysconfig
from pathlib import Path

__all__ = ["libminkasi", "libmkfftw"]

suffix = sysconfig.get_config_var("EXT_SUFFIX")

_libminkasi_path = Path(__file__).parent.joinpath("_libminkasi" + suffix)
libminkasi = ctypes.cdll.LoadLibrary(_libminkasi_path.as_posix())

_libmkfftw_path = Path(__file__).parent.joinpath("_libmkfftw" + suffix)
libmkfftw = ctypes.cdll.LoadLibrary(_libmkfftw_path.as_posix())
