import os
from setuptools import Extension, setup, find_packages
import subprocess

from distutils.core import setup, Extension
from collections import defaultdict
from extension_helpers import get_compiler
from extension_helpers import add_openmp_flags_if_available
from pathlib import Path


minkasi_dir = Path(__file__).parent.joinpath("minkasi/lib")


def pkgconfig(package, kw):
    flag_map = {"-I": "include_dirs", "-L": "library_dirs", "-l": "libraries"}
    cmd = "pkg-config --cflags --libs {}".format(package)
    exitcode, output = subprocess.getstatusoutput(cmd)
    if exitcode != 0:
        raise ValueError(f"Failed to run command `{cmd}`:\n\n{output}\n")
    for token in output.strip().split():
        kw.setdefault(flag_map.get(token[:2]), []).append(token[2:])
    return kw


def _add_common_ext_kwargs(cfg):
    if get_compiler() in ("unix", "mingw32"):
        cfg["extra_compile_args"].extend(
            [
                "-pedantic",
                "-Wno-newline-eof",
                "-Wno-unused-const-variable",
            ]
        )
    return cfg


def minkasi_extension():
    cfg = defaultdict(list)

    files = [
        "minkasi.c",
    ]
    cfg["sources"].extend(minkasi_dir.joinpath(x).as_posix() for x in files)

    # external dependencies
    cfg["libraries"].append("m")
    _add_common_ext_kwargs(cfg)

    return Extension("minkasi.lib._libminkasi", **cfg)


def mkfftw_extension():
    cfg = defaultdict(list)

    files = ["mkfftw.c"]
    cfg["sources"].extend(minkasi_dir.joinpath(x).as_posix() for x in files)

    # external dependencies
    cfg["libraries"].append("m")

    # add fftw3 config
    cfg = pkgconfig("fftw3", cfg)
    # this only added the non-threaded libs, and we also need the threaded
    # ones
    cfg["libraries"].extend(["fftw3_threads", "fftw3f", "fftw3f_threads"])
    _add_common_ext_kwargs(cfg)
    return Extension("minkasi.lib._libmkfftw", **cfg)


def get_extensions():
    extensions = []
    if os.environ.get("MINKASI_COMPILED", 0) == 0:
        extensions = [minkasi_extension(), mkfftw_extension()]
    for ext in extensions:
        add_openmp_flags_if_available(ext)
    return extensions


setup(
    name="minkasi",
    python_requires=">=3.7",
    # packages=["minkasi"],
    packages=find_packages(where="."),
    version="2.0.0",
    install_requires=[
        "requests",
        "importlib-metadata",
        "numpy",
        "astropy",
        "scipy",
        "typing-extensions==4.7.1",
    ],
    extras_require={
        "extras": ["qpoint", "numba", "healpy", "pyregion"],
    },
    ext_modules=get_extensions(),
)
