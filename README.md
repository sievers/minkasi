# Minkasi

## Installation

### The Easy Way

```
git clone https://github.com/sievers/minkasi
cd minkasi
pip install .
```

Hopefully this eventually gets packaged and gets even easier.

### C Library Compilation

By default, the pip install will create a compiled C extension
`_minkasi.<arch>.so` file so the Python code can load.

By default, the compilation uses the system default compiler, and uses the
tool `pkg-config` to locate the dependencies such as the `fftw3` lib. To
specify alternative compiler or path to custom-built `fftw3`, use

```
export CC=/path/to/c/compiler
export PKG_CONFIG_PATH=/path/to/fftw/pkgconfig:${PKG_CONFIG_PATH}
pip install .
```

### For Development

If you want to build the libraries on their own run `make all` from the root of the repo.
This will put the compiled files in `~/.local/lib/`.
If you want to install them somewhere else `prefix=WHEREVER_YOU_WANT make -e all`.

To make your system aware of these files use `ldconfig` if you have root.
Otherwise add a line like:

```
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:~/.local/lib
```

to your `.bashrc` or equivalent.

If you are actively developing `minkasi` and want an live copy installed do:

```
pip install -e .
```

Note that this will not make changes to the C libraries automatically propagate.
If you will be editing those files it is recommended to follow the instructions in the **C Library Installation** first and rerun the build process whenever you make a change.

### Troubleshooting

#### Missing Libraries

If you are missing one of the libraries required for the C libraries the compilation will fail.
This is most commonly due to a missing `fftw3` install in which case the compilation will produce an error message like:

```
minkasi/mkfftw.c:4:10: fatal error: fftw3.h: No such file or directory
 #include <fftw3.h>
           ^~~~~~~~~
```

To resolve this first make sure that you have `fftw3` on your system.
If you are on a cluster you may have to load the correct module to make it available.

If you still are getting an error message like above make sure that the file its looking for is in a place where `gcc` can find it.
To do this set `CPATH` to include the directory with the header file.
Do this with:

```
export CPATH=$CPATH:PATH_TO_DIR
```

#### Linked Library Problems

Sometime `gcc` can't find the linked libraries needed for compilation.
In this case you will get an error message like:

```
ld: cannot find -lfftw3f: No such file or directory
```

To resolve this first make sure you actually have these libraries, similar to above this will
involve either installing or loading the correct modules.

If you do have these libraries somewhere on disk but still get these error messaged you need to let `gcc` know where to look for them.
To do this set `LIBRARY_PATH` to include the directories with the appropriate files (on Linux they will be `.so` files).
Do this with:

```
export LIBRARY_PATH=$LIBRARY_PATH:PATH_TO_DIR
```

### Cluster Instructions

#### Niagara

```
module load gcc fftw
```

#### NERSC

```
module load craype-CRAY_CPU_TARGET
module load cray-fftw
export CPATH=$CPATH:$FFTW_INC
export LIBRARY_PATH=$LIBRARY_PATH:$FFTW_DIR
```

Where `CRAY_CPU_TARGET` is your target CPU (ie: `x86-milan)`.
Run `module spider cray-fftw/VERSION` for more information.

## Fall 2023 Refactor
In Fall of 2023 this repo underwent an API breaking refactor.
If you have old code with a line like:
```
import minkasi
```
where you are expecting all of `minkasi` in a flat namespace you can shim it by replacing it with:
```
from minkasi import minkasi_all as minkasi
```
 Note that `minkasi_all` was implemented to make running legacy code easier,
 if you are writing new code please use the new API.

 If you need to fully revert to the pre-refactor version of `minkasi` this can be done with:
 ```
 git checkout v1.1.1
 ```
