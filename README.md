# Minkasi

## Installation

### The Easy Way

```
git clone https://github.com/sievers/minkasi
cd minkasi
pip install .
```

Hopefully this eventually gets packaged and gets even easier.

### C Library Installation
By default the pip install will put a copy of the compiled `.so` files
somewhere that the python code can find it if can't find them during install time.

If you just want to install and use `minkasi` just from the library this should be fine.

If you want to build the libraries on their own run `make all` from the root of the repo.
This will put the compiled files in `~/.local/lib/`.
If you want to install them somewhere else `prefix=WHEREVER_YOU_WANT make -e all`.

To make your system aware of these files use `ldconfig` if you have root.
Otherwise add a line like:
```
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:~/.local/lib
``` 
to your `.bashrc` or equivalent.

### For Development
If you are actively developing `minkasi` and want an live copy installed do:
```
pip install -e .
```

Note that this will not make changes to the C libraries automatically propagate.
If you will be editing those files it is recommended to follow the instructions in the **C Library Installation** first and rerun the build process whenever you make a change. 
