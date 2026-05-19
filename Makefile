# ---------------------------------------------------------------------------
# If a conda env is active, use it; otherwise fall back to system paths.
# All variables can be overridden from the command line, e.g.:
#   make all CC=gcc prefix=/your/lib/path
# ---------------------------------------------------------------------------

CONDA_PREFIX ?=

ifdef CONDA_PREFIX
  prefix  ?= $(CONDA_PREFIX)/lib
  INCDIR  ?= $(CONDA_PREFIX)/include
  LIBDIR  ?= $(CONDA_PREFIX)/lib
  CC      ?= $(CONDA_PREFIX)/bin/clang
else
  prefix  ?= /usr/local/lib
  INCDIR  ?= /usr/local/include
  LIBDIR  ?= /usr/local/lib
  CC      ?= clang
endif

CFLAGS   := -fopenmp -std=c99 -O3 -shared -fPIC
INCLUDES := -I$(INCDIR)
LDFLAGS  := -L$(LIBDIR)

# ---------------------------------------------------------------------------

build:
	@mkdir -p $(prefix)
	@if [[ :$${LD_LIBRARY_PATH}: != *:$(prefix):* ]]; then \
	  echo "$(prefix) doesn't seem to be in LD_LIBRARY_PATH"; \
	  echo "You probably want to add it to your .bashrc or equivalent"; \
	  echo "Add a line like: export LD_LIBRARY_PATH=$(prefix)"; \
	fi

libminkasi:
	$(CC) $(INCLUDES) $(LDFLAGS) $(CFLAGS) -lm -lgomp \
	  -o $(prefix)/libminkasi.so minkasi/lib/minkasi.c

libmkfftw:
	$(CC) $(INCLUDES) $(LDFLAGS) $(CFLAGS) \
	  -lfftw3f_threads -lfftw3f -lfftw3_threads -lfftw3 -lm -lgomp \
	  -o $(prefix)/libmkfftw.so minkasi/lib/mkfftw.c

all: build libminkasi libmkfftw

clean:
	rm $(prefix)/libminkasi.so $(prefix)/libmkfftw.so