prefix = $(HOME)/.local/lib

build:
	@mkdir -p $(prefix)
	@if [[ :${LD_LIBRARY_PATH}: != *:${prefix}:* ]] ; then\
	        echo "$(prefix) doesn't seem to be in LD_LIBRARY_PATH";\
	        echo "You probably want it add it to your .bashrc or equivalent";\
			echo "Add a line like export LD_LIBRARY_PATH=$(prefix)";\
		fi

libminkasi:
	gcc  -fopenmp -std=c99 -O3 -shared -fPIC -lm -lgomp -o $(prefix)/libminkasi.so minkasi/lib/minkasi.c

libmkfftw:
	gcc  -fopenmp -std=c99 -O3 -shared -fPIC -lfftw3f_threads -lfftw3f -lfftw3_threads -lfftw3  -lm -lgomp -o $(prefix)/libmkfftw.so minkasi/lib/mkfftw.c

all: build libminkasi libmkfftw

clean:
	rm $(prefix)/libminkasi.so $(prefix)/libmkfftw.so
