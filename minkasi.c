#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>

//gcc-4.9 -fopenmp -std=c99 -O3 -shared -fPIC -o libminkasi.so minkasi.c  -lm -lgomp     



void tod2map_simple(double *map, double *dat, int ndet, int ndata, int *pix)
{
  long nn=ndet*ndata;
  for (long i=0;i<nn;i++)
    map[pix[i]]+=dat[i];
}

/*--------------------------------------------------------------------------------*/

void tod2map_omp(double *map, double *dat, int ndet, int ndata, int *pix, int npix)
{

  long nn=ndet*ndata;
#pragma omp parallel 
  {
    double *mymap=(double *)calloc(npix,sizeof(double));
    #pragma omp for
    for (long i=0;i<nn;i++)
      mymap[pix[i]]+=dat[i];
    #pragma omp critical
    for (long i=0;i<npix;i++)
      map[i]+=mymap[i];
    free(mymap);
  }
}

/*--------------------------------------------------------------------------------*/
//project a map into a tod, adding or replacing the map contents.  Add map to tod if do_add is true
void map2tod_simple(double *dat, double *map, int ndet, int ndata, int *pix, int do_add)
{
  long nn=ndet*ndata;
  if (do_add)
    for (long i=0;i<nn;i++)
      dat[i]+=map[pix[i]];
  else
    for (long i=0;i<nn;i++)
      dat[i]=map[pix[i]];


}

/*--------------------------------------------------------------------------------*/
//project a map into a tod using openmp, adding or replacing the map contents.  Add map to tod if do_add is true
void map2tod_omp(double *dat, double *map, int ndet, int ndata, int *pix, int do_add)
{
  long nn=ndet*ndata;
  if (do_add)
#pragma omp parallel for
    for (long i=0;i<nn;i++)
      dat[i]+=map[pix[i]];
  else
#pragma omp parallel for
    for (long i=0;i<nn;i++)
      dat[i]=map[pix[i]];


}


/*--------------------------------------------------------------------------------*/
void set_nthread(int nthread)
{
  omp_set_num_threads(nthread);
}

/*--------------------------------------------------------------------------------*/
void get_nthread(int *nthread)
{
  #pragma omp parallel
  #pragma omp single
  *nthread=omp_get_num_threads();

}
