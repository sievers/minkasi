#include <stdio.h>
#include <complex.h>
#include <math.h>
#include <fftw3.h>
#include <omp.h>

#define PYFFTW_FLAG FFTW_ESTIMATE

//gcc-4.9 -I/Users/sievers/local/include -fopenmp -std=c99 -O3 -shared -fPIC -o libpyfftw.so pyfftw.c -L/Users/sievers/local/lib -lfftw3f_threads -lfftw3f -lfftw3_threads -lfftw3  -lm -lgomp


void set_threaded()
{
  int nthread;
  #pragma omp parallel
  #pragma omp single
  nthread=omp_get_num_threads();

  printf("Setting FFTW to have %d threads.\n",nthread);
  fftwf_plan_with_nthreads(nthread);
  fftw_plan_with_nthreads(nthread);

}

/*--------------------------------------------------------------------------------*/

void many_fft_r2c_1d(double *dat, fftw_complex *datft, int ntrans, int ndata, int rlen, int clen)
{  
  fftw_plan plan=fftw_plan_many_dft_r2c(1,&ndata,ntrans,dat,&ndata,1,rlen,datft,&ndata,1,clen,PYFFTW_FLAG);
  fftw_execute(plan);
  fftw_destroy_plan(plan);
}
/*--------------------------------------------------------------------------------*/

void many_fftf_r2c_1d(float *dat, fftwf_complex *datft, int ntrans, int ndata, int rlen, int clen)
{  
  fftwf_plan plan=fftwf_plan_many_dft_r2c(1,&ndata,ntrans,dat,&ndata,1,rlen,datft,&ndata,1,clen,PYFFTW_FLAG);
  fftwf_execute(plan);
  fftwf_destroy_plan(plan);
}

/*--------------------------------------------------------------------------------*/

void many_fft_c2r_1d(fftw_complex *datft, double *dat,int ntrans, int ndata, int rlen, int clen)
{  
  fftw_plan plan=fftw_plan_many_dft_c2r(1,&ndata,ntrans,datft,&ndata,1,rlen,dat,&ndata,1,clen,PYFFTW_FLAG);
  fftw_execute(plan);
  fftw_destroy_plan(plan);
}


/*--------------------------------------------------------------------------------*/

void many_fftf_c2r_1d(fftwf_complex *datft, float *dat,int ntrans, int ndata, int rlen, int clen)
{  
  fftwf_plan plan=fftwf_plan_many_dft_c2r(1,&ndata,ntrans,datft,&ndata,1,rlen,dat,&ndata,1,clen,PYFFTW_FLAG);
  fftwf_execute(plan);
  fftwf_destroy_plan(plan);
}

/*--------------------------------------------------------------------------------*/
void fft_r2r_1d(double *dat, double *trans, int n, int type)
{
  fftw_r2r_kind flag=FFTW_REDFT00;
  switch (type) {
  case 1:
    flag=FFTW_REDFT00;
    break;
  case 2:
    flag=FFTW_REDFT10;
    break;
  case 3:
    flag=FFTW_REDFT01;
    break;
  case 4:
    flag=FFTW_REDFT11;
    break;

  case 11:
    flag=FFTW_RODFT00;
    break;
  case 12:
    flag=FFTW_RODFT10;
    break;
  case 13:
    flag=FFTW_RODFT01;
    break;
  case 14:
    flag=FFTW_RODFT11;
    break;
    
  }
  //for (int i=0;i<5;i++)
  //printf("dat %d is %14.7e\n",i,dat[i]);

  fftw_plan plan=fftw_plan_r2r_1d(n,dat,trans,flag,FFTW_ESTIMATE);
  fftw_execute(plan);
  fftw_destroy_plan(plan);
  //for (int i=0;i<5;i++)
  //printf("trans %d is %g\n",i,trans[i]);


}

//fftw_plan fftw_plan_r2r_1d(int n, double *in, double *out,
//                           fftw_r2r_kind kind, unsigned flags);


/*--------------------------------------------------------------------------------*/
void many_fft_r2r_1d(double *dat, double *trans, int n, int type, int ntrans)
{
  fftw_r2r_kind flag=FFTW_REDFT00;
  switch (type) {
  case 1:
    flag=FFTW_REDFT00;
    break;
  case 2:
    flag=FFTW_REDFT10;
    break;
  case 3:
    flag=FFTW_REDFT01;
    break;
  case 4:
    flag=FFTW_REDFT11;
    break;

  case 11:
    flag=FFTW_RODFT00;
    break;
  case 12:
    flag=FFTW_RODFT10;
    break;
  case 13:
    flag=FFTW_RODFT01;
    break;
  case 14:
    flag=FFTW_RODFT11;
    break;
    
  }
  flag=FFTW_REDFT00;
  fftw_plan plan=fftw_plan_r2r_1d(n,dat,trans,flag,FFTW_ESTIMATE);
#pragma omp parallel for
  for (int i=0;i<ntrans;i++)
    fftw_execute_r2r(plan,dat+i*n,trans+i*n);
  fftw_destroy_plan(plan);

}

/*--------------------------------------------------------------------------------*/
void many_fftf_r2r_1d(float *dat, float *trans, int n, int type, int ntrans)
{
  fftw_r2r_kind flag=FFTW_REDFT00;
  switch (type) {
  case 1:
    flag=FFTW_REDFT00;
    break;
  case 2:
    flag=FFTW_REDFT10;
    break;
  case 3:
    flag=FFTW_REDFT01;
    break;
  case 4:
    flag=FFTW_REDFT11;
    break;

  case 11:
    flag=FFTW_RODFT00;
    break;
  case 12:
    flag=FFTW_RODFT10;
    break;
  case 13:
    flag=FFTW_RODFT01;
    break;
  case 14:
    flag=FFTW_RODFT11;
    break;
    
  }
  flag=FFTW_REDFT00;
  fftwf_plan plan=fftwf_plan_r2r_1d(n,dat,trans,flag,FFTW_ESTIMATE);
  //printf("first two elements are %12.4g %12.4g\n",dat[0],dat[1]);
#pragma omp parallel for
  for (int i=0;i<ntrans;i++)
    fftwf_execute_r2r(plan,dat+i*n,trans+i*n);
  fftwf_destroy_plan(plan);

}


/*--------------------------------------------------------------------------------*/
void read_wisdom(char *double_file, char *single_file)
{
  printf("files are: .%s. and .%s.\n",double_file,single_file);
  int dd=fftw_import_wisdom_from_filename(double_file);
  int ss=fftwf_import_wisdom_from_filename(single_file);
  printf("return values are %d %d\n",dd,ss);
}

/*--------------------------------------------------------------------------------*/
void write_wisdom(char *double_file, char *single_file)
{
  printf("files are: .%s. and .%s.\n",double_file,single_file);
  int dd=fftw_export_wisdom_to_filename(double_file);
  int ss=fftwf_export_wisdom_to_filename(single_file);
  printf("return values are %d %d\n",dd,ss);
}
