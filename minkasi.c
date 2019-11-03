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
void tod2map_cached(double *maps, double *dat, int ndet, int ndata, int *ipix, int npix)
{
#pragma omp parallel
  {
    double *mymap=omp_get_thread_num()*npix+maps;
    long nn=ndet*ndata;
#pragma omp for
    for (int i=0;i<nn;i++)
      mymap[ipix[i]]+=dat[i];
  }
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
void tod2cuts(double *vec, double *dat, long *imap, int ncut)
{
  for (long i=0;i<ncut;i++)
    vec[i]+=dat[imap[i]];
}

/*--------------------------------------------------------------------------------*/
void cuts2tod(double *dat,double *vec, long *imap, int ncut)
{
  for (long i=0;i<ncut;i++)
    dat[imap[i]]+=vec[i];
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

/*--------------------------------------------------------------------------------*/
void fill_gauss_src(double *param, double *dx, double *dy, double *dat, int n)
{
  double x0=param[0];
  double y0=param[1];
  double sig=param[2];
  double fac=-0.5/sig/sig;
  double amp=param[3];
  double cosdec=cos(y0);
  for (int i=0;i<n;i++) {
    double delx=(x0-dx[i])*cosdec;
    double dely=y0-dy[i];
    double myarg=(delx*delx+dely*dely)*fac;
    if (myarg>-20) {
      //dat[i]+=amp*exp((delx*delx+dely*dely)*fac);
      dat[i]+=amp*exp(myarg);
    }
  }
}
/*--------------------------------------------------------------------------------*/
void fill_isobeta(double *param, double *dx, double *dy, double *dat, int n)
{
  double x0=param[0];
  double y0=param[1];
  double theta=param[2];
  double theta_inv=1.0/theta;
  theta_inv=theta_inv*theta_inv;
  double beta=param[3];
  double amp=param[4];
  double cosdec=cos(y0);
  double mypow=0.5-1.5*beta;
#pragma omp parallel for
  for (int i=0;i<n;i++) {
    double delx=(x0-dx[i])*cosdec;
    double dely=y0-dy[i];
    double rr=1+theta_inv*(delx*delx+dely*dely);
    dat[i]+=pow(rr,mypow)*amp;
  }
}
/*--------------------------------------------------------------------------------*/
void fill_gauss_derivs(double *param, double *dx, double *dy, double *dat, double *derivs, int n)
{
  double x0=param[0];
  double y0=param[1];
  double sig=param[2];
  double amp=param[3];

  double minus_2_sig_inv=-2.0/sig;
  double half_sig_minus_2=-0.5/sig/sig;

  double cosdec=cos(y0);
  double cosinv=1/cosdec;
  double sindec=sin(y0);
#pragma omp parallel for
  for (int i=0;i<n;i++) {
    double delx=(x0-dx[i])*cosdec;
    double dely=y0-dy[i];
    double arg=(delx*delx+dely*dely)*half_sig_minus_2;
    double myexp=exp(arg);
    double f=myexp*amp;
    double dfdx=f*delx*half_sig_minus_2*cosdec*2;
    double dfdy=f*(dely-delx*delx*sindec*cosinv)*half_sig_minus_2*2;
    double dfdsig=f*arg*minus_2_sig_inv;
    dat[i]=f;
    derivs[i]=dfdx;
    derivs[i+n]=dfdy;
    derivs[i+2*n]=dfdsig;
    derivs[i+3*n]=myexp;

  }
}

/*--------------------------------------------------------------------------------*/

void fill_isobeta_derivs(double *param, double *dx, double *dy, double *dat, double *derivs, int n)
{

  double x0=param[0];
  double y0=param[1];
  double theta=param[2];
  double theta_inv=1.0/theta;
  double theta_inv_sqr=theta_inv*theta_inv;
  double beta=param[3];
  double amp=param[4];
  double cosdec=cos(y0);
  double cosdec_inv=1.0/cosdec;
  double sindec=sin(y0);
  double mypow=0.5-1.5*beta;
#pragma omp parallel for
  for (int i=0;i<n;i++) {
    double delx=(x0-dx[i])*cosdec;
    double dely=y0-dy[i];
    double gg=theta_inv_sqr*(delx*delx+dely*dely);
    double g=1+gg;
    double dfda=pow(g,mypow);
    double f=dfda*amp;
    double dfdbeta=-1.5*log(g)*f;
    double dfdg=f*(0.5-1.5*beta)/g;
    double dfdx=dfdg*2*cosdec*delx*theta_inv_sqr;
    double dfdy=dfdg*2*(dely-sindec*cosdec_inv*delx*delx)*theta_inv_sqr;
    double dfdtheta=dfdg*(-2*gg*theta_inv);
    

    dat[i]=f;
    derivs[i]=dfdx;
    derivs[i+n]=dfdy;
    derivs[i+2*n]=dfdtheta;
    derivs[i+3*n]=dfdbeta;
    derivs[i+4*n]=dfda;
    
  }
}
/*--------------------------------------------------------------------------------*/
