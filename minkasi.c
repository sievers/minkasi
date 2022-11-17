#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>

//gcc-4.9 -fopenmp -std=c99 -O3 -shared -fPIC -o libminkasi.so minkasi.c  -lm -lgomp     
//gcc-9 -fopenmp -O3 -shared -fPIC -o libminkasi.so minkasi.c  -lm -lgomp


/*void invsafe_2x2(double *mat, double thresh)
{
  if ((mat[0]==0)&&(mat[3]==0))
    return;
  if ((mat[0]==0)&&(mat[3]>0)) {
    mat[3]=1.0/mat[3];
    mat[1]=0;
    mat[2]=0;
    return;
  }

  if ((mat[0]>0)&&(mat[3]==0)) {
    mat[0]=1.0/mat[0];
    mat[1]=0;
    mat[2]=0;
    return;
  }
  double mydet=mat[0]*mat[3]-mat[1]*mat[2];
  if (abs(mydet)<thresh*abs(mat[0]*mat[3])) {
    //this is the case where the matrix looks singular
    //in this case, pseudo-inverse is the original divided by the trace squared
    double trace=mat[0]+mat[3];
    double fac=(1.0/trace);
    fac=fac*fac;
    for (int i=0;i<4;i++)
      mat[i]=mat[i]*fac;
  } 
  else {
    double tmp=mat[0];
    mat[0]=mat[3]/mydet;
    mat[3]=tmp/mydet;
    tmp=mat[1];
    mat[1]=-mat[2]/mydet;
    mat[2]=-tmp/mydet;
  }
  
}
*/
/*--------------------------------------------------------------------------------*/

void tod2map_simple(double *map, double *dat, int ndet, int ndata, int *pix)
{
  long nn=ndet*ndata;
  for (long i=0;i<nn;i++)
    map[pix[i]]+=dat[i];
}

/*--------------------------------------------------------------------------------*/

void tod2map_atomic(double *map, double *dat, int ndet, int ndata, int *pix)
{
  long nn=ndet*ndata;
#pragma omp parallel for
  for (long i=0;i<nn;i++)
#pragma omp atomic
    map[pix[i]]+=dat[i];
}
/*--------------------------------------------------------------------------------*/
void tod2map_everyone(double *map, double *dat, int ndet, int ndata, int *ipix, int npix, int *edge,int nedge)
//do tod2map where all threads loop over all data, but only assign a region they are responsible for
{
#pragma omp parallel
  {    
    int mythread=omp_get_thread_num();
    if (omp_get_num_threads()<nedge) {  //safety check to make sure # of threads hasn't changed in a bad way
      int itop=edge[mythread+1];
      int ibot=edge[mythread];
      for (int i=0;i<ndet*ndata;i++)
	if ((ipix[i]<itop)&&(ipix[i]>=ibot))
	  map[ipix[i]]+=dat[i];
    }
    
  }
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
void map2tod_iqu_omp(double *dat, double *map, double *twogamma, int ndet, int ndata, int *ipix, int do_add)
{
  long nn=ndet*ndata;
  if (do_add)
#pragma omp parallel for
    for (long i=0;i<nn;i++) {
      dat[i]+=map[3*ipix[i]]+map[3*ipix[i]+1]*cos(twogamma[i])+map[3*ipix[i]+2]*sin(twogamma[i]);
    }
  else
#pragma omp parallel for
    for (long i=0;i<nn;i++) {
      dat[i]=map[3*ipix[i]]+map[3*ipix[i]+1]*cos(twogamma[i])+map[3*ipix[i]+2]*sin(twogamma[i]);
    }
}

/*--------------------------------------------------------------------------------*/
void map2tod_qu_omp(double *dat, double *map, double *twogamma, int ndet, int ndata, int *pix, int do_add)
{
  long nn=ndet*ndata;
  if (do_add)
#pragma omp parallel for
    for (long i=0;i<nn;i++) {
      dat[i]+=map[2*pix[i]]*cos(twogamma[i])+map[2*pix[i]+1]*sin(twogamma[i]);
    }
  else
#pragma omp parallel for
    for (long i=0;i<nn;i++) {
      dat[i]=map[2*pix[i]]*cos(twogamma[i])+map[2*pix[i]+1]*sin(twogamma[i]);
    }
}
/*--------------------------------------------------------------------------------*/
void tod2map_iqu_simple(double *map, double *dat, double *twogamma, int ndet, int ndata, int *pix)
{
  long nn=ndet*ndata;
  for (long i=0;i<nn;i++) {
    map[3*pix[i]]+=dat[i];
    map[3*pix[i]+1]+=dat[i]*cos(twogamma[i]);
    map[3*pix[i]+2]+=dat[i]*sin(twogamma[i]);
  }
  
}
/*--------------------------------------------------------------------------------*/
void tod2map_qu_simple(double *map, double *dat, double *twogamma, int ndet, int ndata, int *pix)
{
  long nn=ndet*ndata;
  for (long i=0;i<nn;i++) {
    map[2*pix[i]]+=dat[i]*cos(twogamma[i]);
    map[2*pix[i]+1]+=dat[i]*sin(twogamma[i]);
  }  
}
/*--------------------------------------------------------------------------------*/
void tod2map_qu_precon_simple(double *map, double *dat, double *twogamma, int ndet, int ndata, int *pix)
{
  long nn=ndet*ndata;
  for (long i=0;i<nn;i++) {
    double mycos=cos(twogamma[i]);
    double mysin=sin(twogamma[i]);
    map[3*pix[i]]+=dat[i]*mycos*mycos;
    map[3*pix[i]+1]+=dat[i]*mysin*mysin;
    map[3*pix[i]+2]+=dat[i]*mysin*mycos;
    
  }  
}

/*--------------------------------------------------------------------------------*/
void scan_map(double *map, int nx, int ny, int npol)
//find non-zero map pixels.  For debugging only.
{
  long ii=0;
  for (int x=0;x<nx;x++)
    for (int y=0;y<ny;y++)
      for (int pol=0;pol<npol;pol++) {
	long ind=x*(ny*npol)+y*npol+pol;
	if (map[ind]!=0)
	  printf("Found map %12.5g on ind %ld %ld\n",map[ind],ind,ii);
	ii=ii+1;
      }
}

///*--------------------------------------------------------------------------------*/
//void invert_qu_precon_simple(double *map, int npix)
//{
//  double *tmp=(double *)malloc(4*sizeof(double));
//  for (int i=0;i<npix;i++) {
//    tmp[0]=map[3*i];
//    tmp[1]=map[3*i+2];
//    tmp[2]=tmp[1];
//    tmp[3]=map[3*i+1];
//    if ((tmp[0]>0)||(tmp[3]>0)) {
//      invsafe_2x2(tmp,1e-6);
//      map[3*i]=tmp[0];
//      map[3*i+1]=tmp[3];
//      map[3*i+2]=tmp[2];  //this had also better equal tmp[1], so it doesn't matter which we take
//    }
//  }
//  free(tmp);
//}
/*--------------------------------------------------------------------------------*/
void tod2map_iqu_precon_simple(double *map, const double *dat, double *twogamma, int ndet, int ndata, const int *pix)
{
  long nn=ndet*ndata;
  for (long i=0;i<nn;i++) {
    double mycos=cos(twogamma[i]);
    double mysin=sin(twogamma[i]);
    map[6*pix[i]]+=dat[i];
    map[6*pix[i]+1]+=dat[i]*mycos;
    map[6*pix[i]+2]+=dat[i]*mysin;
    map[6*pix[i]+3]+=dat[i]*mycos*mycos;
    map[6*pix[i]+4]+=dat[i]*mysin*mycos;
    map[6*pix[i]+5]+=dat[i]*mysin*mysin;    
  }  
}
/*--------------------------------------------------------------------------------*/
void tod2cuts(double *vec, double *dat, long *imap, int ncut,int do_add)
{
  if (do_add)
    for (long i=0;i<ncut;i++)
      vec[i]+=dat[imap[i]];
  else
    for (long i=0;i<ncut;i++)
      vec[i]=dat[imap[i]];
}

/*--------------------------------------------------------------------------------*/
void cuts2tod(double *dat,double *vec, long *imap, int ncut, int do_add)
{
  if (do_add)
    for (long i=0;i<ncut;i++)
      dat[imap[i]]+=vec[i];
  else
    for (long i=0;i<ncut;i++) {
      //if (i<10)
      //printf("assigning dat[%d] to be %12.4f from %12.4f\n",imap[i],vec[i],dat[imap[i]]);
      dat[imap[i]]=vec[i];
      //if (i<10)
      //printf("dat[%d] is now %12.4f\n",imap[i],dat[imap[i]]);
    }
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
void outer(double *A,double *fitp,double *vec,double *out,int n,int ndet,int npar)
{

#pragma omp parallel for
  for (int i=0;i<n;i++)
    for (int j=0;j<ndet;j++) {
      double tmp=vec[i];
      for (int k=0;k<npar;k++)
	tmp+=A[i*npar+k]*fitp[j*npar+k];
      out[i*ndet+j]=tmp;      
    }
}

/*--------------------------------------------------------------------------------*/
void outer_block(double *A,double *fitp,double *vec,double *out,int n,int ndet,int npar)
{

  int bs=16;
  for (int b=0;b<ndet;b+=bs)
    {
      int jmin=b;
      int jmax=b+bs;
      if (jmax>ndet)
	jmax=ndet;
#pragma omp parallel for
      for (int i=0;i<n;i++)
	for (int j=jmin;j<jmax;j++) {
	  double tmp=vec[i];
	  for (int k=0;k<npar;k++)
	    tmp+=A[i*npar+k]*fitp[j*npar+k];
	  //out[i*ndet+j]=tmp;      
	  //out[i]=out[i]+tmp;
	  out[j*n+i]=tmp;
	}
    }
}
