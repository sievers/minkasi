//nvcc -Xcompiler -fPIC -o libtest_model_eval.so test_model_eval.cu -shared -lgomp 


#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <omp.h>

#define NDET_BLOCK 128


__global__
void add_arrays(float *out, float *in1, float *in2, int n, int m)
{
  for (int i=0;i<m;i+=blockDim.x) {
    int myind=i+threadIdx.x;
    if (myind<m) {
      float a=in1[blockIdx.x*m+myind];
      float b=in2[blockIdx.x*m+myind];
      out[blockIdx.x*m+myind]=a+b;
    }
  }
}

/*--------------------------------------------------------------------------------*/

__global__
void eval_gauss(float *out, float *ra, float *dec, int n, int m,float ra0, float dec0, float sig)
{
  float sinv=-0.5/sig/sig;
  float sinv2=-0.5/(sig/2)/(sig/2);
  for (int i=0;i<m;i+=blockDim.x) {
    int myind=i+threadIdx.x;
    if (myind<m) {
      float a=ra[blockIdx.x*m+myind];
      float b=dec[blockIdx.x*m+myind];
      float dx=a-ra0;
      float dy=b-dec0;
      float rsqr=dx*dx+dy*dy;
      float val1=exp(sinv*rsqr);
      //float val1=rsqr;
#if 0
      dx=dx+0.2;
      dy=dy-0.4;
      rsqr=dx*dx+dy*dy;
      val1=exp(sinv2*rsqr)+val1;
#endif
      //float val2=rsqr+val1;
      out[blockIdx.x*m+myind]=val1;
    }
  }
}

/*--------------------------------------------------------------------------------*/
__global__
void eval_pointing_1mat(float *out, float *boresight_ra, float *boresight_dec, float *fitp, int n, int ndet)
{
  __shared__ float offset[NDET_BLOCK];
  __shared__ float slope[NDET_BLOCK];
  __shared__ float ra_coeff[NDET_BLOCK];
  __shared__ float dec_coeff[NDET_BLOCK];
  int ndet_block=ndet/(blockDim.x);
  if (blockDim.x*ndet_block<ndet)
    ndet_block=ndet_block+1;
  float tmp=0;
  for (int detblock=0;detblock<ndet_block;detblock++) {
	__syncthreads();  //when we're looping, make sure everyone has finished previous iteration before overwriting fit parameters
	int mydet=threadIdx.x+detblock*blockDim.x;
	if (mydet<ndet){
	  offset[threadIdx.x]=fitp[mydet];
	  slope[threadIdx.x]=fitp[mydet+ndet];
	  ra_coeff[threadIdx.x]=fitp[mydet+2*ndet];
	  dec_coeff[threadIdx.x]=fitp[mydet+3*ndet];
	}
      
	__syncthreads(); //make sure all parameters have been loaded before moving on
	
	int myoff=blockIdx.x*blockDim.x+threadIdx.x;
	for (int mysamp=myoff;mysamp<n;mysamp+=gridDim.x*blockDim.x) {
	  if (mysamp<n) {
	    float bra=boresight_ra[mysamp];
	    float bdec=boresight_dec[mysamp];
	    float myt=2.0*((float)mysamp/(float)(n-1))-1;
	    
	    for (int i=0;i<NDET_BLOCK;i++) {
	      int mydet=i+NDET_BLOCK*detblock;

	      if (mydet<ndet) 
		out[mysamp+mydet*n]=myt*slope[i]+offset[i]+ra_coeff[i]*bra+dec_coeff[i]*bdec;				
	    }
	      
	  }
	  
	}
  }
}


/*--------------------------------------------------------------------------------*/
__global__
void eval_pointing_1mat_old(float *out, float *boresight_ra, float *boresight_dec, float *fitp, int n, int ndet)
{
  __shared__ float offset[192];
  __shared__ float slope[192];
  __shared__ float ra_coeff[192];
  __shared__ float dec_coeff[192];
  for (int i=threadIdx.x;i<ndet;i+=blockDim.x) {
    offset[i]=fitp[i];
    slope[i]=fitp[ndet+i];
    ra_coeff[i]=fitp[2*ndet+i];
    dec_coeff[i]=fitp[3*ndet+i];    
  }

  __syncthreads(); //make sure all parameters have been loaded before moving on
  
  int myoff=blockIdx.x*blockDim.x+threadIdx.x;
  for (int mysamp=myoff;mysamp<n;mysamp+=gridDim.x*blockDim.x) {
    if (mysamp<n) {
      float bra=boresight_ra[mysamp];
      float bdec=boresight_dec[mysamp];
      float myt=2.0*((float)mysamp/(float)(n-1))-1;
      
      for (int i=0;i<ndet;i++) {
	out[mysamp+i*n]=myt*slope[i]+offset[i]+ra_coeff[i]*bra+dec_coeff[i]*bdec;
      }
    }
  }
}
/*--------------------------------------------------------------------------------*/

extern "C" {
void eval_gauss(float *dat, float *ra, float *dec, int ndet, int n, float *pars, int ngauss)
{
  float *ddat;
  if (cudaMalloc((void **)&ddat,sizeof(float)*n*ndet)!=cudaSuccess)
    fprintf(stderr,"error in cudaMalloc\n");
  float *dra;
  if (cudaMalloc((void **)&dra,sizeof(float)*n*ndet)!=cudaSuccess)
    fprintf(stderr,"error in cudaMalloc\n");
  float *ddec;
  if (cudaMalloc((void **)&ddec,sizeof(float)*n*ndet)!=cudaSuccess)
    fprintf(stderr,"error in cudaMalloc\n");
  if (cudaMemcpy(ddec,dec,sizeof(float)*n*ndet,cudaMemcpyHostToDevice)!=cudaSuccess)
    fprintf(stderr,"Error in copying dec to device.\n");
  if (cudaMemcpy(dra,ra,sizeof(float)*n*ndet,cudaMemcpyHostToDevice)!=cudaSuccess)
    fprintf(stderr,"Error in copying ra to device.\n");
  double t1=omp_get_wtime();
  //add_arrays<<<ndet,256>>>ddat,dra,ddec,ndet,n);
  eval_gauss<<<128,128>>>(ddat,dra,ddec,ndet,n,pars[0],pars[1],pars[2]);
  cudaDeviceSynchronize();
  double t2=omp_get_wtime();
  printf("Took %10.3e milliseconds to process.\n",(t2-t1)*1e3);
  if (cudaMemcpy(dat,ddat,sizeof(float)*n*ndet,cudaMemcpyDeviceToHost)!=cudaSuccess)
    fprintf(stderr,"Error copying result back to host.\n");
}
}
/*--------------------------------------------------------------------------------*/

extern "C" {
  void eval_pointing(float *out, float *bra, float *bdec, int ndet, int n, float *pars)
{
  int npar=4;
  float *dout;
  if (cudaMalloc((void **)&dout,sizeof(float)*n*ndet)!=cudaSuccess)
    fprintf(stderr,"error in cudaMalloc\n");
  float *dbra;
  if (cudaMalloc((void **)&dbra,sizeof(float)*n)!=cudaSuccess)
    fprintf(stderr,"error in cudaMalloc\n");
  float *dbdec;
  if (cudaMalloc((void **)&dbdec,sizeof(float)*n)!=cudaSuccess)
    fprintf(stderr,"error in cudaMalloc\n");
  
  if (cudaMemcpy(dbdec,bdec,sizeof(float)*n,cudaMemcpyHostToDevice)!=cudaSuccess)
    fprintf(stderr,"Error in copying boresight dec to device.\n");
  if (cudaMemcpy(dbra,bra,sizeof(float)*n,cudaMemcpyHostToDevice)!=cudaSuccess)
    fprintf(stderr,"Error in copying boresight ra to device.\n");

  float *dpars;
  if (cudaMalloc((void **)&dpars,sizeof(float)*npar*ndet)!=cudaSuccess)
    fprintf(stderr,"error in cudaMalloc\n");
  if (cudaMemcpy(dpars,pars,sizeof(float)*ndet*npar,cudaMemcpyHostToDevice)!=cudaSuccess)
    fprintf(stderr,"Error in copying parameters to device.\n");
    
  double t1=omp_get_wtime();
  for (int i=0;i<100;i++)
    eval_pointing_1mat<<<NDET_BLOCK,NDET_BLOCK>>>(dout,dbra,dbdec,dpars,n,ndet);
  //add_arrays<<<ndet,256>>>(ddat,dra,ddec,ndet,n);
  //eval_gauss<<<ndet,128>>>(ddat,dra,ddec,ndet,n,pars[0],pars[1],pars[2]);
  cudaDeviceSynchronize();
  double t2=omp_get_wtime();
  printf("Took %10.3e milliseconds to process.\n",(t2-t1)*1e3);
  if (cudaMemcpy(out,dout,sizeof(float)*n*ndet,cudaMemcpyDeviceToHost)!=cudaSuccess)
    fprintf(stderr,"Error copying ra back to host.\n");
  if (cudaFree(dout)!=cudaSuccess)
    fprintf(stderr,"Error freeing\n");
  if (cudaFree(dbra)!=cudaSuccess)
    fprintf(stderr,"Error freeing\n");
  if (cudaFree(dbdec)!=cudaSuccess)
    fprintf(stderr,"Error freeing\n");
  if (cudaFree(dpars)!=cudaSuccess)
    fprintf(stderr,"Error freeing\n");
}
}
/*--------------------------------------------------------------------------------*/

extern "C" {
  void eval_pointing2(float *raout, float *decout, float *bra, float *bdec, int ndet, int n, float *rapars, float *decpars)
{
  int npar=4;
  float *draout;
  if (cudaMalloc((void **)&draout,sizeof(float)*n*ndet)!=cudaSuccess)
    fprintf(stderr,"error in cudaMalloc\n");

  float *ddecout;
  if (cudaMalloc((void **)&ddecout,sizeof(float)*n*ndet)!=cudaSuccess)
    fprintf(stderr,"error in cudaMalloc\n");


  float *dbra;
  if (cudaMalloc((void **)&dbra,sizeof(float)*n)!=cudaSuccess)
    fprintf(stderr,"error in cudaMalloc\n");
  float *dbdec;
  if (cudaMalloc((void **)&dbdec,sizeof(float)*n)!=cudaSuccess)
    fprintf(stderr,"error in cudaMalloc\n");
  
  if (cudaMemcpy(dbdec,bdec,sizeof(float)*n,cudaMemcpyHostToDevice)!=cudaSuccess)
    fprintf(stderr,"Error in copying boresight dec to device.\n");
  if (cudaMemcpy(dbra,bra,sizeof(float)*n,cudaMemcpyHostToDevice)!=cudaSuccess)
    fprintf(stderr,"Error in copying boresight ra to device.\n");

  float *drapars;
  if (cudaMalloc((void **)&drapars,sizeof(float)*npar*ndet)!=cudaSuccess)
    fprintf(stderr,"error in cudaMalloc\n");
  if (cudaMemcpy(drapars,rapars,sizeof(float)*ndet*npar,cudaMemcpyHostToDevice)!=cudaSuccess)
    fprintf(stderr,"Error in copying parameters to device.\n");

  
  float *ddecpars;
  if (cudaMalloc((void **)&ddecpars,sizeof(float)*npar*ndet)!=cudaSuccess)
    fprintf(stderr,"error in cudaMalloc\n");
  if (cudaMemcpy(ddecpars,decpars,sizeof(float)*ndet*npar,cudaMemcpyHostToDevice)!=cudaSuccess)
    fprintf(stderr,"Error in copying parameters to device.\n");

  double t1=omp_get_wtime();
  for (int i=0;i<100;i++) {
    eval_pointing_1mat<<<64,NDET_BLOCK>>>(draout,dbra,dbdec,drapars,n,ndet);
    eval_pointing_1mat<<<64,NDET_BLOCK>>>(ddecout,dbra,dbdec,ddecpars,n,ndet);
  }
  //add_arrays<<<ndet,256>>>(ddat,dra,ddec,ndet,n);
  //eval_gauss<<<ndet,128>>>(ddat,dra,ddec,ndet,n,pars[0],pars[1],pars[2]);
  cudaDeviceSynchronize();
  double t2=omp_get_wtime();
  printf("Took %10.3e milliseconds to process.\n",(t2-t1)*1e3);
  if (cudaMemcpy(raout,draout,sizeof(float)*n*ndet,cudaMemcpyDeviceToHost)!=cudaSuccess)
    fprintf(stderr,"Error copying ra back to host.\n");
  if (cudaMemcpy(decout,ddecout,sizeof(float)*n*ndet,cudaMemcpyDeviceToHost)!=cudaSuccess)
    fprintf(stderr,"Error copying dec back to host.\n");
  if (cudaFree(draout)!=cudaSuccess)
    fprintf(stderr,"Error freeing\n");
  if (cudaFree(ddecout)!=cudaSuccess)
    fprintf(stderr,"Error freeing\n");
  if (cudaFree(dbra)!=cudaSuccess)
    fprintf(stderr,"Error freeing\n");
  if (cudaFree(dbdec)!=cudaSuccess)
    fprintf(stderr,"Error freeing\n");
  if (cudaFree(drapars)!=cudaSuccess)
    fprintf(stderr,"Error freeing\n");
  if (cudaFree(ddecpars)!=cudaSuccess)
    fprintf(stderr,"Error freeing\n");

}
}
