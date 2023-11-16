import numpy as np
import time
import ctypes
from matplotlib import pyplot as plt
plt.ion()


mylib=ctypes.cdll.LoadLibrary("libtest_model_eval.so")

cupoint=mylib.eval_pointing
cupoint.argtypes=(ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p)

cupoint2=mylib.eval_pointing2
cupoint2.argtypes=(ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p)



ra=np.load('ms0735_ra.npy')
dec=np.load('ms0735_dec.npy')

ra0=np.mean(ra,axis=0)
dec0=np.mean(dec,axis=0)
ra0=ra0-ra0.mean()
dec0=dec0-dec0.mean()

n=len(ra0)
ndet=ra.shape[0]
A=np.zeros([n,4])
A[:,0]=1
A[:,1]=np.linspace(-1,1,n)
A[:,2]=ra0
A[:,3]=dec0
#A[:,4]=A[:,1]**2

lhs=A.T@A
rhs=A.T@(ra.T)
fitp_ra=np.linalg.inv(lhs)@rhs

rhs=A.T@(dec.T)
fitp_dec=np.linalg.inv(lhs)@rhs

dd=A@fitp_dec
rr=A@fitp_ra

dra=(ra-rr.T)*180/np.pi*3600
ddec=(dec-dd.T)*180/np.pi*3600

print('scats are ',np.std(ddec),np.std(dra),np.max(np.abs(dra)),np.max(np.abs(ddec)))


ra0=np.asarray(ra0,dtype='float32')
dec0=np.asarray(dec0,dtype='float32')
fitp_ra=np.asarray(fitp_ra,dtype='float32')
fitp_dec=np.asarray(fitp_dec,dtype='float32')

outra=np.ones([ndet,n],dtype='float32')
outdec=np.zeros([ndet,n],dtype='float32')

t1=time.time()
for i in range(100):
    cupoint(outra.ctypes.data,ra0.ctypes.data,dec0.ctypes.data,ndet,n,fitp_ra.ctypes.data)
    cupoint(outdec.ctypes.data,ra0.ctypes.data,dec0.ctypes.data,ndet,n,fitp_dec.ctypes.data)
t2=time.time()
print('took ',(t2-t1)*1000,' msec to get pointing to CPU')

derr=(dec-outdec)*180/np.pi*3600
rerr=(ra-outra)*180/np.pi*3600
print("dec std/max err (arcsec): ",np.std(derr),np.max(np.abs(derr)))
print("ra std/max err (arcsec): ",np.std(rerr),np.max(np.abs(rerr)))


outra[:]=0
outdec[:]=0
t1=time.time()
cupoint2(outra.ctypes.data,outdec.ctypes.data,ra0.ctypes.data,dec0.ctypes.data,ndet,n,fitp_ra.ctypes.data,fitp_dec.ctypes.data)

t2=time.time()
print('took ',(t2-t1)*1000,' msec to get pointing to CPU in dual call')

derr=(dec-outdec)*180/np.pi*3600
rerr=(ra-outra)*180/np.pi*3600
print("dec std/max err (arcsec): ",np.std(derr),np.max(np.abs(derr)))
print("ra std/max err (arcsec): ",np.std(rerr),np.max(np.abs(rerr)))
