import minkasi
import numpy as np
from matplotlib import pyplot as plt
import time

fname = '/scratch/r/rbond/jorlo/MS0735/TS_EaCMS0f0_51_5_Oct_2021/Signal_TOD-AGBT19A_092_08-s26.fits'

dat = minkasi.read_tod_from_fits(fname)
minkasi.truncate_tod(dat)

# figure out a guess at common mode and (assumed) linear detector drifts/offset
# drifts/offsets are removed, which is important for mode finding.  CM is *not* removed.
dd, pred2, cm = minkasi.fit_cm_plus_poly(dat["dat_calib"], cm_ord=3, full_out=True)
dat["dat_calib"] = dd
dat["pred2"] = pred2
dat["cm"] = cm

flatten = True

if flatten:
    dat['dat_calib'] -= pred2

tod = minkasi.Tod(dat)
tod.set_apix()

todvec=minkasi.TodVec()
todvec.add_tod(tod)

for tod in todvec.tods:
    tod.set_noise(minkasi.NoiseSmoothedSVD)

tsBowl = minkasi.tsBowl(tod)

mapset=minkasi.Mapset()
mapset.add_map(tsBowl)

rhs=mapset.copy()
todvec.make_rhs(rhs)

x0=rhs.copy()
x0.clear()
t1=time.time()
mapset_out=minkasi.run_pcg(rhs,x0,todvec,maxiter=50)
t2 = time.time()

print('Took {} sec to fit 1 tod'.format(t2-t1))
print(mapset_out.maps[0].params[0])

plt.plot(tod.info['apix'][0], tod.info['dat_calib'][0])
plt.plot(tod.info['apix'][0], np.dot(mapset_out.maps[0].params[0], mapset_out.maps[0].vecs[0].T))

plt.ylim(-0.05, 0.05)

plt.xlabel('apix')
if flatten:
    plt.ylabel('dat_calib - pred2')
else:
    plt.ylabel('dat_calib')

plt.show()


