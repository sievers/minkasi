import numpy as np
from astropy.convolution import convolve, Gaussian2DKernel

############################################################################
#Test to see if ACT response matrix needs to keep scales less than ACT beam#
############################################################################
kernel = Gaussian2DKernel(int(1.0*60/3))
actmap = convolve(tmap, kernel)

#First keep all modes when going to and from need space
tmp = map2wav_real(actmap, wmap.needlet.filters)
allneed = wav2map_real(tmp, wmap.needlet.filters)

i = 0
for i in range(len(wmap.needlet.filters)):
    if wmap.needlet.get_need_lims(i, real_space=True)[1] < 60:
        break

tmp[0,i:] = 0*tmp[0,i:] #0 out scales roughly less than beam
someneed = wav2map_real(tmp, wmap.needlet.filters)
plt.imshow(actmap - someneed[0])
plt.colorbar()
plt.show()

