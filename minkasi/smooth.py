
def smooth_spectra(spec,fwhm):
    nspec=spec.shape[0]
    n=spec.shape[1]

    x=np.arange(n)
    sig=fwhm/np.sqrt(8*np.log(2))
    to_conv=np.exp(-0.5*(x/sig)**2)
    tot=to_conv[0]+to_conv[-1]+2*to_conv[1:-1].sum() #r2r normalization
    to_conv=to_conv/tot
    to_conv_ft=mkfftw.fft_r2r(to_conv)
    xtrans=mkfftw.fft_r2r(spec)
    for i in range(nspec):
        xtrans[i,:]=xtrans[i,:]*to_conv_ft
    #return mkfftw.fft_r2r(xtrans)/(2*(xtrans.shape[1]-1)),to_conv
    return xtrans,to_conv_ft
def smooth_many_vecs(vecs,fwhm=20):
    n=vecs.shape[1]
    nvec=vecs.shape[0]
    x=np.arange(n)
    sig=fwhm/np.sqrt(8*np.log(2))
    to_conv=np.exp(-0.5*(x/sig)**2)
    tot=to_conv[0]+to_conv[-1]+2*to_conv[1:-1].sum() #r2r normalization
    to_conv=to_conv/tot
    to_conv_ft=mkfftw.fft_r2r(to_conv)
    xtrans=mkfftw.fft_r2r(vecs)
    for i in range(nvec):
        xtrans[i,:]=xtrans[i,:]*to_conv_ft
    back=mkfftw.fft_r2r(xtrans)
    return back/(2*(n-1))
def smooth_vec(vec,fwhm=20):
    n=vec.size
    x=np.arange(n)
    sig=fwhm/np.sqrt(8*np.log(2))
    to_conv=np.exp(-0.5*(x/sig)**2)
    tot=to_conv[0]+to_conv[-1]+2*to_conv[1:-1].sum() #r2r normalization
    to_conv=to_conv/tot
    to_conv_ft=mkfftw.fft_r2r(to_conv)
    xtrans=mkfftw.fft_r2r(vec)
    back=mkfftw.fft_r2r(xtrans*to_conv_ft)
    return back/2.0/(n-1)
