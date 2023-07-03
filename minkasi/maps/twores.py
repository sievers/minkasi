
class SkyMapTwoRes:
    """A pair of maps to serve as a prior for multi-experiment mapping.  This would e.g. be the ACT map that e.g. Mustang should agree
    with on large scales."""
    def __init__(self,map_lowres,lims,osamp=1,smooth_fac=0.0):
        small_wcs,lims_use,map_corner=get_aligned_map_subregion_car(lims,map_lowres,osamp=osamp)
        self.small_lims=lims_use
        self.small_wcs=small_wcs
        self.map=read_fits_map(map_lowres)
        self.osamp=osamp
        self.map_corner=map_corner
        self.beamft=None
        self.mask=None
        self.map_deconvolved=None
        self.noise=None
        self.fine_prior=None
        self.nx_coarse=None
        self.ny_coarse=None
        self.grid_facs=None
        self.isglobal_prior=True
        self.smooth_fac=smooth_fac
    def copy(self):
        return copy.copy(self)
    def get_map_deconvolved(self,map_deconvolved):
        self.map_deconvolved=read_fits_map(map_deconvolved)
    def set_beam_gauss(self,fwhm_pix):
        tmp=0*self.map
        xvec=get_ft_vec(tmp.shape[0])
        yvec=get_ft_vec(tmp.shape[1])
        xx,yy=np.meshgrid(yvec,xvec)
        rsqr=xx**2+yy**2
        sig_pix=fwhm_pix/np.sqrt(8*np.log(2))
        beam=np.exp(-0.5*rsqr/(sig_pix**2))
        beam=beam/np.sum(beam)
        self.beamft=np.fft.rfft2(beam)
    def set_beam_1d(self,prof,pixsize):
        tmp=0*self.map
        xvec=get_ft_vec(tmp.shape[0])
        yvec=get_ft_vec(tmp.shape[1])
        xx,yy=np.meshgrid(yvec,xvec)
        rsqr=xx**2+yy**2
        rr=np.sqrt(rsqr)*pixsize
        beam=interp(rr,prof[:,0],prof[:,1])
        beam=beam/np.sum(beam)
        self.beamft=np.fft.rfft2(beam)


    def set_noise_white(self,ivar_map,isinv=True,nfac=1.0):
        self.noise=MapNoiseWhite(ivar_map,isinv,nfac)
    def maps2fine(self,fine,coarse):
        out=fine.copy()
        for i in range(self.nx_coarse):
            for j in range(self.ny_coarse):
                out[(i*self.osamp):((i+1)*self.osamp),(j*self.osamp):((j+1)*self.osamp)]=coarse[i+self.map_corner[0],j+self.map_corner[1]]
        out[self.mask]=fine[self.mask]
        return out
    def maps2coarse(self,fine,coarse):
        out=coarse.copy()
        for i in range(self.nx_coarse):
            for j in range(self.ny_coarse):
                out[i+self.map_corner[0],j+self.map_corner[1]]=(1-self.grid_facs[i,j])*coarse[i+self.map_corner[0],j+self.map_corner[1]]+np.sum(fine[(i*self.osamp):((i+1)*self.osamp),(j*self.osamp):((j+1)*self.osamp)])/self.osamp**2
        return out
    def coarse2maps(self,inmap):
        coarse=1.0*inmap
        fine=np.zeros([self.nx_coarse*self.osamp,self.ny_coarse*self.osamp])
        for i in range(self.nx_coarse):
            for j in range(self.ny_coarse):
                coarse[i+self.map_corner[0],j+self.map_corner[1]]=(1-self.grid_facs[i,j])*inmap[i+self.map_corner[0],j+self.map_corner[1]]
                fine[(i*self.osamp):((i+1)*self.osamp),(j*self.osamp):((j+1)*self.osamp)]=inmap[i+self.map_corner[0],j+self.map_corner[1]]/self.osamp**2
        fine=fine*self.mask
        return coarse,fine
    def set_mask(self,hits,thresh=0):
        self.mask=hits>thresh
        self.fine_prior=0*hits
        self.nx_coarse=int(np.round(hits.shape[0]/self.osamp))
        self.ny_coarse=int(np.round(hits.shape[1]/self.osamp))
        self.grid_facs=np.zeros([self.nx_coarse,self.ny_coarse])
        for i in range(self.nx_coarse):
            for j in range(self.ny_coarse):
                self.grid_facs[i,j]=np.mean(self.mask[(i*self.osamp):((i+1)*self.osamp),(j*self.osamp):((j+1)*self.osamp)])
                self.fine_prior[(i*self.osamp):((i+1)*self.osamp),(j*self.osamp):((j+1)*self.osamp)]=self.map_deconvolved[self.map_corner[0]+i,self.map_corner[1]+j]
    def apply_Qinv(self,map):
        tmp=self.fine_prior.copy()
        tmp[self.mask]=map[self.mask]
        tmp2=0*self.map_deconvolved.copy()
        for i in range(self.nx_coarse):
            for j in range(self.nx_coarse):
                tmp2[self.map_corner[0]+i,self.map_corner[1]+j]=np.mean(tmp[(i*self.osamp):((i+1)*self.osamp),(j*self.osamp):((j+1)*self.osamp)])
        tmp2_conv=np.fft.irfft2(np.fft.rfft2(tmp2)*self.beamft)
        tmp2_conv_filt=self.noise.apply_noise(tmp2_conv)
        tmp2_reconv=np.fft.irfft2(np.fft.rfft2(tmp2_conv_filt)*self.beamft)
        #tmp2_reconv=np.fft.irfft2(np.fft.rfft2(tmp2_conv)*self.beamft)
        #tmp2_reconv=tmp2.copy()
        fac=1.0/self.osamp**2
        for i in range(self.nx_coarse):
            for j in range(self.ny_coarse):
                tmp[(i*self.osamp):((i+1)*self.osamp),(j*self.osamp):((j+1)*self.osamp)]=fac*tmp2_reconv[i+self.map_corner[0],j+self.map_corner[1]]
        ans=0.0*tmp
        ans[self.mask]=tmp[self.mask]
        return ans
    def apply_H(self,coarse,fine):
        mm=self.maps2coarse(coarse,fine)
        mm=self.beam_convolve(mm)
        return mm
    def apply_HT(self,mm):
        mm=self.beam_convolve(mm)
        coarse,fine=self.coarse2maps(mm)
        return coarse,fine
    def get_rhs(self,mapset):
        #if map is None:
        #    map=self.map
        #map_filt=self.noise.apply_noise(map)
        #map_filt_conv=np.fft.irfft2(np.fft.rfft2(map_filt)*self.beamft)
        #tmp=0.0*self.mask
        #fac=1.0/self.osamp**2
        #for i in range(self.nx_coarse):
        #    for j in range(self.ny_coarse):
        #        tmp[(i*self.osamp):((i+1)*self.osamp),(j*self.osamp):((j+1)*self.osamp)]=fac*map_filt_conv[i+self.map_corner[0],j+self.map_corner[1]]

        #ans=0*tmp
        #ans[self.mask]=tmp[self.mask]
        #return ans
        

        coarse_ind=None
        fine_ind=None
        for i in range(mapset.nmap):
            if isinstance(mapset.maps[i],SkyMapCoarse):
                coarse_ind=i
            else:
                if isinstance(mapset.maps[i],SkyMap):
                    fine_ind=i
        if (coarse_ind is None)|(fine_ind is None):
            print("Errror in twolevel prior:  either fine or coarse skymap not found.")
            return

        
        mm=self.noise.apply_noise(self.map)
        if True:
            coarse,fine=self.apply_HT(mm)
            mapset.maps[coarse_ind].map[:]=mapset.maps[coarse_ind].map[:]+coarse
            mapset.maps[fine_ind].map[:]=mapset.maps[fine_ind].map[:]+fine
        else:
            mm=self.beam_convolve(mm)
            coarse,fine=self.coarse2maps(mm)
            i1=self.map_corner[0]
            i2=i1+self.nx_coarse
            j1=self.map_corner[1]
            j2=j1+self.ny_coarse
            coarse[i1:i2,j1:j2]=coarse[i1:i2,j1:j2]*(1-self.grid_facs)
            mapset.maps[coarse_ind].map[:]=mapset.maps[coarse_ind].map[:]+coarse
            mapset.maps[fine_ind].map[self.mask]=mapset.maps[fine_ind].map[self.mask]+fine[self.mask]/self.osamp**2

    def beam_convolve(self,map):
        mapft=np.fft.rfft2(map)
        mapft=mapft*self.beamft
        return np.fft.irfft2(mapft)
    def apply_prior(self,mapset,outmapset):
        coarse_ind=None
        fine_ind=None
        for i in range(mapset.nmap):
            if isinstance(mapset.maps[i],SkyMapCoarse):
                coarse_ind=i
            else:
                if isinstance(mapset.maps[i],SkyMap):
                    fine_ind=i
        if (coarse_ind is None)|(fine_ind is None):
            print("Errror in twolevel prior:  either fine or coarse skymap not found.")
            return
        if True:
            mm=self.apply_H(mapset.maps[fine_ind].map,mapset.maps[coarse_ind].map)
            mm_filt=self.noise.apply_noise(mm)
            coarse,fine=self.apply_HT(mm_filt)
            
        else:
            summed=self.maps2coarse(mapset.maps[fine_ind].map,mapset.maps[coarse_ind].map)
            summed=self.beam_convolve(summed)
            summed=self.noise.apply_noise(summed)
            summed=self.beam_convolve(summed)
            coarse,fine=self.coarse2maps(summed)

        outmapset.maps[fine_ind].map[self.mask]=outmapset.maps[fine_ind].map[self.mask]+fine[self.mask]
        outmapset.maps[coarse_ind].map[:]=outmapset.maps[coarse_ind].map[:]+coarse

        if self.smooth_fac>0:
            summed=self.maps2coarse(mapset.maps[fine_ind].map,mapset.maps[coarse_ind].map)
            summed_smooth=self.beam_convolve(summed)
            delt=summed-summed_smooth
            delt_filt=self.noise.apply_noise(delt)*self.smooth_fac
            delt_filt=delt_filt-self.beam_convolve(delt_filt)
            coarse,fine=self.coarse2maps(delt_filt)
            outmapset.maps[fine_ind].map[self.mask]=outmapset.maps[fine_ind].map[self.mask]+fine[self.mask]
            outmapset.maps[coarse_ind].map[:]=outmapset.maps[coarse_ind].map[:]+coarse
            



    def __bust_apply_prior(self,map,outmap):
        outmap.map[:]=outmap.map[:]+self.apply_Qinv(map.map)

