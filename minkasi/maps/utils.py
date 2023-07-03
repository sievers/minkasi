
def read_fits_map(fname,hdu=0,do_trans=True):
    f=fits.open(fname)
    raw=f[hdu].data
    tmp=raw.copy()
    f.close()
    if do_trans:
        tmp=(tmp.T).copy()
    return tmp
def write_fits_map_wheader(map,fname,header,do_trans=True):
    if do_trans:
        map=(map.T).copy()
    hdu=fits.PrimaryHDU(map,header=header)
    try:
        hdu.writeto(fname,overwrite=True)
    except:
        hdu.writeto(fname,clobber=True)
def get_wcs(lims,pixsize,proj='CAR',cosdec=None,ref_equ=False):
    w=wcs.WCS(naxis=2)    
    dec=0.5*(lims[2]+lims[3])
    if cosdec is None:
        cosdec=np.cos(dec)
    if proj=='CAR':
        #CAR in FITS seems to already correct for cosin(dec), which has me confused, but whatever...
        cosdec=1.0
        if ref_equ:
            w.wcs.crval=[0.0,0.0]
            #this seems to be a needed hack if you want the position sent
            #in for the corner to actually be the corner.
            w.wcs.crpix=[lims[1]/pixsize+1,-lims[2]/pixsize+1]
            #w.wcs.crpix=[lims[1]/pixsize,-lims[2]/pixsize]
            #print 'crpix is ',w.wcs.crpix
        else:
            w.wcs.crpix=[1.0,1.0]
            w.wcs.crval=[lims[1]*180/np.pi,lims[2]*180/np.pi]
        w.wcs.cdelt=[-pixsize/cosdec*180/np.pi,pixsize*180/np.pi]
        w.wcs.ctype=['RA---CAR','DEC--CAR']
        return w
    print('unknown projection type ',proj,' in get_wcs.')
    return None

def get_aligned_map_subregion_car(lims,fname=None,big_wcs=None,osamp=1):
    """Get a wcs for a subregion of a map, with optionally finer pixellization.  
    Designed for use in e.g. combining ACT maps and Mustang data.  Input arguments
    are RA/Dec limits for the subregion (which will be tweaked as-needed) and either a 
    WCS structure or the name of a FITS file containing the WCS info the sub-region
    will be aligned with."""
    
    if big_wcs is None:
        if fname is None:
            print("Error in get_aligned_map_subregion_car.  Must send in either a file or a WCS.")
        big_wcs=wcs.WCS(fname)
    ll=np.asarray(lims)
    ll=ll*180/np.pi 
    
    #get the ra/dec limits in big pixel coordinates
    corner1=big_wcs.wcs_world2pix(ll[0],ll[2],0)
    corner2=big_wcs.wcs_world2pix(ll[1],ll[3],0)

    #get the pixel edges for the corners.  FITS works in
    #pixel centers, so edges are a half-pixel off
    corner1[0]=np.ceil(corner1[0])+0.5
    corner1[1]=np.floor(corner1[1])-0.5
    corner2[0]=np.floor(corner2[0])-0.5
    corner2[1]=np.ceil(corner2[1])+0.5
    
    corner1_radec=big_wcs.wcs_pix2world(corner1[0],corner1[1],0)
    corner2_radec=big_wcs.wcs_pix2world(corner2[0],corner2[1],0)

    dra=(corner1_radec[0]-corner2_radec[0])/(corner1[0]-corner2[0])
    ddec=(corner1_radec[1]-corner2_radec[1])/(corner1[1]-corner2[1])
    assert(np.abs(dra/ddec)-1<1e-5)  #we are not currently smart enough to deal with rectangular pixels
    
    lims_use=np.asarray([corner1_radec[0],corner2_radec[0],corner1_radec[1],corner2_radec[1]])
    pixsize=ddec/osamp
    lims_use=lims_use+np.asarray([0.5,-0.5,0.5,-0.5])*pixsize
    
    small_wcs=get_wcs(lims_use*np.pi/180,pixsize*np.pi/180,ref_equ=True)
    imin=int(np.round(corner2[0]+0.5))
    jmin=int(np.round(corner1[1]+0.5))
    map_corner=np.asarray([imin,jmin],dtype='int')
    lims_use=lims_use*np.pi/180

    return small_wcs,lims_use,map_corner

def get_ft_vec(n):
    x=np.arange(n)
    x[x>n/2]=x[x>n/2]-n
    return x

