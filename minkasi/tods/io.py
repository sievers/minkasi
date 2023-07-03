
def read_tod_from_fits_cbass(fname,dopol=False,lat=37.2314,lon=-118.2941,v34=True,nm20=False):
    f=pyfits.open(fname)
    raw=f[1].data
    ra=raw['RA']
    dec=raw['DEC']
    flag=raw['FLAG']
    I=0.5*(raw['I1']+raw['I2'])


    mjd=raw['MJD']
    tvec=(mjd-2455977.5+2400000.5)*86400+1329696000
    #(mjd-2455977.5)*86400+1329696000;
    dt=np.median(np.diff(tvec))

    dat={}
    dat['dx']=np.reshape(np.asarray(ra,dtype='float64'),[1,len(ra)])
    dat['dy']=np.reshape(np.asarray(dec,dtype='float64'),[1,len(dec)])
    dat['dt']=dt
    dat['ctime']=tvec
    if dopol:
        dat['dx']=np.vstack([dat['dx'],dat['dx']])
        dat['dy']=np.vstack([dat['dy'],dat['dy']])
        Q=0.5*(raw['Q1']+raw['Q2'])
        U=0.5*(raw['U1']+raw['U2'])
        dat['dat_calib']=np.zeros([2,len(Q)])
        if v34:  #We believe this is the correct sign convention for V34
            dat['dat_calib'][0,:]=-U
            dat['dat_calib'][1,:]=Q
        else:
            dat['dat_calib'][0,:]=Q
            dat['dat_calib'][1,:]=U            
        az=raw['AZ']
        el=raw['EL']
        #JLS- changing default az/el to radians and not degrees in TOD
        dat['az']=az*np.pi/180
        dat['el']=el*np.pi/180
        
        #dat['AZ']=az
        #dat['EL']=el
        #dat['ctime']=tvec
        dat['mask']=np.zeros([2,len(Q)],dtype='int8')
        dat['mask'][0,:]=1-raw['FLAG']
        dat['mask'][1,:]=1-raw['FLAG']
        if have_qp:
            Q = qp.QPoint(accuracy='low', fast_math=True, mean_aber=True,num_threads=4)
            #q_bore = Q.azel2bore(dat['AZ'], dat['EL'], 0*dat['AZ'], 0*dat['AZ'], lon*np.pi/180, lat*np.pi/180, dat['ctime'])
            q_bore = Q.azel2bore(az,el, 0*az, 0*az, lon, lat, dat['ctime'])
            q_off = Q.det_offset(0.0,0.0,0.0)
            #ra, dec, sin2psi, cos2psi = Q.bore2radec(q_off, ctime, q_bore)
            ra, dec, sin2psi, cos2psi = Q.bore2radec(q_off, tvec, q_bore)
            tmp=np.arctan2(sin2psi,cos2psi) 
            tmp=tmp-np.pi/2 #this seems to be needed to get these coordinates to line up with 
                            #the expected, in IAU convention I believe.  JLS Nov 12 2020
            #dat['twogamma_saved']=np.arctan2(sin2psi,cos2psi)
            dat['twogamma_saved']=np.vstack([tmp,tmp+np.pi/2])
            #print('pointing rms is ',np.std(ra*np.pi/180-dat['dx']),np.std(dec*np.pi/180-dat['dy']))
            dat['ra']=ra*np.pi/180
            dat['dec']=dec*np.pi/180
    else:
        dat['dat_calib']=np.zeros([1,len(I)])
        dat['dat_calib'][:]=I
        dat['mask']=np.zeros([1,len(I)],dtype='int8')
        dat['mask'][:]=1-raw['FLAG']


    dat['pixid']=[0]
    dat['fname']=fname

    if nm20:
        try:
        #kludget to read in bonus cuts, which should be in f[3]
            raw=f[3].data 
            dat['nm20_start']=raw['START']
            dat['nm20_stop']=raw['END']
            #nm20=0*dat['flag']
            print(dat.keys())
            nm20=0*dat['mask']
            start=dat['nm20_start']
            stop=dat['nm20_stop']
            for i in range(len(start)):
                nm20[:,start[i]:stop[i]]=1
                #nm20[:,start[i]:stop[i]]=0
            dat['mask']=dat['mask']*nm20
        except:
            print('missing nm20 for ',fname)

    f.close()
    return dat

def read_tod_from_fits(fname,hdu=1,branch=None):
    f=pyfits.open(fname)
    raw=f[hdu].data
    #print 'sum of cut elements is ',np.sum(raw['UFNU']<9e5)
    try : #read in calinfo (per-scan beam volumes etc) if present
        calinfo={'calinfo':True}
        kwds=('scan','bunit','azimuth','elevatio','beameff','apereff','antgain','gainunc','bmaj','bmin','bpa','parang','beamvol','beamvunc')#for now just hardwired ones we want
        for kwd in kwds:
            calinfo[kwd]=f[hdu].header[kwd]
    except KeyError : 
        print('WARNING - calinfo information not found in fits file header - to track JytoK etc you may need to reprocess the fits files using mustangidl > revision 932') 
        calinfo['calinfo']=False

    pixid=raw['PIXID']
    dets=np.unique(pixid)
    ndet=len(dets)
    nsamp=len(pixid)/len(dets)
    if True:
        ff=180/np.pi
        xmin=raw['DX'].min()*ff
        xmax=raw['DX'].max()*ff
        ymin=raw['DY'].min()*ff
        ymax=raw['DY'].max()*ff
        print('nsamp and ndet are ',ndet,nsamp,len(pixid),' on ',fname, 'with lims ',xmin,xmax,ymin,ymax)
    else:
        print('nsamp and ndet are ',ndet,nsamp,len(pixid),' on ',fname)
    #print raw.names
    dat={}
    #this bit of odd gymnastics is because a straightforward reshape doesn't seem to leave the data in
    #memory-contiguous order, which causes problems down the road
    #also, float32 is a bit on the edge for pointing, so cast to float64
    dx=raw['DX']
    if not(branch is None):
        bb=branch*np.pi/180.0
        dx[dx>bb]=dx[dx>bb]-2*np.pi
    #dat['dx']=np.zeros([ndet,nsamp],dtype=type(dx[0]))
    ndet=int(ndet)
    nsamp=int(nsamp)
    dat['dx']=np.zeros([ndet,nsamp],dtype='float64')
    dat['dx'][:]=np.reshape(dx,[ndet,nsamp])[:]
    dy=raw['DY']
    #dat['dy']=np.zeros([ndet,nsamp],dtype=type(dy[0]))
    dat['dy']=np.zeros([ndet,nsamp],dtype='float64')
    dat['dy'][:]=np.reshape(dy,[ndet,nsamp])[:]
    if 'ELEV' in raw.names:
        elev=raw['ELEV']*np.pi/180
        dat['elev']=np.zeros([ndet,nsamp],dtype='float64')
        dat['elev'][:]=np.reshape(elev,[ndet,nsamp])[:]

    tt=np.reshape(raw['TIME'],[ndet,nsamp])
    tt=tt[0,:]
    dt=np.median(np.diff(tt))
    dat['dt']=dt
    pixid=np.reshape(pixid,[ndet,nsamp])
    pixid=pixid[:,0]
    dat['pixid']=pixid
    dat_calib=raw['FNU']
    #print 'shapes are ',raw['FNU'].shape,raw['UFNU'].shape,np.mean(raw['UFNU']>9e5)
    #dat_calib[raw['UFNU']>9e5]=0.0

    #dat['dat_calib']=np.zeros([ndet,nsamp],dtype=type(dat_calib[0]))
    dat['dat_calib']=np.zeros([ndet,nsamp],dtype='float64') #go to double because why not
    dat_calib=np.reshape(dat_calib,[ndet,nsamp])

    dat['dat_calib'][:]=dat_calib[:]
    if np.sum(raw['UFNU']>9e5)>0:
        dat['mask']=np.reshape(raw['UFNU']<9e5,dat['dat_calib'].shape)
        dat['mask_sum']=np.sum(dat['mask'],axis=0)
    #print 'cut frac is now ',np.mean(dat_calib==0)
    #print 'cut frac is now ',np.mean(dat['dat_calib']==0),dat['dat_calib'][0,0]
    dat['fname']=fname
    dat['calinfo']=calinfo
    f.close()
    return dat


def read_octave_struct(fname):
    f=open(fname)
    nkey=np.fromfile(f,'int32',1)[0]
    #print 'nkey is ' + repr(nkey)
    dat={}
    for i in range(nkey):
        key=f.readline().strip()
        #print 'key is ' + key
        ndim=np.fromfile(f,'int32',1)[0]
        dims=np.fromfile(f,'int32',ndim)
        dims=np.flipud(dims)
        #print 'Dimensions of ' + key + ' are ' + repr(dims)
        nbyte=np.fromfile(f,'int32',1)[0]
        #print 'nbyte is ' + repr(nbyte)
        dtype=get_type(nbyte)
        tmp=np.fromfile(f,dtype,dims.prod())
        dat[key]=np.reshape(tmp,dims)
    f.close()
    return dat


def todvec_from_files_octave(fnames):
    todvec=TodVec()
    for fname in fnames:
        info=read_octave_struct(fname)
        tod=Tod(info)
        todvec.add_tod(tod)
    return todvec
        
def cut_blacklist(tod_names,blacklist):
    mydict={}
    for nm in tod_names:
        tt=nm.split('/')[-1]
        mydict[tt]=nm
    ncut=0
    for nm in blacklist:
        tt=nm.split('/')[-1]
        #if mydict.has_key(tt):
        if tt in mydict:
            ncut=ncut+1
            del(mydict[tt])
    if ncut>0:
        print('deleted ',ncut,' bad files.')
        mynames=mydict.values()
        mynames.sort()
        return mynames
    else:
        return tod_names 

