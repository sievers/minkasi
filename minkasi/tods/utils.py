
def slice_with_copy(arr,ind):
    if isinstance(arr,np.ndarray):
        myshape=arr.shape

        if len(myshape)==1:
            ans=np.zeros(ind.sum(),dtype=arr.dtype)
            print(ans.shape)
            print(ind.sum())
            ans[:]=arr[ind]
        else:   
            mydims=np.append(np.sum(ind),myshape[1:])
            print(mydims,mydims.dtype)
            ans=np.zeros(mydims,dtype=arr.dtype)
            ans[:,:]=arr[ind,:].copy()
        return ans
    return None #should not get here
def split_dict(mydict,vec,thresh):
    #split a dictionary into sub-dictionaries wherever a gap in vec is larger than thresh.
    #useful for e.g. splitting TODs where there's a large time gap due to cuts.
    inds=np.where(np.diff(vec)>thresh)[0]
    #print(inds,len(inds))
    if len(inds)==0:
        return [mydict]
    ndict=len(inds)+1
    inds=np.hstack([[0],inds+1,[len(vec)]])
    #print(inds)

    out=[None]*ndict
    for i in range(ndict):
        out[i]={}
    for key in mydict.keys():
        tmp=mydict[key]
        for i in range(ndict):
            out[i][key]=tmp
        try:
            dims=tmp.shape
            ndim=len(dims)
            if ndim==1:
                if dims[0]==len(vec):
                    for i in range(ndict):
                        out[i][key]=tmp[inds[i]:inds[i+1]].copy()
            if ndim==2:
                if dims[1]==len(vec):
                    for i in range(ndict):
                        out[i][key]=tmp[:,inds[i]:inds[i+1]].copy()
                elif dims[0]==len(vec):
                    for i in range(ndict):
                        out[i][key]=tmp[inds[i]:inds[i+1],:].copy()
        except:
            continue
            #print('copying ',key,' unchanged')
            #don't need below as it's already copied by default
            #for i in range(ndict):
            #    out[i][key]=mydict[key]

    return out

def mask_dict(mydict,mask):
    for key in mydict.keys():
        tmp=mydict[key]
        try:
            dims=tmp.shape
            ndim=len(dims)
            if ndim==1:
                if dims[0]==len(mask):
                    tmp=tmp[mask]
                    mydict[key]=tmp
            if ndim==2:
                if dims[0]==len(mask):
                    tmp=tmp[mask,:]
                if dims[1]==len(mask):
                    tmp=tmp[:,mask]
                mydict[key]=tmp
            if ndim==3:
                if dims[0]==len(mask):
                    tmp=tmp[mask,:,:]
                if dims[1]==len(mask):
                    tmp=tmp[:,mask,:]
                if dims[2]==len(maks):
                    tmp=tmp[:,:,mask]
                mydict[key]=tmp
        except:
            continue
