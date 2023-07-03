
class Mapset:
    def __init__(self):
        self.nmap=0
        self.maps=[]
    def add_map(self,map):
        self.maps.append(map.copy())
        self.nmap=self.nmap+1
    def clear(self):
        for i in range(self.nmap):
            self.maps[i].clear()
    def copy(self):
        new_mapset=Mapset()
        for i in range(self.nmap):
            new_mapset.add_map(self.maps[i].copy())
        return new_mapset
    def dot(self,mapset):
        tot=0.0
        for i in range(self.nmap):
            tot=tot+self.maps[i].dot(mapset.maps[i])
        return tot
    def axpy(self,mapset,a):
        for i in range(self.nmap):
            self.maps[i].axpy(mapset.maps[i],a)
    def __add__(self,mapset):
        mm=self.copy()
        mm.axpy(mapset,1.0)
        return mm

    def __sub__(self,mapset):
        mm=self.copy()
        mm.axpy(mapset,-1.0)
        return mm
    def __mul__(self,mapset):
        #mm=self.copy()
        mm=mapset.copy()
        #return mm
        for i in range(self.nmap):
            #print('callin mul on map ',i)
            mm.maps[i]=self.maps[i]*mapset.maps[i]
        return mm
    def get_caches(self):
        for i in range(self.nmap):
            self.maps[i].get_caches()
    def clear_caches(self):
        for i in range(self.nmap):
            self.maps[i].clear_caches()
    def apply_prior(self,x,Ax):
        for i in range(self.nmap):
            if not(self.maps[i] is None):
                try:                    
                    if self.maps[i].isglobal_prior:
                        #print('applying global prior')
                        self.maps[i].apply_prior(x,Ax)
                    else:
                        self.maps[i].apply_prior(x.maps[i],Ax.maps[i])
                except:
                    #print('going through exception')
                    self.maps[i].apply_prior(x.maps[i],Ax.maps[i])
    def mpi_reduce(self):
        if have_mpi:
            for map in self.maps:
                map.mpi_reduce()
           
