"""
Pattern Recognition Dataset class

Should provide a simple and consistent way to deal with datasets.
A dataset contains:
    data     a data matrix of NxD,  where N is the number of objects,
             and D is the number of features
    targets  the output values that should be predicted from the objects
Additionally, a dataset name, or feature labels can be provided.

The main goal is to keep the labels consistent with the data, when you
try to slice your dataset, or when you want to split your data in a
training and test set:
    a = gendatb([50,50])     generate a Banana prdataset
    [x,z] = gendat(a,0.8)    split in train and test set
    b = a[:,:1]              only select the first feature
    c = a[30:50,:]           only select a few samples
"""

import numpy
import copy

import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from mpl_toolkits import mplot3d

# === prdataset ============================================
class prdataset(object):
    "Prdataset in python"

    def __init__(self,data,targets=None):
        if isinstance(data,prdataset):
            #self = copy.deepcopy(data) # why does this not work??
            self.__dict__ = data.__dict__.copy()   # sigh..
            return
        if not isinstance(data,(numpy.ndarray, numpy.generic)):
            data = numpy.array(data,ndmin=2)
            if not isinstance(data,(numpy.ndarray, numpy.generic)):
                raise TypeError('Data matrix should be a numpy matrix.')
        if targets is not None:
            if not isinstance(targets,(numpy.ndarray, numpy.generic)):
                raise TypeError('Target vector should be a numpy matrix.')
            assert (data.shape[0]==targets.shape[0]), \
                    'Number of targets does not match number of data samples.'
            if (len(targets.shape)<2):
                targets = targets[:,numpy.newaxis]  # SIGH
        else:
            targets = numpy.zeros(data.shape[0])
        self.name = ''
        self.featlab = numpy.arange(data.shape[1])
        self.setdata(data)
        self.targets = targets
        self.targettype = 'crisp'
        self._targetnames_ = ()
        self._targets_ = []
        self.prior = []
        self.user = []

    def __str__(self):
        sz = self.data.shape
        if (len(self.name)>0):
            outstr = "%s %d by %d prdataset" % (self.name,sz[0],sz[1])
        else:
            outstr = "%d by %d prdataset" % (sz[0],sz[1])
        if (self.targettype=='crisp'):
            cnt = self.classsizes()
            nrcl = len(cnt)
            if (nrcl==0):
                outstr += " with no targets"
            elif (nrcl==1):
                outstr += " with 1 class: [%d]"%sz[0]
            else:
                outstr += " with %d classes: [%d"%(nrcl,cnt[0])
                for i in range(1,nrcl):
                    outstr += " %d"%cnt[i]
                outstr += "]"
        elif (self.targettype=='regression'):
            outstr += " with continuous targets."
        return outstr

    def float(self):
        return self.data.copy()
    def __pos__(self):
        return self.float()
    def __add__(self,other):
        newd = copy.deepcopy(self)
        if (isinstance(other,prdataset)):
            other = other.float()
        newd.data += other
        return newd
    def __sub__(self,other):
        newd = copy.deepcopy(self)
        if (isinstance(other,prdataset)):
            other = other.float()
        newd.data -= other
        return newd
    def __mul__(self,other):
        #print('prdataset multiplication with right')
        #print('   self='+str(self))
        #print('   other='+str(other))

        if (isinstance(other,prdataset)):
            other = other.float()
        if (isinstance(other,(int,float))):
            newd = copy.deepcopy(self)
            newd.data *= other
            return newd
        else:
            return NotImplemented
    def __rmul__(self,other):
        #print('prdataset multiplication with right')
        #print('   self='+str(self))
        #print('   other='+str(other))

        if (isinstance(other,prdataset)):
            other = other.float()
        if (isinstance(other,(int,float))):
            newd = copy.deepcopy(self)
            newd.data *= other
            return newd
        else:
            return NotImplemented
    def __div__(self,other):
        newd = copy.deepcopy(self)
        if (isinstance(other,prdataset)):
            other = other.float()
        newd.data /= other
        return newd

    def lablist(self):
        return numpy.unique(self.targets)
    def nlab(self):
        (ll,I) = numpy.unique(self.targets,return_inverse=True)
        I = numpy.array(I)
        I = I[:,numpy.newaxis] # python is so terrible..:-(
        return I
    def signlab(self,posclass=1):  # what is a good default?
        ll = self.lablist()
        if (len(ll)>2):
            raise ValueError('Labels +-1 only for two-class problems.')
        lab = self.nlab()
        if (posclass==0):
            lab = 1-2.0*lab
        else:
            lab = 2.0*lab-1
        return lab
    def setdata(self,data):
        self.data = data
        self.shape = data.shape
        if len(self.featlab) != data.shape[1]:
            self.featlab = ['Feature 0']
            for i in range(1,data.shape[1]):
                self.featlab.append('Feature %d'%i)
        return self

    def classsizes(self):
        if (self.targettype=='regression'):
            return None
        try:       # in older versions of numpy the 'count' is not available
            (k,count) = numpy.unique(numpy.array(self.targets),return_counts=True)
        except:
            ll = numpy.unique(numpy.array(self.targets))
            count = numpy.zeros((len(ll),1))
            for i in range(len(ll)):
                count[i] = numpy.sum(1.*(self.targets==ll[i]))
        return count
    def nrclasses(self):
        ll = numpy.unique(self.targets)
        return len(ll)
    def findclass(self,cname):
        ll = numpy.unique(self.targets)
        I = numpy.where(ll==cname)
        return I[0][0]

    def __getitem__(self,key):
        newd = copy.deepcopy(self)
        # be resistant to 'tuples' that come out of (..).nonzero()
        if isinstance(key[0],tuple):
            raise TypeError("Object indices should be a integer vector "\
                    "(used all output\nfrom I=(..).nonzero()? Use only I[0]).")
        # be resistant to 'integer ndarray' indices:
        if isinstance(key[0],numpy.ndarray):
            key = (key[0].tolist(), key[1])
        if isinstance(key[1],numpy.ndarray):
            key = (key[0], key[1].tolist())
        # be resistant to 'single integer' indices:
        if isinstance(key[0],int):
            key = ([key[0]], key[1])
        if isinstance(key[1],int):
            key = (key[0], [key[1]])
        # when we select columns (features), we have to take care of the
        # feature labels, because that is a *list*:
        if isinstance(key[1],slice):
            # we select columns: in the data and the feature labels
            newd.data = newd.data[:,key[1]]
            newd.featlab = newd.featlab[key[1]]
        elif isinstance(key[1],list):
            if (max(key[1])>=newd.data.shape[1]):
                raise ValueError("Feature indices should be smaller than %d."%newd.data.shape[1])
            newd.data = newd.data[:,key[1]] # ndarrays can handle it
            newd.featlab = [newd.featlab[i] for i in key[1]]
        else:
            print(key[1])
            print(type(key[1]))
            raise ValueError("Only slices or integer lists can be used in indexing.")
        # we select objects: in the data and targets
        newd.data = newd.data[key[0],:]
        newd.targets = newd.targets[key[0]]
        # select rows from targets
        if (len(newd._targets_)>0):
            newd._targets_ = newd._targets_[key[0],:]
        # make the shape consistent:
        newd.shape = newd.data.shape
        return newd

    def __setitem__(self,key,item):
        self.data[key] = item

    def getprior(self):
        if (len(self.prior)>0):
            return self.prior
        sz = self.classsizes()
        return sz/float(numpy.sum(sz))

    def concatenate(self,other,axis=None):
        # Concatenate dataset with something else
        # If the axis is not given, try to infer from the sizes along
        # which direction the concatenation should be performed. The
        # first guess is along dimension 0:
        if (axis is None):
            if (self.shape[1]==other.shape[1]):
                axis = 0
            elif (self.shape[0]==other.shape[0]):
                axis = 1
            else:
                raise ValueError('Datasets do not match size.')

        out = copy.deepcopy(self)
        if (axis==0):
            out = out.setdata(numpy.concatenate((out.data,other.data),axis=0))
            out.targets = numpy.concatenate((out.targets,other.targets),axis=0)
            out._targets_ = numpy.concatenate((out._targets_,other._targets_),axis=0)
        elif (axis==1):
            newfeatlab = numpy.concatenate((out.featlab,out.featlab))
            out = out.setdata(numpy.concatenate((out.data,other.data),axis=1))
            out.featlab = newfeatlab
        else:
            raise ValueError("Concatenation is only possible along axis 0 or 1.")
        return out

    def settargets(self,labelname,targets):
        # does the size match?
        if (len(targets.shape)==1):
            targets = targets[:,numpy.newaxis]
        if (targets.shape[0] != self.data.shape[0]):
            # try transposing:
            targets = targets.transpose()
            if (targets.shape[0] != self.data.shape[0]):
                raise ValueError("Number of targets does not match number of objects.")
        # does labelname already exist?
        if labelname in self._targetnames_:
            # probably overwrite it:
            I = self._targetnames_.index(labelname)
            self._targets_[:,I:(I+1)] = targets
        else:
            # add a new label:
            if (len(self._targetnames_)>0):
                self._targetnames_.append(labelname)
                self._targets_ = numpy.append(self._targets_,targets,1)
            else:
                if not isinstance(labelname,list):
                    labelname = [labelname]
                self._targetnames_ = labelname
                self._targets_ = targets

    def gettargets(self,labelname):
        if labelname in self._targetnames_:
            I = self._targetnames_.index(labelname)
            return self._targets_[:,I:(I+1)]
        else:
            return None

    def showtargets(self,I=None):
        if I is None:
            if (len(self._targetnames_)>0):
                print("This dataset has these targets defined:"),
                print(self._targetnames_)
            else:
                print("No targets defined.")
        else:
            targets = self.gettargets(I)
            if I is None:
                print("Cannot find targets ", I)
            else:
                print(targets)






# === useful functions =====================================
def scatterd(a,clrs=None):
    if (clrs is None):
        clrs = a.nlab().flatten()
    sz = a.data.shape
    if (sz[1]>1):
        plt.scatter(a.data[:,0],a.data[:,1],c=clrs)
        ylab = a.featlab[1]
    else:
        plt.scatter(a.data[:,0],numpy.zeros((sz[0],1)),c=clrs)
        ylab = ''
    plt.title(a.name)
    plt.xlabel('Feature '+str(a.featlab[0]))
    plt.ylabel('Feature '+str(ylab))
    plt.winter()

def scatter3d(a):
    clrs = a.nlab().flatten()
    sz = a.data.shape
    if (sz[1]>2):
        ax = plt.axes(projection='3d')
        ax.scatter3D(a.data[:,0],a.data[:,1],a.data[:,2],c=clrs)
        ylab = a.featlab[1]
        zlab = a.featlab[2]
    else:
        raise ValueError('Please supply at least 3D data.')
    plt.title(a.name)
    ax.set_xlabel('Feature '+str(a.featlab[0]))
    ax.set_ylabel('Feature '+str(ylab))
    ax.set_zlabel('Feature '+str(zlab))
    plt.winter()

def scatterr(a):
    sz = a.data.shape
    if (sz[1]==1):
        plt.scatter(a.data[:,0],a.targets)
        plt.title(a.name)
        plt.xlabel('Feature '+str(a.featlab[0]))
        plt.ylabel('Target')
        plt.winter()
    elif (sz[1]==2):
        ax = plt.axes(projection='3d')
        ax.scatter3D(a.data[:,0],a.data[:,1],a.targets)
        ylab = a.featlab[1]
        plt.title(a.name)
        ax.set_xlabel('Feature '+str(a.featlab[0]))
        ax.set_ylabel('Feature '+str(ylab))
        ax.set_zlabel('Targets')
    else:
        raise ValueError('Please supply at least 2D data.')

def dendro(X, link):
    """
    Plots the hierarchical clustering as a dendrogram
    :param X: prdataset feature vectors
    :param link: linkage type to be used for the dendogram generation
    """
    z = hierarchy.linkage(X, link)
    plt.figure()
    dn = hierarchy.dendrogram(z, orientation='top', show_leaf_counts=True)
    plt.show()
    return dn

def fusion_graph(X, link):
    """
    Plots the hierarchical clustering fusion graph. This functions also
    plots the dendrogram out of which the fusion graph is generated
    :param X: prdataset feature vectors
    :param link: linkage type to be used for the fusion graph generation
    """
    dn = dendro(X, link)
    # Compute fusion levels and number of clusters
    fusion_levels = [el[1] for el in dn['dcoord']]
    fusion_levels = sorted(fusion_levels, key=float, reverse=True)  # sort in descending order
    clusters = [c + 1 for c in range(len(fusion_levels))]
    # Plot fusion graph
    plt.plot(clusters, fusion_levels, 'o-')
    plt.xticks(numpy.arange(1, len(clusters) + 1, step=int(len(clusters)/5)))
    plt.ylabel('Fusion level')
    plt.xlabel('Number of clusters')
    plt.title('Fusion graph')
    plt.show()

# === datasets ===============================

def seldat(x,cl):
    newd = copy.deepcopy(x)
    I = (x.nlab()==cl).nonzero()
    return newd[I[0],:]   # grr, this python..

def genclass(n,p):
    "Generate class frequency distribution"
    if isinstance(n,int):
        n = [n]  # make a list
    n = numpy.asarray(n)
    if (len(n)>1):
        return n
    if (len(n.shape)>1):
        return n
    p = numpy.asarray(p)
    if (numpy.abs(numpy.sum(p)-1)>1e-9):
        raise ValueError('Probabilities do not add up to 1.')
    c = len(p)
    P = numpy.cumsum(p)
    P = numpy.concatenate((numpy.zeros(1),P)) # I hate python
    z = numpy.zeros(c,dtype=int)
    x = numpy.random.rand(numpy.sum(n))
    for i in range(0,c):
        z[i] = numpy.sum((x>P[i]) & (x<P[i+1]))
    return z

def genlab(n,lab):
    if (isinstance(n,int)):
        n = [n]  # grrr Python...
    if (isinstance(lab,str)):
        lab = [lab]
    if (len(n)!=len(lab)):
        raise ValueError('Number of values in N should match number in lab')
    out = numpy.repeat(lab[0],n[0])
    #out = numpy.tile(lab[0],[n[0],1])
    for i in range(1,len(n)):
        out=numpy.concatenate((out,numpy.repeat(lab[i],n[i])))
        #out=numpy.concatenate((out,numpy.tile(lab[i],[n[i],1])))
    return out

def gendat(x,n,seed=None):
    nrcl = x.nrclasses()
    clsz = x.classsizes()
    prior = x.getprior()
    # and all the casting:
    if isinstance(n,float):  # we start with a fraction
        n = n*clsz
    if isinstance(n,int):    # we start with a total number
        n = genclass(n,prior)
    if isinstance(n,tuple):
        n = list(n)
    if isinstance(n[0],float) and (n[0]<1.): # a vector/list of fractions
        n = n*clsz
    # take care for the seed:
    if seed is not None:
        numpy.random.seed(seed)
    # now generate the data:
    i=0  # first class is special:
    x1 = seldat(x,i)
    if (n[i]==clsz[i]):
        # take a bootstrap sample:
        I = numpy.random.randint(0,n[i],n[i])
    elif (n[i]<clsz[i]):
        I = numpy.random.permutation(range(clsz[i]))
        I = I[0:int(n[i])]
    else:
        I = numpy.random.randint(clsz[i],size=int(n[i]))
    out = x1[I,:]
    allI = range(clsz[i])
    J = numpy.setdiff1d(allI,I)
    leftout = x1[J,:]
    # now the other classes:
    for i in range(1,nrcl):
        xi = seldat(x,i)
        if (n[i]==clsz[i]):
            # take a bootstrap sample:
            I = numpy.random.randint(0,n[i],n[i])
        elif (n[i]<clsz[i]):
            I = numpy.random.permutation(range(clsz[i]))
            I = I[0:int(n[i])]
        else:
            I = numpy.random.randint(clsz[i],size=int(n[i]))
        out = out.concatenate(xi[I,:])
        allI = range(clsz[i])
        J = numpy.setdiff1d(allI,I)
        leftout = leftout.concatenate(xi[J,:])

    return out,leftout

def gendatr(x,targets):
    """
    Generate a regression dataset

          a = gendatr(X,Y)

    Generate a regression dataset from data matrix X and target values
    Y. Data matrix X should be NxD, where N is the number of objects,
    and D the feature dimensionality. Target Y should be Nx1.

    Example:
    x = numpy.random.randn(100,2)
    y = numpy.sin(x[:,0])*numpy.sin(x[:,1])
    a = gendatr(x,y)
    """
    a = prdataset(x,targets)
    a.targettype = 'regression'
    return a
