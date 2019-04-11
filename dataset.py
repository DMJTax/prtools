"""
Pattern Recognition Dataset class

Should provide a simple and consistent way to deal with datasets.
"""

import numpy
import copy

import matplotlib.pyplot as plt

# === prdataset ============================================
class prdataset(object):
    "Prdataset in python"

    def __init__(self,data,labels=None):
        if isinstance(data,prdataset):
            #self = copy.deepcopy(data) # why does this not work??
            self.__dict__ = data.__dict__.copy()   # sigh..
            return
        if not isinstance(data,(numpy.ndarray, numpy.generic)):
            data = numpy.array(data,ndmin=2)
            if not isinstance(data,(numpy.ndarray, numpy.generic)):
                raise ValueError('Data matrix should be a numpy matrix.')
        if labels is not None:
            if not isinstance(labels,(numpy.ndarray, numpy.generic)):
                raise ValueError('Label vector should be a numpy matrix.')
            if (data.shape[0]!=labels.shape[0]):
                raise ValueError('Number of labels does not match number of data samples.')
        else:
            labels = numpy.zeros(data.shape[0])
        self.name = ''
        self.featlab = numpy.arange(data.shape[1])
        self.setdata(data)
        self.labels = labels
        self._labelnames_ = ()
        self._labels_ = []
        self.prior = []
        self.user = []

    def __str__(self):
        sz = self.data.shape
        if (len(self.name)>0):
            outstr = "%s %d by %d prdataset" % (self.name,sz[0],sz[1])
        else:
            outstr = "%d by %d prdataset" % (sz[0],sz[1])
        cnt = self.classsizes()
        nrcl = len(cnt)
        if (nrcl==0):
            outstr += " with no labels"
        elif (nrcl==1):
            outstr += " with 1 class: [%d]"%sz[0]
        else:
            outstr += " with %d classes: [%d"%(nrcl,cnt[0])
            for i in range(1,nrcl):
                outstr += " %d"%cnt[i]
            outstr += "]"
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
        return numpy.unique(self.labels)
    def nlab(self):
        (ll,I) = numpy.unique(self.labels,return_inverse=True)
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
        try:       # in older versions of numpy the 'count' is not available
            (k,count) = numpy.unique(self.labels,return_counts=True)
        except:
            ll = numpy.unique(self.labels)
            count = numpy.zeros((len(ll),1))
            for i in range(len(ll)):
                count[i] = numpy.sum(1.*(self.labels==ll[i]))
        return count
    def nrclasses(self):
        ll = numpy.unique(self.labels)
        return len(ll)
    def findclass(self,cname):
        ll = numpy.unique(self.labels)
        I = numpy.where(ll==cname)
        return I[0][0]

    def __getitem__(self,key):
        newd = copy.deepcopy(self)
        # deep magic with indices:
        k1 = key[0]
        k2 = key[1]
        if isinstance(k1,int):  # fucking python!
            k1 = range(k1,k1+1)
        if isinstance(k2,int):  # fucking python!!!!
            k2 = range(k2,k2+1)
        newkey = (k1,k2)
        # select columns of feature labels
        newfeatlab = newd.featlab[key[1]]
        if not isinstance(newfeatlab,numpy.ndarray): #DXD why??
            newfeatlab = numpy.array([newfeatlab]) # GRR Python
        newd.featlab = newfeatlab
        # select rows/columns from dataset:
        newd = newd.setdata(newd.data[newkey])
        # select rows from labels
        newd.labels = newd.labels[newkey[0]]
        # select rows from labels
        if (len(newd._labels_)>0):
            newd._labels_ = newd._labels_[newkey[0],:]
        return newd

    def seldat(self,cl):
        newd = copy.deepcopy(self)
        #I = (self.labels==cl).nonzero()
        I = (self.nlab()==cl).nonzero()
        return newd[I[0],:]   # grr, this python..

    def getprior(self):
        if (len(self.prior)>0):
            return self.prior
        sz = self.classsizes()
        return sz/float(numpy.sum(sz))

    def concatenate(self,other):
        out = copy.deepcopy(self)
        out = out.setdata(numpy.concatenate((out.data,other.data),axis=0))
        out.labels = numpy.concatenate((out.labels,other.labels),axis=0)
        out._labels_ = numpy.concatenate((out._labels_,other._labels_),axis=0)
        return out

    def setlabels(self,labelname,labels):
        # does the size match?
        if (len(labels.shape)==1):
            labels = labels[:,numpy.newaxis]
        if (labels.shape[0] != self.data.shape[0]):
            # try transposing:
            labels = labels.transpose()
            if (labels.shape[0] != self.data.shape[0]):
                raise ValueError("Number of labels does not match number of objects.")
        # does labelname already exist?
        if labelname in self._labelnames_:
            # probably overwrite it:
            I = self._labelnames_.index(labelname)
            self._labels_[:,I:(I+1)] = labels
        else:
            # add a new label:
            if (len(self._labelnames_)>0):
                self._labelnames_.append(labelname)
                self._labels_ = numpy.append(self._labels_,labels,1)
            else:
                if not isinstance(labelname,list):
                    labelname = [labelname]
                self._labelnames_ = labelname
                self._labels_ = labels

    def getlabels(self,labelname):
        if labelname in self._labelnames_:
            I = self._labelnames_.index(labelname)
            return self._labels_[:,I:(I+1)]
        else:
            return None

    def showlabels(self,I=None):
        if I is None:
            if (len(self._labelnames_)>0):
                print("This dataset has these labels defined:"),
                print(self._labelnames_)
            else:
                print("No labels defined.")
        else:
            labels = self.getlabels(I)
            if I is None:
                print("Cannot find labels ", I)
            else:
                print(labels)






# === useful functions =====================================
def scatterd(a):
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

def scatterr(a):
    plt.scatter(a.data[:,0],a.labels)
    plt.title(a.name)
    plt.xlabel('Feature '+str(a.featlab[0]))
    plt.ylabel('Target')
    plt.winter()

# === datasets ===============================
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
    if (len(n)!=len(lab)):
        raise ErrorValue('Number of values in N should match number in lab')
    out = numpy.tile(lab[0],[n[0],1])
    for i in range(1,len(n)):
        out=numpy.concatenate((out,numpy.tile(lab[i],[n[i],1])))
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
    numpy.random.seed(seed)
    # now generate the data:
    i=0  # first class is special:
    x1 = x.seldat(i)
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
        xi = x.seldat(i)
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

