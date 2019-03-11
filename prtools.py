import numpy
import matplotlib.pyplot as plt
import copy
import sys
import mlearn

# What does python do, if we do  A*W?
# 1. try to use A.__mul__
# 2. try to use W.__rmul__
#
# Ok, what can we expect:
#  a*w    w untrained -> train w
#  a*w    w trained   -> eval w
# === prdataset ============================================
class prdataset(object):
    "Prdataset in python"

    def __init__(self,data,labels=None):
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
        self.setdata(data)
        self.labels = labels
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
        if (nrcl==1):
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
        newd = copy.deepcopy(self)
        if (isinstance(other,prmapping)):
            return other(newd)
        elif (isinstance(other,prdataset)):
            other = other.float()
        newd.data *= other
        return newd
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
        self.featlab = ['Feature 0']
        for i in range(1,data.shape[1]):
            self.featlab.append('Feature %d'%i)
        return self

    def classsizes(self):
        (k,count) = numpy.unique(self.labels,return_counts=True)
        return count

    def __getitem__(self,key):
        newd = copy.deepcopy(self)
        k1 = key[0]
        k2 = key[1]
        if isinstance(k1,int):  # fucking python!
            k1 = range(k1,k1+1)
        if isinstance(k2,int):  # fucking python!!!!
            k2 = range(k2,k2+1)
        newkey = (k1,k2)
        newd.data = newd.data[newkey]
        newd.labels = newd.labels[newkey[0]]
        newd.featlab = newd.featlab[key[1]]
        return newd

    def shape(self,I=None):
        [n,dim] = self.data.shape
        nrcl = len(self.lablist())
        if I is not None:
            if (I==0):
                return n
            elif (I==1):
                return dim
            elif (I==2):
                return nrcl
            else:
                raise ValueError('Only dim=0,1,2 are possible.')
        else:
            return (n,dim,nrcl)

    def seldat(self,cl):
        newd = copy.deepcopy(self)
        #I = (self.labels==cl).nonzero()
        I = (self.nlab()==cl).nonzero()
        return newd[I[0],:]   # grr, this python..

    def getprior(self):
        if (len(self.prior)>0) and (self.prior>0):
            return self.prior
        sz = self.classsizes()
        return sz/float(numpy.sum(sz))

    def concatenate(self,other):
        out = copy.deepcopy(self)
        out.data = numpy.concatenate((out.data,other.data),axis=0)
        out.labels = numpy.concatenate((out.labels,other.labels),axis=0)
        return out



# === prmapping ============================================
class prmapping(object):
    "Prmapping in python"

    def __init__(self,mapping_file,*args):
        self.mapping_file = mapping_file
        self.mapping_type = 'untrained'
        self.args = ()
        if (len(args)==0):
            self.name = mapping_file('untrained') 
        else:
            self.name,self.args = mapping_file('untrained',args) 
        self.data = []
        self.size_in = 0
        self.size_out = 0
        self.user = []

    def __repr__(self):
        return "prmapping("+self.mapping_file.func_name+","+self.mapping_type+")"
    def __str__(self):
        outstr = ""
        if (len(self.name)>0):
            outstr = "%s, " % self.name
        if (self.mapping_type=="untrained"):
            outstr += "untrained mapping"
        elif (self.mapping_type=="trained"):
            outstr += "%d to %d trained mapping" % (self.size_in,self.size_out)
        else:
            raise ValueError('Mapping type is not defined.')
        return outstr
    def __shape__(self):
        return
    def shape(self,I=None):
        if I is not None:
            if (I==0):
                return self.size_in
            elif (dim==1):
                return self.size_out
            else:
                raise ValueError('Only dim=0,1 are possible.')
        else:
            return (self.size_in,self.size_out)

    def __call__(self,x):
        if (self.mapping_type=="untrained"):
            # train
            self = copy.deepcopy(self)  # shit shit shit python
            self.mapping_type = 'trained'
            if (len(self.args)==0):
                self.data = self.mapping_file('train',x)
            else:
                self.data = self.mapping_file('train',x,*(self.args))
            if (hasattr(x,'shape')):  # combiners do not eat datasets
                self.size_in = x.shape(1)
                # and the output size?
                xx = +x[0,:]   # hmmm??
                out = self.mapping_file(self.data,xx)
                self.size_out = out.shape[1]
            return self
        elif (self.mapping_type=="trained"):
            # evaluate
            out = self.mapping_file(self.data,x)
            return out
        else:
            print(self.mapping_type)
            raise ValueError('Mapping type is not defined.')
        return 

    def __mul__(self,other):
        #print('prmapping multiplication with right')
        #print('   self='+str(self))
        #print('   other='+str(other))
        if (isinstance(other,prmapping)):
            # we get a sequential mapping
            leftm = copy.deepcopy(self)
            rightm = copy.deepcopy(other)
            out = prmapping(sequential)
            out = out((leftm,rightm))
            #out.mapping_type = 'untrained'
            #if (leftm.mapping_type=='trained') and (rightm.mapping_type=='trained'):
            #    out.mapping_type = 'trained'
            return out
        else:
            raise ValueError('Prmapping times something not defined.')

def sequential(w,x=None):
    "Sequential mapping"
    #print "sequential: w=",w
    #print "sequential: x=",x
    if isinstance(w,basestring):
        if (w=='untrained'):
            return 'Sequential'
        else: #train the mapping
            # nothing more than save the mappings:
            return x
    #if isinstance(x,prdataset):  # evaluate
    if (True):
        # now it depends which mappings are already trained or not
        if (w[0].mapping_type=='untrained'):
            newm = prmapping(sequential)
            # train the first one:
            w0 = w[0](x)
            newm.size_in = w0.size_in
            if (w[1].mapping_type=='untrained'):
                # map data and train the second mapping:
                out = w0(x)
                w1 = w[1](out)
                # output the combination:
                newm.data = (w0,w1)
                newm.size_out = w1.size_out
            else:
                # just output the fully trained mapping:
                newm.data = (w0,w[1])
                newm.size_out = w[1].size_out
            newm.mapping_type = 'trained'
            return newm

        else: # first mapping is trained, and the second?
            # map data 
            newx = w[0](x)
            if (w[1].mapping_type=='untrained'):
                newm = prmapping(sequential)
                newm.size_in = w[0].size_in
                # train the second mapping:
                print("In sequential, first is trained, second now?")
                print(newx)
                print(w[1].mapping_type)
                w1 = w[1](newx)
                newm.data = (w[0],w1)
                newm.size_out = w1.size_out
                newm.mapping_type = 'trained'
                return newm
            else:
                # all is trained already!
                return w[1](newx)
    else: # evaluate:
        raise ValueError('This combination with sequential looks strange.')

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
    plt.xlabel(a.featlab[0])
    plt.ylabel(ylab)
    plt.winter()

def plotc(f,levels=[0.0],colors=None,gridsize = 30):
    ax = plt.gca()
    xl = ax.get_xlim()
    yl = ax.get_ylim()
    dx = (xl[1]-xl[0])/(gridsize-1)
    dy = (yl[1]-yl[0])/(gridsize-1)
    x = numpy.arange(xl[0],xl[1]+0.01*dx,dx)
    y = numpy.arange(yl[0],yl[1]+0.01*dy,dy)
    z = numpy.zeros((gridsize,gridsize))
    for i in range(0,gridsize):
        for j in range(0,gridsize):
            # have I already told you that I hate python?
            featvec = numpy.array([x[i],y[j]],ndmin=2)
            #z[j,i] = f(featvec)[0]
            z[j,i] = f(featvec)
    plt.contour(x,y,z,levels,colors=colors)

# === mappings ===============================

def scalem(w,x=None):
    "Scale mapping"
    if isinstance(w,basestring):
        if (w=='untrained'):
            # just return the name
            return 'Scalem'
        else:
            # we are going to train the mapping
            mn = numpy.mean(+x,axis=0)
            sc = numpy.std(+x,axis=0)
            return mn,sc
    else:
        # we are applying to new data
        x = x-w[0]
        x = x/w[1]
        return x

def proxm(w,x=None,*args):
    "Proximity mapping"
    if isinstance(w,basestring):
        if (w=='untrained'):
            # just return the name
            return 'Proxm',x
        else:
            # we only need to store the repr. set
            if (isinstance(x,prdataset)):
                R = +x
            else:
                R = numpy.copy(x)
            if (args[0]=='eucl'):
                return 'eucl',R
            if (args[0]=='city'):
                return 'city',R
            elif (args[0]=='rbf'):
                return 'rbf',R,args[1]
            else:
                raise ValueError('Proxm type not defined')
    else:
        # we are applying to new data:
        dat = +x
        n0 = dat.shape[0]
        n1 = w[1].shape[0]
        if (w[0]=='eucl'):
            D = numpy.zeros((n0,n1))
            for i in range(0,n0):
                for j in range(0,n1):
                    df = dat[i,:] - w[1][j,:]
                    D[i,j] = numpy.dot(df.T,df)
        elif (w[0]=='city'):
            D = numpy.zeros((n0,n1))
            for i in range(0,n0):
                for j in range(0,n1):
                    df = dat[i,:] - w[1][j,:]
                    D[i,j] = numpy.sum(numpy.abs(df))
        elif (w[0]=='rbf'):
            s = w[2]*w[2]
            D = numpy.zeros((n0,n1))
            for i in range(0,n0):
                for j in range(0,n1):
                    df = dat[i,:] - w[1][j,:]
                    d = numpy.dot(df.T,df)
                    D[i,j] = numpy.exp(-d/s)
        else:
            raise ValueError('Proxm type not defined')
        if isinstance(x,prdataset):
            x.setdata(D)
            return x
        else:
            return D


def softmax(w,x=None):
    "Softmax function"
    if isinstance(w,basestring):
        if (w=='untrained'):
            # just return the name
            return 'Classc'
        else:
            # we are going to train the mapping
            # nothing here
            return 1.
    else:
        # we are applying to new data
        out = 1./(1.+numpy.exp(-x))
        #DXD: also for more than 1D output?!
        return out

def classc():
    w = prmapping(softmax)
    w.mapping_type = 'trained'
    return w

def nmc(w,x=None):
    "Nearest mean classifier"
    if isinstance(w,basestring):
        if (w=='untrained'):
            # just return the name
            return 'Nearest mean'
        else:
            # we are going to train the mapping
            x0 = x.seldat(0)
            x1 = x.seldat(1)
            mn0 = numpy.mean(+x0,axis=0)
            mn1 = numpy.mean(+x1,axis=0)
            return numpy.vstack((mn0,mn1))
    else:
        # we are applying to new data
        out = sqeucldistm(+x,w)
        df = out[:,1] - out[:,0]
        df = df[:,numpy.newaxis]  # python is soooo stupid
        if isinstance(x,prdataset):
            x.setdata(df)
            return x
        else:
            return df

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

def gendat(x,n):
    nrcl = x.shape(2)
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
    # now generate the data:
    i=0  # first class is special:
    x1 = x.seldat(i)
    if (n[i]==clsz[i]):
        # take a bootstrap sample:
        I = numpy.random.randint(0,n[i],n[i])
        out = x1[I,:]
    elif (n[i]<clsz[i]):
        I = numpy.random.permutation(clsz[i])
        I = I[0:int(n[i])]
        out = x1[I,:]
    else:
        I = numpy.random.randint(clsz[i],size=int(n[i]))
        out = x1[I,:]
    # now the other classes:
    for i in range(1,nrcl):
        xi = x.seldat(i)
        if (n[i]==clsz[i]):
            # take a bootstrap sample:
            I = numpy.random.randint(0,n[i],n[i])
            outi = xi[I,:]
        elif (n[i]<clsz[i]):
            I = numpy.random.permutation(clsz[i])
            I = I[0:int(n[i])]
            outi = xi[I,:]
        else:
            I = numpy.random.randint(clsz[i],size=int(n[i]))
            outi = xi[I,:]
        out = out.concatenate(outi)

    return out


def gendats(n,dim=2,delta=2.):
    N = genclass(n,[0.5,0.5])
    x0 = numpy.random.randn(N[0],dim)
    x1 = numpy.random.randn(N[1],dim)
    x1[:,0] = x1[:,0] + delta  # move data from class 1
    x = numpy.concatenate((x0,x1),axis=0)
    y = genlab(N,(-1,1))
    out = prdataset(x,y)
    out.name = 'Simple dataset'
    out.prior = [0.5,0.5]
    return out

def gendatb(n,s=1.0):
    r = 5
    p = [0.5,0.5]
    N = genclass(n,p)
    domaina = 0.125*numpy.pi + 1.25*numpy.pi*numpy.random.rand(N[0],1)
    a = numpy.concatenate((r*numpy.sin(domaina),r*numpy.cos(domaina)),axis=1)
    a += s*numpy.random.randn(N[0],2)

    domainb = 0.375*numpy.pi - 1.25*numpy.pi*numpy.random.rand(N[1],1)
    b = numpy.concatenate((r*numpy.sin(domainb),r*numpy.cos(domainb)),axis=1)
    b += s*numpy.random.randn(N[1],2)
    b -= 0.75*r*numpy.ones((N[1],2))

    x = numpy.concatenate((a,b),axis=0)
    y = genlab(N,(-1,1))
    out = prdataset(x,y)
    out.name = 'Banana dataset'
    out.prior = [0.5,0.5]
    return out

def classificationerror(f,a=None):
    "Test map"
    if isinstance(f,basestring):
        if (f=='untrained'):
            # just return the name
            return 'Testc'
        else:
            # nothing to train
            return None
    else:
        # we are classifying new data
        pred = +a[:,0]  # this is sooo ugly
        l = mlearn.loss_01(pred,a.signlab())[0]  # but it can be worse
        err = numpy.mean(l)
        return err
def testc():
    out = prmapping(classificationerror)
    out.mapping_type = 'trained'
    return out

def sqeucldistm(a,b):
    n0,dim = a.shape
    n1,dim1 = b.shape
    if (dim!=dim1):
        raise ValueError('Dimensions do not match.')
    D = numpy.zeros((n0,n1))
    for i in range(0,n0):
        for j in range(0,n1):
            df = a[i,:] - b[j,:]
            D[i,j] = numpy.dot(df.T,df)
    return D


#
def m2p(f,*args):
    "ML to PRtools mapping"
    if isinstance(f,basestring):
        if (f=='untrained'):
            return 'M2P '
    if isinstance(f,mlearn.mlmodel):
        # store the model in a prmapping:
        newm = prmapping(m2p)
        newm.data = (f,args) # a bit ugly, but needed
        newm.name += f.name
        newm.size_in = f.dim
        newm.size_out = 1
        newm.mapping_type = 'trained'
        return newm
    else:
        # we are applying to new data, stored in args[0]
        if isinstance(f[0],mlearn.mlmodel):
            functionargs = f[1]
            out = f[0](args[0],*functionargs)  # bloody Python magic
        else:
            print("we did not get a ml model!!")

        return out[0]

