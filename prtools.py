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
        self.featlab = numpy.arange(data.shape[1])
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
        newd = copy.deepcopy(self)
        if (isinstance(other,prmapping)):
            newother = copy.deepcopy(other)
            return newother(newd)
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

    def __init__(self,mapping_func,x=[],hyperp=[]):
        # exception: when only hyperp are given
        if (not isinstance(x,prdataset)) and (not hyperp): 
            hyperp = x
            x = []
        self.mapping_func = mapping_func
        self.mapping_type = "untrained"
        self.name, self.hyperparam = self.mapping_func("untrained",hyperp)
        self.data = () 
        self.labels = ()
        self.size_in = 0
        self.size_out = 0
        self.user = []
        if isinstance(x,prdataset):
            self = self.train(x)
        #print(self.mapping_type)


    def __repr__(self):
        return "prmapping("+self.mapping_func.func_name+","+self.mapping_type+")"
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

    def init(self,mappingfunc,**kwargs):
        self.mapping_func = mappingfunc
        self.mapping_type = "untrained"
        self.hyperparam = kwargs
        self = self.mapping_func('init',kwargs)
        return self

    def train(self,x,args=None):
        # train
        if (self.mapping_type=="trained"):
            raise ValueError('The mapping is already trained and will be retrained.')
        # maybe the supplied parameters overrule the stored ones:
        if args is not None:
            self.hyperparam = args
        if (len(self.hyperparam)==0):
            self.data,self.labels = self.mapping_func('train',x)
        else:
            self.data,self.labels = self.mapping_func('train',x,self.hyperparam)

        self.mapping_type = 'trained'
        # set the input and output sizes 
        if (hasattr(x,'shape')):  # combiners do not eat datasets
            self.size_in = x.shape(1)
            # and the output size?
            xx = +x[0,:]   # hmmm??
            out = self.mapping_func("eval",xx,self)
            self.size_out = out.shape[1]
        return self

    def eval(self,x):
        # evaluate
        if (self.mapping_type=="untrained"):
            raise ValueError('The mapping is not trained and cannot be evaluated.')
        # not a good idea to supply the true labels:
        x_nolab = copy.deepcopy(x)
        x_nolab.labels = ()
        out = self.mapping_func("eval",x_nolab,self)
        if ((len(self.labels)>0) and (out.shape[1] != len(self.labels))):
            print(len(self.labels))
            raise ValueError('Output of mapping does not match number of labels.')
        if isinstance(x,prdataset):
            x.data = out
            x.featlab = self.labels
            x.size_out = out.shape[1]
            return x
        else:
            return out

    def __call__(self,x):
        if (self.mapping_type=="untrained"):
            # train
            out = self.train(x)
            return out
        elif (self.mapping_type=="trained"):
            # evaluate
            out = self.eval(x)
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
            out = prmapping(sequentialm,(leftm,rightm))
            return out
        else:
            raise ValueError('Prmapping times something not defined.')


def sequentialm(task=None,x=None,w=None):
    "Sequential mapping"
    if not isinstance(task,basestring):
        # we should have gotten a list of two prmappings
        if not isinstance(task,tuple):
            raise ValueError('Sequential map expects a list of 2 prmappings.')
        if not isinstance(task[0],prmapping):
            raise ValueError('Sequential map expects a list of 2 prmappings.')
        if not isinstance(task[1],prmapping):
            raise ValueError('Sequential map expects a list of 2 prmappings.')
        newm = copy.deepcopy(task)
        # if both mappings are trained, the sequential mapping is also trained!
        # (this is an exception to the standard, where you need data in order to
        # train a prmapping)
        if (newm[0].mapping_type=='trained') and \
                (newm[1].mapping_type=='trained'):
            # now the seq.map is already trained:
            if (newm[0].size_out != newm[1].size_in):
                raise ValueError('Output size map1 does not match input size map2.')
            w = prmapping(sequentialm)
            w.data = newm
            w.labels = newm[1].labels
            w.size_in = newm[0].size_in
            w.size_out = newm[1].size_out
            w.mapping_type = 'trained'
            return w
        else:
            return prmapping(sequentialm,newm,x)
    if (task=='untrained'):
        # just return the name, and hyperparameters
        return 'Sequentialm', x
    elif (task=="train"):
        # we are going to train the mapping
        u = copy.deepcopy(w)
        if (u[0].mapping_type=='untrained'):
            neww = u[0].train(x)
            u = (neww, u[1])
        x2 = u[0](x)
        if (u[1].mapping_type=='untrained'):
            neww = u[1].train(x2)
            u = (u[0],neww)
        return u, u[1].labels
    elif (task=="eval"):
        # we are applying to new data
        W = w.data   # get the parameters out
        return W[1](W[0](x))
    else:
        print(task)
        raise ValueError('This task is *not* defined for scalem.')

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

def scalem(task=None,x=None,w=None):
    "Scale mapping"
    if not isinstance(task,basestring):
        out = prmapping(scalem,task,x)
        return out
    if (task=='untrained'):
        # just return the name, and hyperparameters
        return 'Scalem', ()
    elif (task=="train"):
        # we are going to train the mapping
        mn = numpy.mean(+x,axis=0)
        sc = numpy.std(+x,axis=0)
        # return the parameters, and feature labels
        labels = x.featlab
        return (mn,sc), x.featlab
    elif (task=="eval"):
        # we are applying to new data
        W = w.data   # get the parameters out
        x = +x-W[0]
        x = +x/W[1]
        return x
    else:
        print(task)
        raise ValueError('This task is *not* defined for scalem.')

def proxm(task=None,x=None,w=None):
    "Proximity mapping"
    if not isinstance(task,basestring):
        # A direct call to proxm, refer back to prmapping:
        return prmapping(proxm,task,x)
    if (task=='untrained'):
        # just return the name, and hyperparams
        return 'Proxm',x
    elif (task=="train"):
        # we only need to store the representation set
        if (isinstance(x,prdataset)):
            R = +x
        else:
            R = numpy.copy(x)
        if (w[0]=='eucl'):
            return ('eucl',R), numpy.arange(R.shape[1])
        if (w[0]=='city'):
            return ('city',R), numpy.arange(R.shape[1])
        elif (w[0]=='rbf'):
            return ('rbf',R,w[1]), numpy.arange(R.shape[1])
        else:
            raise ValueError('Proxm type not defined')
    elif (task=="eval"):
        # we are applying to new data:
        W = w.data
        dat = +x
        n0 = dat.shape[0]
        n1 = W[1].shape[0]
        if (W[0]=='eucl'):
            D = numpy.zeros((n0,n1))
            for i in range(0,n0):
                for j in range(0,n1):
                    df = dat[i,:] - W[1][j,:]
                    D[i,j] = numpy.dot(df.T,df)
        elif (W[0]=='city'):
            D = numpy.zeros((n0,n1))
            for i in range(0,n0):
                for j in range(0,n1):
                    df = dat[i,:] - W[1][j,:]
                    D[i,j] = numpy.sum(numpy.abs(df))
        elif (W[0]=='rbf'):
            s = W[2]*W[2]
            D = numpy.zeros((n0,n1))
            for i in range(0,n0):
                for j in range(0,n1):
                    df = dat[i,:] - W[1][j,:]
                    d = numpy.dot(df.T,df)
                    D[i,j] = numpy.exp(-d/s)
        else:
            raise ValueError('Proxm type not defined')
        return D
    else:
        print(task)
        raise ValueError('This task is *not* defined for proxm.')


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


def labeld(task=None,x=None,w=None):
    "Label mapping"
    if not isinstance(task,basestring):
        out = prmapping(labeld)
        out.mapping_type = "trained"
        return out
    if (task=='untrained'):
        # just return the name, and hyperparameters
        return 'Label', ()
    elif (task=="train"):
        print("We cannot train the label mapping.")
        return 0, x.featlab
    elif (task=="eval"):
        # we are applying to new data
        I = numpy.argmax(+x,axis=1)
        n = x.shape(0)
        out = numpy.zeros((n,1))
        for i in range(n):
            out[i] = x.featlab[I[i]]
        return out
    else:
        print(task)
        raise ValueError('This task is *not* defined for scalem.')


def nmc(task=None,x=None,w=None):
    "Nearest mean classifier"
    if not isinstance(task,basestring):
        out = prmapping(nmc,task,x)
        return out
    if (task=='untrained'):
        # just return the name, and hyperparameters
        return 'Nearest mean', ()
    elif (task=="train"):
        # we are going to train the mapping
        x0 = x.seldat(0)
        x1 = x.seldat(1)
        mn0 = numpy.mean(+x0,axis=0)
        mn1 = numpy.mean(+x1,axis=0)
        # store the parameters, and labels:
        print(x.lablist())
        return numpy.vstack((mn0,mn1)),x.lablist()
    elif (task=="eval"):
        # we are applying to new data
        W = w.data
        out = sqeucldistm(+x,W)
        df = out[:,1] - out[:,0]
        df = df[:,numpy.newaxis]  # python is soooo stupid
        return numpy.hstack((df,-df))
    else:
        print(task)
        raise ValueError('This task is *not* defined for scalem.')


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
    elif (n[i]<clsz[i]):
        I = numpy.random.permutation(clsz[i])
        I = I[0:int(n[i])]
    else:
        I = numpy.random.randint(clsz[i],size=int(n[i]))
    out = x1[I,:]
    allI = numpy.arange(clsz[i])
    J = numpy.setdiff1d(allI,I)
    leftout = x1[J,:]
    # now the other classes:
    for i in range(1,nrcl):
        xi = x.seldat(i)
        if (n[i]==clsz[i]):
            # take a bootstrap sample:
            I = numpy.random.randint(0,n[i],n[i])
        elif (n[i]<clsz[i]):
            I = numpy.random.permutation(clsz[i])
            I = I[0:int(n[i])]
        else:
            I = numpy.random.randint(clsz[i],size=int(n[i]))
        out = out.concatenate(xi[I,:])
        allI = numpy.arange(clsz[i])
        J = numpy.setdiff1d(allI,I)
        leftout = leftout.concatenate(xi[J,:])

    return out,leftout


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

