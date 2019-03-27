import numpy
import matplotlib.pyplot as plt
import copy
import mlearn
from sklearn import svm
from sklearn import tree


# === prmapping ============================================
class prmapping(object):
    "Prmapping in Python"

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
        self.shape = [0,0]
        self.user = []
        if isinstance(x,prdataset):
            self = self.train(copy.deepcopy(x))

    def __repr__(self):
        return "prmapping("+self.mapping_func.func_name+","+self.mapping_type+")"
    def __str__(self):
        outstr = ""
        if (len(self.name)>0):
            outstr = "%s, " % self.name
        if (self.mapping_type=="untrained"):
            outstr += "untrained mapping"
        elif (self.mapping_type=="trained"):
            outstr += "%d to %d trained mapping" % (self.shape[0],self.shape[1])
        else:
            raise ValueError('Mapping type is not defined.')
        return outstr

    def shape(self,I=None):
        if I is not None:
            if (I==0):
                return self.shape[0]
            elif (dim==1):
                return self.shape[1]
            else:
                raise ValueError('Only dim=0,1 are possible.')
        else:
            return self.shape

    def init(self,mappingfunc,**kwargs):
        self.mapping_func = mappingfunc
        self.mapping_type = "untrained"
        self.hyperparam = kwargs
        self.name,self.hyperparam = self.mapping_func('untrained',kwargs)
        return self

    def train(self,x,args=None):
        # train
        if (self.mapping_type=="trained"):
            raise ValueError('The mapping is already trained and will be retrained.')
        # maybe the supplied parameters overrule the stored ones:
        if args is not None:
            self.hyperparam = args
        #if (len(self.hyperparam)==0):
        #    self.data,self.labels = self.mapping_func('train',x)
        #else:
        self.data,self.labels = self.mapping_func('train',x,self.hyperparam)

        self.mapping_type = 'trained'
        # set the input and output sizes 
        if (hasattr(x,'shape')):  # combiners do not eat datasets
            self.shape[0] = x.shape[1]
            # and the output size?
            xx = +x[:1,:]   # hmmm??
            out = self.mapping_func("eval",xx,self)
            self.shape[1] = out.shape[1]
        return self

    def eval(self,x):
        # evaluate
        if (self.mapping_type=="untrained"):
            raise ValueError('The mapping is not trained and cannot be evaluated.')
        # not a good idea to supply the true labels?
        # but it is needed for testc!
        #if isinstance(x,prdataset):
        #    x_nolab = copy.deepcopy(x)
        #    x_nolab.labels = ()
        #    out = self.mapping_func("eval",x_nolab,self)
        #else:
        newx = copy.deepcopy(x)
        out = self.mapping_func("eval",newx,self)
        if ((len(self.labels)>0) and (out.shape[1] != len(self.labels))):
            print(out.shape)
            print(self.labels)
            raise ValueError('Output of mapping does not match number of labels.')
        if (len(self.labels)>0):  # is this better than above?
            if not isinstance(x,prdataset):
                newx = prdataset(newx)
            newx.featlab = self.labels
            newx = newx.setdata(+out)
            return newx
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
            # avoid the constructor of prmapping: the constructor will always
            # return an 'untrained' mapping, while it *might* be possible to
            # get a trained one when the two input mappings are already
            # trained:
            out = sequentialm((leftm,rightm))
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
            if (newm[0].shape[1] != 0) and (newm[1].shape[0] !=0) and \
                    (newm[0].shape[1] != newm[1].shape[0]):
                raise ValueError('Output size map1 does not match input size map2.')
            # do the constructor, but make sure that the hyperparameters are None:
            w = prmapping(sequentialm,None)
            w.data = newm
            w.shape[0] = newm[0].shape[0]
            w.shape[1] = newm[1].shape[1]
            if (len(newm[1].labels)==0) and (newm[1].shape[1]==0):
                # the second mapping does not have labels and sizes defined:
                w.labels = newm[0].labels
            else:
                w.labels = newm[1].labels
            w.mapping_type = 'trained'
            w.name = newm[0].name+'+'+newm[1].name
            return w
        else:
            if x is None:
                w = prmapping(sequentialm,newm,x)
            else:
                newx = copy.deepcopy(x)
                w = prmapping(sequentialm,newm,newx)
            w.name = newm[0].name+'+'+newm[1].name
            return w
    if (task=='untrained'):
        # just return the name, and hyperparameters
        if x is None:
            mapname = 'Sequential'
        else:
            mapname = x[0].name+'+'+x[1].name
        return mapname, x
    elif (task=="train"):
        # we are going to train the mapping
        u = copy.deepcopy(w)  # I hate Python..
        x1 = copy.deepcopy(x) # Did I say that I hate Python??
        if (u[0].mapping_type=='untrained'):
            neww = u[0].train(x1)
            u = (neww, u[1])
        newx = copy.deepcopy(x1)
        x2 = u[0](newx)
        if (u[1].mapping_type=='untrained'):
            neww = u[1].train(x2)
            u = (u[0],neww)
        # fix the labels:
        if (len(u[1].labels)==0) and (u[1].shape[1]==0):
            # the second mapping does not have labels and sizes defined:
            labels = u[0].labels
        else:
            labels = u[1].labels
        return u, labels
    elif (task=="eval"):
        # we are applying to new data
        W = w.data   # get the parameters out
        return W[1](W[0](x))
    else:
        print(task)
        raise ValueError('This task is *not* defined for scalem.')

# === useful functions =====================================

def plotc(f,levels=[0.0],colors=None,gridsize = 30):
    ax = plt.gca()
    xl = ax.get_xlim()
    yl = ax.get_ylim()
    dx = (xl[1]-xl[0])/(gridsize-1)
    dy = (yl[1]-yl[0])/(gridsize-1)
    x = numpy.arange(xl[0],xl[1]+0.01*dx,dx)
    y = numpy.arange(yl[0],yl[1]+0.01*dy,dy)
    X0,X1 = numpy.meshgrid(x,y)
    X0.shape = (gridsize*gridsize, 1)
    X1.shape = (gridsize*gridsize, 1)
    dat = numpy.hstack((X0,X1))
    out = +f(dat)
    for i in range(1,out.shape[1]):
        otherout = copy.deepcopy(out)
        otherout[:,i] = -numpy.inf
        z = out[:,i] - numpy.amax(otherout,axis=1)
        z.shape = (gridsize,gridsize)
        plt.contour(x,y,z,levels,colors=colors)

def plotm(f,nrlevels=10,colors=None,gridsize = 30):
    ax = plt.gca()
    xl = ax.get_xlim()
    yl = ax.get_ylim()
    dx = (xl[1]-xl[0])/(gridsize-1)
    dy = (yl[1]-yl[0])/(gridsize-1)
    x = numpy.arange(xl[0],xl[1]+0.01*dx,dx)
    y = numpy.arange(yl[0],yl[1]+0.01*dy,dy)
    X0,X1 = numpy.meshgrid(x,y)
    X0.shape = (gridsize*gridsize, 1)
    X1.shape = (gridsize*gridsize, 1)
    dat = numpy.hstack((X0,X1))
    out = +f(dat)
    for i in range(out.shape[1]):
        z = out[:,i]
        levels = numpy.linspace(numpy.min(z),numpy.max(z),nrlevels)
        z.shape = (gridsize,gridsize)
        plt.contour(x,y,z,levels,colors=colors)

# === mappings ===============================

def scalem(task=None,x=None,w=None):
    "Scale mapping"
    if not isinstance(task,basestring):
        return prmapping(scalem,task,x)
    if (task=='untrained'):
        # just return the name, and hyperparameters
        return 'Scalem', ()
    elif (task=="train"):
        # we are going to train the mapping
        mn = numpy.mean(+x,axis=0)
        sc = numpy.std(+x,axis=0)
        # return the parameters, and feature labels
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
            return ('eucl',R), numpy.arange(R.shape[0])
        if (w[0]=='city'):
            return ('city',R), numpy.arange(R.shape[0])
        elif (w[0]=='rbf'):
            return ('rbf',R,w[1]), numpy.arange(R.shape[0])
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


def softmax(task=None,x=None,w=None):
    "Softmax mapping"
    if not isinstance(task,basestring):
        out = prmapping(softmax)
        out.mapping_type = "trained"
        if task is not None:
            out = out(task)
        return out
    if (task=='untrained'):
        # just return the name, and hyperparameters
        return 'Softmax', ()
    elif (task=="train"):
        print("Softmax: We cannot train the softmax mapping.")
        return 0, x.featlab
    elif (task=="eval"):
        # we are applying to new data
        dat = numpy.exp(+x)
        sumx = numpy.sum(dat,axis=1)
        sumx = sumx[:,numpy.newaxis]
        x.setdata( dat/sumx )
        return x
    else:
        print(task)
        raise ValueError('This task is *not* defined for softmax.')

def classc(task=None,x=None,w=None):
    "Classc mapping"
    if not isinstance(task,basestring):
        out = prmapping(classc)
        out.mapping_type = "trained"
        if task is not None:
            out = out(task)
        return out
    if (task=='untrained'):
        # just return the name, and hyperparameters
        return 'Classc', ()
    elif (task=="train"):
        print("Classc: We cannot train the classc mapping.")
        return 0, x.featlab
    elif (task=="eval"):
        # we are applying to new data
        if (numpy.any(+x<0.)):
            print('classc(): Suspicious negative values in Classc.')
        sumx = numpy.sum(+x,axis=1)
        sumx = sumx[:,numpy.newaxis]
        x.setdata( +x/sumx )
        return x
    else:
        print(task)
        raise ValueError('This task is *not* defined for classc.')


def labeld(task=None,x=None,w=None):
    "Label mapping"
    if not isinstance(task,basestring):
        out = prmapping(labeld)
        out.mapping_type = "trained"
        if task is not None:
            out = out(task)
        return out
    if (task=='untrained'):
        # just return the name, and hyperparameters
        return 'Labeld', ()
    elif (task=="train"):
        print("Labeld: We cannot train the label mapping.")
        return 0, x.featlab
    elif (task=="eval"):
        # we are applying to new data
        I = numpy.argmax(+x,axis=1)
        n = x.shape[0]
        out = numpy.zeros((n,1))
        for i in range(n):
            out[i] = x.featlab[I[i]]
        return out
    else:
        print(task)
        raise ValueError('This task is *not* defined for labeld.')

def testc(task=None,x=None,w=None):
    "Test classifier"
    if not isinstance(task,basestring):
        out = prmapping(testc)
        out.mapping_type = "trained"
        return out
    if (task=='untrained'):
        # just return the name, and hyperparameters
        return 'Test classifier', ()
    elif (task=="train"):
        # nothing to train
        return None,()
    elif (task=="eval"):
        # we are classifying new data
        err = (labeld(x) != x.labels)*1.
        if (len(x.weights)>0):
            err *= x.weights
        return numpy.mean(err)
    else:
        print(task)
        raise ValueError('This task is *not* defined for testc.')

def mclassc(task=None,x=None,w=None):
    "Multiclass classifier from two-class classifier"
    if not isinstance(task,basestring):
        out = prmapping(mclassc,task,x)
        return out
    if (task=='untrained'):
        # just return the name, and hyperparameters
        if isinstance(x,prmapping):
            name = 'Multiclass '+x.name
        else:
            name = 'Multiclass'
        return name, x
    elif (task=="train"):
        # we are going to train the mapping
        c = x.nrclasses()
        lablist = x.lablist()
        orglab = x.nlab()
        f = []
        for i in range(c):
            # relabel class i to +1, and the rest to -1:
            newlab = (orglab==i)*2 - 1
            x.labels = newlab
            u = copy.deepcopy(w)
            f.append(u.train(x))
        # store the parameters, and labels:
        return f,lablist
    elif (task=="eval"):
        # we are applying to new data
        W = w.data
        c = len(W)
        pred = numpy.zeros((x.shape[0],c))
        for i in range(c):
            out = +(W[i](x))
            # which output should we choose?
            I = numpy.where(W[i].labels==+1)
            pred[:,i:(i+1)] = out[:,I[0]]
        return pred
    else:
        print(task)
        raise ValueError('This task is *not* defined for mclassc.')

def bayesrule(task=None,x=None,w=None):
    "Bayesrule"
    if not isinstance(task,basestring):
        out = prmapping(bayesrule)
        out.mapping_type = "trained"
        return out
    if (task=='untrained'):
        # just return the name, and hyperparameters
        return 'Bayes rule', ()
    elif (task=="train"):
        # nothing to train
        return None,()
    elif (task=="eval"):
        # we are classifying new data
        if (len(x.prior)>0):
            dat = x.data*x.prior
        else:
            dat = x.data
        Z = numpy.sum(dat,axis=1)
        out = dat/Z[:,numpy.newaxis]
        x = x.setdata(out)
        return x
    else:
        print(task)
        raise ValueError('This task is *not* defined for bayesrule.')

def gaussm(task=None,x=None,w=None):
    "Gaussian density"
    if not isinstance(task,basestring):
        out = prmapping(gaussm,task,x)
        return out
    if (task=='untrained'):
        # just return the name, and hyperparameters
        if x is None:
            x = ('full',[0.])
        return 'Gaussian density', x
    elif (task=="train"):
        # we are going to train the mapping
        c = x.nrclasses()
        dim = x.shape[1]
        prior = x.getprior()
        mn = numpy.zeros((c,dim))
        cv = numpy.zeros((c,dim,dim))
        icov = numpy.zeros((c,dim,dim))
        Z = numpy.zeros((c,1))
        for i in range(c):
            xi = x.seldat(i)
            mn[i,:] = numpy.mean(+xi,axis=0)
            cv[i,:,:] = numpy.cov(+xi,rowvar=False)
        # depending of the type, we have to treat the cov's:
        if (w[0]=='full'):
            for i in range(c):
                # regularise
                cv[i,:,:] += w[1]*numpy.eye(dim)
                icov[i,:,:] = numpy.linalg.inv(cv[i,:,:])
                Z[i] = numpy.sqrt(numpy.linalg.det(cv[i,:,:])*(2*numpy.pi)**dim)
        elif (w[0]=='meancov'):
            meancov = numpy.mean(cv,axis=0) + w[1]*numpy.eye(dim)
            meanZ = numpy.sqrt(numpy.linalg.det(meancov)*(2*numpy.pi)**dim)
            meanicov = numpy.linalg.inv(meancov)
            for i in range(c):
                icov[i,:,:] = meanicov
                Z[i] = meanZ
        else:
            raise ValueError('This cov.mat is *not* defined for gaussm.')
        # store the parameters, and labels:
        return (prior,mn,icov,Z),x.lablist()
    elif (task=="eval"):
        # we are applying to new data
        X = +x
        n = X.shape[0]
        if (len(X.shape)>1):
            dim = len(X.shape)
        else:
            dim = 1
        W = w.data
        c = len(W[0])
        out = numpy.zeros((n,c))
        for i in range(c):
            df = X - W[1][i,:]
            if (dim>1):
                out[:,i] = W[0][i] * numpy.sum(numpy.dot(df,W[2][i,:,:])*df,axis=1)
            else:
                out[:,i] = W[0][i] * numpy.dot(df,W[2][i,:,:])*df
            out[:,i] = numpy.exp(-out[:,i]/2)/W[3][i]
        return out
    else:
        print(task)
        raise ValueError('This task is *not* defined for gaussm.')

def ldc(task=None,x=None,w=None):
    if x is None:  # no regularization of the cov.matrix
        x = [0.]
    u = gaussm(task,('meancov',x))*bayesrule()
    u.name = 'LDA'
    return u

def qdc(task=None,x=None,w=None):
    if x is None:
        x = [0.]
    u = gaussm(task,('full',x))*bayesrule()
    u.name = 'QDA'
    return u

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
        c = x.nrclasses()
        mn = numpy.zeros((c,x.shape[1]))
        for i in range(c):
            xi = x.seldat(i)
            mn[i,:] = numpy.mean(+xi,axis=0)
        # store the parameters, and labels:
        return mn,x.lablist()
    elif (task=="eval"):
        # we are applying to new data
        W = w.data
        out = sqeucldist(+x,W)
        return -out
    else:
        print(task)
        raise ValueError('This task is *not* defined for nmc.')

def fisherc(task=None,x=None,w=None):
    "Fisher classifier"
    if not isinstance(task,basestring):
        out = prmapping(fisherc,task,x)
        return out
    if (task=='untrained'):
        # just return the name, and hyperparameters
        return 'Fisher', ()
    elif (task=="train"):
        # we are going to train the mapping
        c = x.nrclasses()
        dim = x.shape[1]
        if (c>2):
            raise ValueError('Fisher classifier is defined for two classes.')
        mn = numpy.zeros((c,dim))
        cv = numpy.zeros((dim,dim))
        v0 = 0.
        for i in range(c):
            xi = x.seldat(i)
            mn[i,:] = numpy.mean(+xi,axis=0)
            thiscov = numpy.cov(+xi,rowvar=False)
            #DXD: is this a good idea?
            thiscov += 1e-9*numpy.eye(dim)
            cv += thiscov
            icv = numpy.linalg.inv(thiscov)
            #icv = numpy.linalg.pinv(thiscov)
            v0 += mn[i,].dot(icv.dot(mn[i,:]))/2.
        cv /= c # normalise by nr. of classes (2)
        v = numpy.linalg.inv(cv).dot(mn[1,:]-mn[0,:])
        #v = numpy.linalg.pinv(cv).dot(mn[1,:]-mn[0,:])
        # store the parameters, and labels:
        return (v,v0),x.lablist()
    elif (task=="eval"):
        # we are applying to new data
        W = w.data
        X = +x
        out = X.dot(W[0]) - W[1] 
        if (len(out.shape)<2):  # This numpy is pathetic
            out = out[:,numpy.newaxis]
        gr = numpy.hstack((-out,out))
        if (len(gr.shape)<2):
            gr = gr[numpy.newaxis,:]
        return gr
    else:
        print(task)
        raise ValueError('This task is *not* defined for fisherc.')

def knnm(task=None,x=None,w=None):
    "k-Nearest neighbor classifier"
    if not isinstance(task,basestring):
        out = prmapping(knnm,task,x)
        return out
    if (task=='untrained'):
        # just return the name, and hyperparameters
        if x is None:
            x = [1]
        return 'k-Nearest neighbor', x
    elif (task=="train"):
        # we only need to store the data
        # store the parameters, and labels:
        return x,x.lablist()
    elif (task=="eval"):
        # we are applying to new data
        W = w.data
        nrcl = len(w.labels)
        k = w.hyperparam[0]
        n = x.shape[0]
        lab = W.nlab()
        out = numpy.zeros((n,nrcl))
        D = sqeucldist(+x,+W)
        I = numpy.argsort(D,axis=1)
        for i in range(n):
            thislab = lab[I[i,0:k]]
            thislab.shape = (1,k)
            out[i,:] = numpy.bincount(thislab[0],minlength=nrcl)/k
        return out
    else:
        print(task)
        raise ValueError('This task is *not* defined for knnc.')

def knnc(task=None,x=None,w=None):
    return knnm(task,x)*bayesrule()

def parzenm(task=None,x=None,w=None):
    "Parzen density estimate per class"
    if not isinstance(task,basestring):
        out = prmapping(parzenm,task,x)
        return out
    if (task=='untrained'):
        # just return the name, and hyperparameters
        if x is None:
            x = [1]
        return 'Parzen density', x
    elif (task=="train"):
        # we only need to store the data
        # store the parameters, and labels:
        return x,x.lablist()
    elif (task=="eval"):
        # we are applying to new data
        W = w.data
        nrcl = len(w.labels)
        h = w.hyperparam[0]
        n,dim = x.shape
        Z = numpy.sqrt(2*numpy.pi)*h**dim
        out = numpy.zeros((n,nrcl))
        for i in range(nrcl):
            xi = W.seldat(i)
            D = sqeucldist(+x,+xi)
            out[:,i] = numpy.sum( numpy.exp(-D/(2*h*h)), axis=1)/Z
        return out
    else:
        print(task)
        raise ValueError('This task is *not* defined for parzenm.')

def parzenc(task=None,x=None,w=None):
    return parzenm(task,x,w)*bayesrule()

def naivebm(task=None,x=None,w=None):
    "Naive Bayes density estimate"
    if not isinstance(task,basestring):
        out = prmapping(naivebm,task,x)
        return out
    if (task=='untrained'):
        # just return the name, and hyperparameters
        if x is None:
            x = [gaussm()]
        return 'Naive Bayes density', x
    elif (task=="train"):
        # we only to estimate the densities for each feature:
        c = x.shape[1]
        f = []
        for i in range(c):
            u = copy.deepcopy(w[0])
            f.append(x[:,i:(i+1)]*u)
        # store the parameters, and labels:
        return f,x.lablist()
    elif (task=="eval"):
        # we are applying to new data
        W = w.data
        nrcl = len(w.labels)
        nrfeat = len(W)
        if not isinstance(x,prdataset):
            x = prdataset(x)
        n,dim = x.shape
        out = numpy.ones((n,nrcl))
        for i in range(nrfeat):
            out *= +(x[:,i:(i+1)]*W[i])
        return out
    else:
        print(task)
        raise ValueError('This task is *not* defined for naivebm.')

def naivebc(task=None,x=None,w=None):
    return naivebm(task,x,w)*bayesrule()

def baggingc(task=None,x=None,w=None):
    "Bagging"
    if not isinstance(task,basestring):
        out = prmapping(baggingc,task,x)
        return out
    if (task=='untrained'):
        # just return the name, and hyperparameters
        if x is None:
            x = (nmc,100)
        return 'Baggingc', x
    elif (task=="train"):
        # we are going to train the mapping
        clsz = x.classsizes()
        f = []
        for t in range(w[1]):
            xtr,xtst = gendat(x,clsz) # just a simple bootstrap
            #DXD we could do feature subsampling as well..
            u = copy.deepcopy(w[0])
            f.append(u(xtr))
        # store the parameters, and labels:
        return (f),x.lablist()
    elif (task=="eval"):
        # we are applying to new data
        X = +x
        n = X.shape[0]
        W = w.data
        T = len(W)  # nr of aggregated classifiers
        c = len(W[0].labels) # nr of classes
        out = numpy.zeros((n,c))
        J = range(n)
        for i in range(T):
            # do majority voting:
            pred = W[i](X)
            I = numpy.argmax(+pred,axis=1)
            out[J,I] += 1 

        return out
    else:
        print(task)
        raise ValueError('This task is *not* defined for baggingc.')

def stumpc(task=None,x=None,w=None):
    "Decision stump classifier"
    if not isinstance(task,basestring):
        out = prmapping(stumpc,task,x)
        return out
    if (task=='untrained'):
        # just return the name, and hyperparameters
        return 'Decision stump', ()
    elif (task=="train"):
        # we are going to train the mapping
        if (x.nrclasses() != 2):
            raise ValueError('Stumpc can only deal with 2 classes.')
        # allow weights:
        n,dim = x.shape
        w = x.weights
        if (len(w)==0):
            w = numpy.ones((n,1))
        w /= numpy.sum(w)
        # initialise:
        X = +x
        y = x.signlab(posclass=0)   # make labels +1/-1
        wy = w*y
        Nplus = numpy.sum(wy[y>0])
        Nmin = -numpy.sum(wy[y<0])
        bestfeat,bestthres,bestsign,besterr = 0,0,0,10.

        # Do an exhaustive search over all features:
        for f in range(dim):
            I = numpy.argsort(X[:,f])
            sortlab = wy[I]
            sumlab = numpy.cumsum(numpy.vstack((Nmin,sortlab)))
            J = numpy.argmin(sumlab)
            if (sumlab[J]<besterr):
                besterr = sumlab[J]
                bestfeat = f
                if (J==0):
                    bestthres = X[0,f] - 1e-6
                elif (J==n):
                    bestthres = X[n,f] + 1e-6
                else:
                    bestthres = (X[I[J],f]+X[I[J-1],f])/2.
                #print("Better feature %d, th=%f, sg=+, has error %f"%(f,bestthres,sumlab[J]))
                bestsign = +1
            sumlab = numpy.cumsum(numpy.vstack((Nplus,-sortlab)))
            J = numpy.argmin(sumlab)
            if (sumlab[J]<besterr):
                besterr = sumlab[J]
                bestfeat = f
                if (J==0):
                    bestthres = X[0,f] - 1e-6
                elif (J==n):
                    bestthres = X[n-1,f] + 1e-6
                else:
                    bestthres = (X[I[J],f]+X[I[J-1],f])/2.
                #print("Better feature %d, th=%f, sg=-, has error %f"%(f,bestthres,sumlab[J]))
                bestsign = -1

        # store the parameters, and labels:
        #print("In training the decision stump:")
        #print x.lablist()
        ll = x.lablist()
        return (bestfeat,bestthres,bestsign),ll
    elif (task=="eval"):
        # we are applying to new data
        W = w.data
        X = +x
        if (W[2]>0):
            out = (X[:,int(W[0])] >= W[1])*1.
        else:
            out = (X[:,int(W[0])] < W[1])*1.
        # How the F*** can I force numpy to behave?!:
        if (len(out.shape)==1):
            out = out[:,numpy.newaxis]
        if (x.shape[0]>1) and (out.shape[0]==1):
            out = out[:,numpy.newaxis]  # GRRR
        dat = numpy.hstack((out,1.-out))
        if (x.shape[0]==1) and (dat.shape[0]>1):
            dat = dat[numpy.newaxis,:]  # GRRRR
        return dat
    else:
        print(task)
        raise ValueError('This task is *not* defined for stumpc.')

def adaboostc(task=None,x=None,w=None):
    "AdaBoost classifier"
    if not isinstance(task,basestring):
        out = prmapping(adaboostc,task,x)
        return out
    if (task=='untrained'):
        # just return the name, and hyperparameters
        if x is None:
            x = [100]
        return 'AdaBoost', x
    elif (task=="train"):
        # we are going to train the mapping
        # setup vars
        T = w[0]
        N = x.shape[0]
        # find the 'first' class in the dataset: that will be the first output
        # of the decision stump, and that will be easy to retrieve. We will make
        # that the positive class
        Iclass1 = x.lablist()[0]
        y = 1 - 2*x.nlab()
        h = numpy.zeros((T,3))

        tmp = prmapping(stumpc)
        w = numpy.ones((N,1))

        alpha = numpy.zeros((T,1))
        for t in range(T):
            #print("Iteration %d in Adaboost" % t)
            x.weights = w
            tmp.data, ll = stumpc('train',x)
            h[t,0],h[t,1],h[t,2] = tmp.data
            #print('  -> Feature %d, with threshold %f and sign %d'% (h[t,0],h[t,1],h[t,2]))
            pred = stumpc('eval',x, tmp)
            # nasty nasty trick [:,:1]
            pred = 2*pred[:,:1]-1
            err = numpy.sum(w*(pred!=y))
            #print('Error is %f' % err)
            if (err==0):
                print("Stop it!")
                perfecth = h[t,:]
                return (perfecth,1.),x.lablist()
            alpha[t] = numpy.log(numpy.sum(w)/err - 1.)/2.
            w *= numpy.exp(-alpha[t]*y*pred)
        
        # store the parameters, and labels:
        return (h,alpha),x.lablist()
    elif (task=="eval"):
        # we are applying to new data
        W = w.data
        N = x.shape[0]
        pred = numpy.zeros((N,1))
        tmp = prmapping(stumpc)
        for t in range(len(W[1])):
            #print("Eval hypothesis %d in Adaboost" % t)
            tmp.data = W[0][t]
            out = stumpc('eval',x,tmp)
            out2 = 2.*(+out[:,:1]) - 1.
            pred += W[1][t]*out2
        out = numpy.hstack((pred,-pred))
        return out
    else:
        print(task)
        raise ValueError('This task is *not* defined for adaboostc.')

def svc(task=None,x=None,w=None):
    "Support vector classifier"
    if not isinstance(task,basestring):
        out = prmapping(svc,task,x)
        return out
    if (task=='untrained'):
        # just return the name, and hyperparameters
        if x is None:
            kernel = 'rbf'
            x = 1.
            C = 1.
        else:
            kernel = x[0]
            C = x[2]
            x = x[1]
        if (kernel=='linear'):
            clf = svm.SVC(kernel='linear',degree=x,C=C,probability=True)
        else:
            clf = svm.SVC(kernel='rbf',gamma=x,C=C,probability=True)
        return 'Support vector classifier', clf
    elif (task=="train"):
        # we are going to train the mapping
        X = +x
        y = numpy.ravel(x.labels)
        clf = copy.deepcopy(w)
        clf.fit(X,y)
        return clf,x.lablist()
    elif (task=="eval"):
        # we are applying to new data
        clf = w.data
        pred = clf.decision_function(+x) 
        if (len(pred.shape)==1): # oh boy oh boy, we are in trouble
            pred = pred[:,numpy.newaxis]
            pred = numpy.hstack((-pred,pred)) # sigh
        return pred
    else:
        print(task)
        raise ValueError('This task is *not* defined for svc.')

def dectreec(task=None,x=None,w=None):
    "Decision tree classifier"
    if not isinstance(task,basestring):
        out = prmapping(dectreec,task,x)
        return out
    if (task=='untrained'):
        # just return the name, and hyperparameters
        if x is None:
            max_d = None
        else:
            max_d = x[0]
        clf = tree.DecisionTreeClassifier(max_depth=max_d)
        return 'Decision tree', clf
    elif (task=="train"):
        # we are going to train the mapping
        X = +x
        y = numpy.ravel(x.labels)
        clf = copy.deepcopy(w)
        clf.fit(X,y)
        return clf,x.lablist()
    elif (task=="eval"):
        # we are applying to new data
        clf = w.data
        pred = clf.predict_proba(+x) 
        if (len(pred.shape)==1): # oh boy oh boy, we are in trouble
            pred = pred[:,numpy.newaxis]
            pred = numpy.hstack((-pred,pred)) # sigh
        return pred
    else:
        print(task)
        raise ValueError('This task is *not* defined for dectreec.')

def pcam(task=None,x=None,w=None):
    "Principal Component Analysis "
    if not isinstance(task,basestring):
        out = prmapping(pcam,task,x)
        return out
    if (task=='untrained'):
        # just return the name, and hyperparameters
        return 'PCA', x
    elif (task=="train"):
        # we are going to train the mapping
        if w is None: # no dimensionality given: use all
            pcadim = x.shape[1]
        elif (w<1):
            pcadim = int(w*x.shape[1])
        else:
            pcadim = w
        # get eigenvalues and eigenvectors
        C = numpy.cov(+x,rowvar=False)
        l,v = numpy.linalg.eig(C)
        # sort it:
        I = numpy.argsort(l)
        I = I[:pcadim]
        l = l[I]
        v = v[:,I]
        featlab = range(pcadim)
        # store the parameters, and labels:
        return v,featlab
    elif (task=="eval"):
        # we are applying to new data
        dat = +x
        return dat.dot(w.data)
    else:
        print(task)
        raise ValueError('This task is *not* defined for pcam.')

def sqeucldist(a,b):
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

def cleval(a,u,trainsize=[2,3,5,10,20,30],nrreps=3):
    nrcl = a.nrclasses()
    clsz = a.classsizes()
    if (numpy.max(trainsize)>=numpy.min(clsz)):
        raise ValueError('Not enough objects per class available.')
    N = len(trainsize)
    err = numpy.zeros((N,nrreps))
    err_app = numpy.zeros((N,nrreps))
    for f in range(nrreps):
        print("Iteration %d." % f)
        for i in range(N):
            sz = trainsize[i]*numpy.ones((1,nrcl))
            x,z = gendat(a, sz[0],seed=f)
            w = x*u
            err[i,f] = z*w*testc()
            err_app[i,f] = x*w*testc()
    # show it?
    h = plt.errorbar(trainsize,numpy.mean(err,axis=1),numpy.std(err,axis=1),\
            label=u.name)
    thiscolor = h[0].get_color()
    plt.errorbar(trainsize,numpy.mean(err_app,axis=1),numpy.std(err_app,axis=1),\
            fmt='--',color=thiscolor)
    plt.xlabel('Nr. train objects per class')
    plt.ylabel('Error')
    plt.title('Learning curve %s' % a.name)
    return err, err_app

def clevalf(a,u,trainsize=0.6,nrreps=5):
    dim = a.shape[1]
    err = numpy.zeros((dim,nrreps))
    err_app = numpy.zeros((dim,nrreps))
    for f in range(nrreps):
        for i in range(1,dim):
            x,z = gendat(a[:,:i], trainsize,seed=f)
            w = x*u
            err[i,f] = z*w*testc()
            err_app[i,f] = x*w*testc()
    # show it?
    h = plt.errorbar(range(dim),numpy.mean(err,axis=1),numpy.std(err,axis=1),\
            label=u.name)
    thiscolor = h[0].get_color()
    plt.errorbar(range(dim),numpy.mean(err_app,axis=1),numpy.std(err_app,axis=1),\
            fmt='--',color=thiscolor)
    plt.xlabel('Feature dimensionality')
    plt.ylabel('Error')
    plt.title('Feature curve %s' % a.name)
    return err, err_app



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
        newm.shape[0] = f.dim
        newm.shape[1] = 1
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

