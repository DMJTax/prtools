"""
Pattern Recognition Mapping class

Should provide a uniform and consistent way of defining transformation for datasets.

You can train a mapping directly on a dataset:
    w = scalem(a)
To apply it to a new dataset:
    b = a*w
You can also start with an untrained mapping, and then train it:
    u = scalem()
    w = a*u
You can also concatenate mappings into long processing pipelines:
    u = scalem()*pcam()*scalem()*nmc()
    w = a*u

When you dislike this 'Matlab-style' call, you can also do it in the
Pythonic way:
    w = scalem()
    w.train(a)
    b = w.eval(a)

You can define your own prmapping by defining one function that is able
to perform three tasks: initialisation, training and evaluation. This is
indicated by the first input parameter 'task'. A basic mapping looks
like:

def scalem(task=None,x=None,w=None):
    if not isinstance(task,str):
        # return a prmapping:
        return prmapping(scalem,task,x)
    if (task=='init'):
        # x now contains the provided hyperparameters
        # return the name of the mapping, and hyperparameters:
        return 'Scalem', x
    elif (task=='train'):
        # we are going to train the mapping on data x.
        # The hyperparameters are available in input parameter w
        mn = numpy.mean(+x,axis=0)
        sc = numpy.std(+x,axis=0)
        # return the trained parameters, and feature labels:
        return (mn,sc), x.featlab
    elif (task=='eval'):
        # apply the mapping to new data x. The full mapping is available
        # in w.
        W = w.data   # get the parameters out of the mapping
        x = +x-W[0]
        x = +x/W[1]
        return x


"""

import numpy
import copy
import matplotlib.pyplot as plt
from sklearn import linear_model

from .dataset import prdataset

# === prmapping ============================================
class prmapping(object):
    "Prmapping in Python"

    def __init__(self,mapping_func,x=[],hyperp=None):
        # exception: when only hyperp are given
        #if (not isinstance(x,prdataset)) and (not hyperp): 
        if (not isinstance(x,prdataset)) and (hyperp is None): 
            hyperp = x
            x = []
        self.mapping_func = mapping_func
        self.mapping_type = 'untrained'
        self.name, self.hyperparam = self.mapping_func('init',hyperp)
        self.data = ()    # all trained parameters and settings
        self.targets = () # feature names of the outputted prdataset
        self.shape = [0,0]
        self.user = []
        if isinstance(x,prdataset):
            self = self.train(copy.deepcopy(x))

    def __repr__(self):
        return "prmapping("+self.name+","+self.mapping_type+")"
    def __str__(self):
        outstr = ""
        if (len(self.name)>0):
            outstr = "%s, " % self.name
        if (self.mapping_type=='untrained'):
            outstr += "untrained mapping"
        elif (self.mapping_type=='trained'):
            outstr += "%d to %d trained mapping" % (self.shape[0],self.shape[1])
        else:
            raise ValueError('Mapping type is not defined.')
        return outstr

    def shape(self,I=None):
        if I is not None:
            if (I==0):
                return self.shape[0]
            elif (I==1):
                return self.shape[1]
            else:
                raise ValueError('Only dim=0,1 are possible.')
        else:
            return self.shape

    def init(self,mappingfunc,**kwargs):
        self.mapping_func = mappingfunc
        self.mapping_type = 'untrained'
        self.hyperparam = kwargs
        self.name,self.hyperparam = self.mapping_func('init',kwargs)
        return self

    def train(self,x,args=None):
        # train
        if (self.mapping_type=='trained'):
            raise ValueError('The mapping is already trained and will be retrained.')
        # maybe the supplied parameters overrule the stored ones:
        if args is not None:
            self.hyperparam = args
        #if (len(self.hyperparam)==0):
        #    self.data,self.targets = self.mapping_func('train',x)
        #else:
        self.data,self.targets = self.mapping_func('train',x,self.hyperparam)

        self.mapping_type = 'trained'
        # set the input and output sizes 
        if (hasattr(x,'shape')):  # combiners do not eat datasets
            self.shape[0] = x.shape[1]
            # and the output size?
            #xx = +x[:1,:]   # hmmm??
            xx = x[:1,:]   # or??
            out = self.mapping_func('eval',xx,self)
            self.shape[1] = out.shape[1]
        return self

    def eval(self,x):
        # evaluate
        if (self.mapping_type=='untrained'):
            raise ValueError('The mapping is not trained and cannot be evaluated.')
        # not a good idea to supply the true targets?
        # but it is needed for testc!
        newx = copy.deepcopy(x)
        out = self.mapping_func('eval',newx,self)
        if ((len(self.targets)>0) and (out.shape[1] != len(self.targets))):
            print(out.shape)
            print(self.targets)
            print(len(self.targets))
            raise ValueError('Output of mapping does not match number of targets.')
        #if (len(self.targets)>0):
        #    if not isinstance(x,prdataset):
        #        newx = prdataset(newx)
        #    newx.featlab = self.targets
        ## tricky: if I don't input a prdataset, should I return one?
        ##  For parallelm I need a prdataset as output!
        ##if isinstance(x,prdataset) and (len(self.targets)>0):
        #if ((len(self.targets)>0) and (not isinstance(out,prdataset))):
        #    newx.setdata(+out)
        #else:
        #    newx = out
        #return newx

        # Ok, that was a mess, again:
        if (len(self.targets)>0):  # we output a prdataset
            if (isinstance(out,prdataset)): # check if we can set featlab:
                if (len(self.targets) != out.shape[1]):
                    raise ValueError('Nr output features does not match nr of feature labels.')
                out.featlab = self.targets
                return out
            else:   # convert to prdataset, keeping as much of the
                # original dataset x as possible
                if (isinstance(x,prdataset)):
                    newx.setdata(+out)
                else:
                    newx = prdataset(out)
                newx.featlab = self.targets
                return newx
        else:  # just return what the mapping did
            return out


    def __call__(self,x):
        if (self.mapping_type=='untrained'):
            # train
            out = self.train(x)
            return out
        elif (self.mapping_type=='trained'):
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
            out = sequentialm([leftm,rightm])
            return out
        else:
            raise ValueError('Prmapping times something not defined.')

    def __rmul__(self,other):
        if (isinstance(other,prdataset)):
            #print('prmapping multiplication with left')
            #print('   self='+str(self))
            #print('   other='+str(other))
            newself = copy.deepcopy(self)
            return newself(other)
        else:
            return NotImplemented

    def float(self):
        # Print the values of w and the intercept
        if(isinstance(self.data, numpy.ndarray)):
            return self.data.copy()
        elif(isinstance(self.data, linear_model.Lasso)):
            coef = self.data.coef_.reshape(self.data.coef_.size, 1)
            intercept = self.data.intercept_
            return numpy.vstack((coef, intercept))
        else:
            raise ValueError('Mapping has no way to print values (yet)')
    
    def __pos__(self):
        return self.float()


def sequentialm(task=None,x=None,w=None):
    "Sequential mapping"
    if not isinstance(task,str):
        # we should have gotten a list of two prmappings
        if not isinstance(task,list):
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
            if (len(newm[1].targets)==0) and (newm[1].shape[1]==0):
                # the second mapping does not have targets and sizes defined:
                w.shape[1] = newm[0].shape[1]
                w.targets = newm[0].targets
            else:
                w.shape[1] = newm[1].shape[1]
                w.targets = newm[1].targets
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
    if (task=='init'):
        # just return the name, and hyperparameters
        if x is None:
            mapname = 'Sequential'
        else:
            mapname = x[0].name+'+'+x[1].name
        return mapname, x
    elif (task=='train'):
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
        # fix the targets:
        if (len(u[1].targets)==0) and (u[1].shape[1]==0):
            # the second mapping does not have targets and sizes defined:
            targets = u[0].targets
        else:
            targets = u[1].targets
        return u, targets
    elif (task=='eval'):
        # we are applying to new data
        W = w.data   # get the parameters out
        return W[1](W[0](x))
    else:
        print(task)
        raise ValueError('This task is *not* defined for sequentialm.')

def parallelm(task=None,x=None,w=None):
    "Parallel mapping"
    if not isinstance(task,str):
        # we should have gotten a list of prmappings
        if not isinstance(task,list):
            raise ValueError('Parallel map expects a list of prmappings.')
        alltrained = True
        allshapeout = 0
        alltargets = numpy.ndarray((0,))
        for thismap in task:
            if not isinstance(thismap,prmapping):
                raise ValueError('Parallel map expects a list of prmappings.')
            alltrained = alltrained and (thismap.mapping_type=='trained')
            allshapeout += thismap.shape[1]
            alltargets = numpy.concatenate((alltargets,thismap.targets))

        newm = copy.deepcopy(task)
        # if all mappings are trained, the parallel mapping is also trained!
        # (this is an exception to the standard, where you need data in
        # order to train a prmapping)
        if alltrained:
            # do the constructor, but make sure that the hyperparameters
            # are None:
            w = prmapping(parallelm,None)
            w.data = newm
            # for the input size, just copy the shape of the first (DXD:
            # check !!)
            w.shape[0] = newm[0].shape[0]
            w.shape[1] = allshapeout
            w.targets = alltargets
            w.mapping_type = 'trained'
        else:
            if x is None:
                w = prmapping(parallelm,newm,x)
            else:
                newx = copy.deepcopy(x)
                w = prmapping(parallelm,newm,newx)
        w.name = "Parallel"
        return w
    if (task=='init'):
        # just return the name, and hyperparameters
        mapname = 'Parallel'
        return mapname, x
    elif (task=='train'):
        # we are going to train all the mappings
        u = copy.deepcopy(w)  # I hate Python..
        allshapeout = 0
        alltargets = numpy.ndarray((0,))
        for i in range(len(w)):
            wi = copy.deepcopy(w[i])  # I deeply hate Python..
            x1 = copy.deepcopy(x) # Did I say that I hate Python??
            if (wi.mapping_type=='untrained'):
                #print('train mapping %d...'%i)
                u[i] = wi.train(x1)
            # output size:
            allshapeout += u[i].shape[1]
            # collect the targets:
            alltargets = numpy.concatenate((alltargets,u[i].targets))
        return u, alltargets
    elif (task=='eval'):
        # we are applying to new data
        W = w.data   # get the parameters out
        out = W[0](x)
        for i in range(1,len(W)):
            # sigh, concatenation is different for matrices and datasets
            if (isinstance(out,prdataset)):
                out = out.concatenate(W[i](x),axis=1)
            else:
                out = numpy.concatenate((out,W[i](x)),axis=1)
        return out
    else:
        print(task)
        raise ValueError('This task is *not* defined for parallelm.')

def fixedcc(task=None,x=None,w=None):
    """
    Fixed classifier combiner
    Combine the output of a parallel combiner using
    TASK = 'mean'    mean combiner
           'prod'    product combiner
           'min'     minimum combiner
           'max'     maximum combiner
           'pow',P   powered combination:   = (sum_i (p_i^P) )^(1/P)

    a = gendatb()
    u = parallelm([nmc(),ldc(),qdc()])*fixedcc('mean')
    u = parallelm([nmc(),ldc(),qdc()])*fixedcc('pow',3)
    w = a*u
    """
    if (task in set(['mean','prod','min','max','pow'])):
        if (task=='pow'):
            x = [task,x]
        else:
            x = [task]
        task = None
    if not isinstance(task,str):
        out = prmapping(fixedcc,task,x)
        out.mapping_type = "trained"
        # DXD this is a hack: avoid copying the targets of the previous
        # mapping (probably a parallelm) to the output
        out.shape[1]=-1  
        if task is not None:
            out = out(task)
        return out
    if (task=='init'):
        # just return the name, and hyperparameters
        if x is None:
            x = ['mean']
        return 'Classifier combiner', x
    elif (task=='train'):
        # nothing to train
        return None,()
    elif (task=='eval'):
        # we are classifying new data
        #print(+x)
        # we need a dataset for the feature labels
        outputdataset = True
        if not isinstance(x,prdataset):
            outputdataset = False
            x = prdataset(x)
        # get the feature labels
        (flist,I) = numpy.unique(x.featlab,return_inverse=True)
        out = numpy.zeros((x.shape[0],len(flist)))
        # run over the sets:
        for i in range(len(flist)):
            J = numpy.where(I==i)[0]
            C = J.shape[0]
            xi = x[:,J]
            if (w.hyperparam[0]=='mean'):
                # mean combination:
                out[:,i:(i+1)] = numpy.mean(+xi,axis=1,keepdims=True)
            elif (w.hyperparam[0]=='prod'):
                # product combination:
                out[:,i:(i+1)] = numpy.prod(+xi,axis=1,keepdims=True)
            elif (w.hyperparam[0]=='min'):
                # mean combination:
                out[:,i:(i+1)] = numpy.min(+xi,axis=1,keepdims=True)
            elif (w.hyperparam[0]=='max'):
                # max combination:
                out[:,i:(i+1)] = numpy.max(+xi,axis=1,keepdims=True)
            elif (w.hyperparam[0]=='pow'):
                # powered combination:
                out[:,i:(i+1)] = numpy.power( \
                        numpy.mean((+xi)**w.hyperparam[1],axis=1,keepdims=True),\
                        1/w.hyperparam[1])
            else:
                raise ValueError("Combining '%s' is *not* defined for\
                        fixedcc."%w.hyperparam)

        if outputdataset:
            x = x.setdata(out)
            x.featlab = flist
            return x
        else:
            return out
    else:
        raise ValueError("Task '%s' is *not* defined for fixedcc."%task)



# === useful functions =====================================

def plotc(f,levels=[0.0],colors=None,gridsize = 30):
    """
    Plot decision boundary

        plotc(W)

    Plot the decision boundary of trained classifier W

    Example:
    a = gendatb(100)
    w = parzenc(a)
    scatterd(a)
    plotc(w)
    """
    ax = plt.gca()
    if colors is None:
        colors = next(ax._get_lines.prop_cycler)['color']
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
    if (f.shape[0]>2):
        print("PLOTC: Mapping is >2D, set remaining inputs to 0.")
        X2 = numpy.zeros((gridsize*gridsize,f.shape[0]-2))
        dat = numpy.hstack((dat,X2))

    out = +f(dat)
    for i in range(1,out.shape[1]):
        otherout = copy.deepcopy(out)
        otherout[:,i] = -numpy.inf
        z = out[:,i] - numpy.amax(otherout,axis=1)
        z.shape = (gridsize,gridsize)
        CS = plt.contour(x,y,z,levels,colors=colors)
    CS.collections[0].set_label(f.name)

def plotm(f,nrlevels=10,colors=None,gridsize = 30):
    """
    Plot mapping outputs

        plotm(W)

    Plot the output of mapping W.

    Example:
    a = gendatb(100)
    w = parzenm(a)
    scatterd(a)
    plotm(w)
    """
    ax = plt.gca()
    if colors is None:
        colors = next(ax._get_lines.prop_cycler)['color']
    xl = ax.get_xlim()
    dx = (xl[1]-xl[0])/(gridsize-1)
    x = numpy.arange(xl[0],xl[1]+0.01*dx,dx)

    if (f.shape[0]==1):
        x.shape = (gridsize,1)
        plt.plot(x,+f(x))
        return

    yl = ax.get_ylim()
    dy = (yl[1]-yl[0])/(gridsize-1)
    y = numpy.arange(yl[0],yl[1]+0.01*dy,dy)
    X0,X1 = numpy.meshgrid(x,y)
    X0.shape = (gridsize*gridsize, 1)
    X1.shape = (gridsize*gridsize, 1)
    dat = numpy.hstack((X0,X1))
    if (f.shape[0]>2):
        print("PLOTM: Mapping is >2D, set remaining inputs to 0.")
        X2 = numpy.zeros((gridsize*gridsize,f.shape[0]-2))
        dat = numpy.hstack((dat,X2))

    out = +f(dat)
    for i in range(out.shape[1]):
        z = out[:,i]
        levels = numpy.linspace(numpy.min(z),numpy.max(z),nrlevels)
        z.shape = (gridsize,gridsize)
        CS = plt.contour(x,y,z,levels,colors=colors)
    CS.collections[0].set_label(f.name)

def plotr(f,color=None,gridsize=100):
    """
    Plot regression outputs

        plotr(W)

    Plot the output of regressor W.

    Example:
    a = gendatsinc(100)
    w = kernelr(a,0.5)
    scatterr(a)
    plotr(w)
    """
    ax = plt.gca()
    if color is None:
        color = next(ax._get_lines.prop_cycler)['color']
    xl = ax.get_xlim()
    dx = (xl[1]-xl[0])/(gridsize-1)
    if (f.shape[0]==1):
        x = numpy.arange(xl[0],xl[1]+0.01*dx,dx)
        xx = prdataset(x[:,numpy.newaxis])
        y = +f(xx)
        plt.plot(x,y,color=color)
    elif (f.shape[0]==2):
        yl = ax.get_ylim()
        dy = (yl[1]-yl[0])/(gridsize-1)
        x = numpy.arange(xl[0],xl[1]+0.1*dx,dx)
        y = numpy.arange(yl[0],yl[1]+0.1*dy,dy)
        X0,X1 = numpy.meshgrid(x,y)
        X0.shape = (gridsize*gridsize, 1)
        X1.shape = (gridsize*gridsize, 1)
        dat = numpy.hstack((X0,X1))
        X0.shape = (gridsize,gridsize)
        X1.shape = (gridsize,gridsize)
        out = +f(dat)
        out.shape = (gridsize,gridsize)
        ax.plot_wireframe(X0,X1,out)
