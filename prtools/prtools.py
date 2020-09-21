"""
Prtools for Python
==================

This module implements a general class of dataset and mapping, inspired
by the original Matlab toolbox Prtools. It should abstract away the
details of different classifiers, regressors, data-preprocessings and
error evaluations, and allows for easy visualisation, comparison and
combination of different methods.

A (small) subset of the methods are:
    nmc       nearest mean classifier
    ldc       linear discriminant classifier
    qdc       quadratic discriminant classifier
    parzenc   Parzen classifier
    knnc      k-Nearest neighbor classifier
    mogc      mixture of Gaussians classifier
    ababoostc AdaBoost
    svc       support vector classifier
    loglc     logistic classifier
    dectreec  decision tree classifier
    lassoc    logistic classifier

    linearr   linear regression
    ridger    ridgeregression
    lassor    LASSO

    kmeans    K-Means clustering
    hclust    Hierarchical Clustering clustering
    plotdg    Plot dendrogram
    dbi       Davies-Bouldin index

    labeld    labeling objects
    testc     test classifier
    testr     test regressor
    cleval    classifier evaluation
    prcrossval  crossvalidation

    scalem    scale mapping 
    proxm     proximity mapping

    pcam      principal component analysis
    icam      independent component analysis
    fisherm   Fisher mapping
    
    feateval  feature evaluation
    featseli  individual feature selection
    featsetf  sequential forward feature selection
    featselb  sequential backward feature selection

A (small) subset of datasets:
    gendatb   banana-shaped dataset
    gendats   simple dataset
    gendatd   difficult dataset
    gendath   Highleyman dataset
    gendatsinc   simple regression dataset
    boomerangs   3D 2-class problem
"""

from prtools import *
from sklearn import svm
from sklearn import linear_model
from sklearn import tree

from sklearn.manifold import LocallyLinearEmbedding, Isomap


# === mappings ===============================

def scalem(task=None,x=None,w=None):
    """
    Scale mapping

        W = scalem(A,STYPE)
        W = A*scalem(STYPE)

    Scales the features of dataset A. The scaling is done according to
    STYPE:
    'unitvar'  (default) mean is set to 0, variance is set to 1
    '01'       minimum is set to 0, maximum is set to 1

    Example:
    >> w = scalem(a,'unitvar')
    >> b = a*w
    """
    # special case:
    if (task in set(['unitvar','01'])):
            x = task
            task = None
    if not isinstance(task,str): # direct call: return a mapping
        return prmapping(scalem,task,x)
    if (task=='init'):
        if x is None:                # default scaling
            x = {'scaling':'unitvar'}
        if not isinstance(x,dict):   # always cast to dict
            x = {'scaling':x}
        # return the name, and hyperparameters:
        return 'Scalem', x
    elif (task=='train'):
        # we are going to train the mapping on data x
        # The hyperparameters are available in input parameter w
        stype = w['scaling']
        if (stype=='unitvar'):
            mn = numpy.mean(+x,axis=0)
            sc = numpy.std(+x,axis=0)
            return (mn,sc), x.featlab
        elif (stype=='01'):
            minx = numpy.min(+x,axis=0)
            sc = numpy.max(+x,axis=0) - minx
            return (minx,sc), x.featlab
        else:
            raise ValueError("Scaling '%s' is not defined."%stype)
    elif (task=='eval'):
        # we are applying to new data
        x = +x - w.data[0]
        x = +x / w.data[1]
        return x
    else:
        raise ValueError("Task '%s' is *not* defined for scalem."%task)

def proxm(task=None,x=None,w=None):
    """
    Proximity mapping

        W = proxm(A,(K,K_par))
        W = A*proxm([],(K,K_par))

    Fit a proximity/kernel mapping on dataset A. The kernel is defined
    by K and its parameter K_par. The available proximities are:
        'eucl'      Euclidean distance
        'city'      City-block distance
        'rbf'       Radial basis function kernel with width K_par

    Example:
    >> u = proxm(('rbf',4))*nmc()
    >> w = a*u   # will approximate a Parzen classifier

    """
    if (task in set(['eucl','city','rbf'])):
            x = task
            task = None
    if not isinstance(task,str):
        # A direct call to proxm, refer back to prmapping:
        return prmapping(proxm,task,x)
    if (task=='init'):
        # just return the name, and hyperparams
        return 'Proxm',x
    elif (task=='train'):
        # we only need to store the representation set
        if (isinstance(x,prdataset)):
            R = +x
        else:
            R = numpy.copy(x)
        # the feature labels:
        featlab = numpy.arange(R.shape[0])
        if (w[0]=='eucl'):
            return ('eucl',R), featlab
        elif (w[0]=='city'):
            return ('city',R), featlab
        elif (w[0]=='rbf'):
            return ('rbf',R,w[1]), featlab
        else:
            raise ValueError("Proxm type '%s' not defined"%w[0])
    elif (task=='eval'):
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
            raise ValueError("Proxm type '%s' not defined"%W[0])
        return D
    else:
        raise ValueError("Task '%s' is *not* defined for proxm."%task)


def softmax(task=None,x=None,w=None):
    """
    Softmax mapping

         W = softmax(A)
         W = A*softmax()

    Compute the softmax of each row in A, by exponentiating each element
    in the row, summing them, and dividing each element in the row by
    this sum:
      A_new(i,j) = exp(A(i,j)) / sum_k exp(A(i,k))

    Example:
    >> a = gendatb
    >> w = nmc(a)
    >> conf = +(a*w*softmax())
    """
    if not isinstance(task,str):
        out = prmapping(softmax)
        out.mapping_type = "trained"
        if task is not None:
            out = out(task)
        return out
    if (task=='init'):
        # just return the name, and hyperparameters
        return 'Softmax', ()
    elif (task=='train'):
        print("Softmax: We cannot train the softmax mapping.")
        return 0, x.featlab
    elif (task=='eval'):
        # we are applying to new data
        dat = numpy.exp(+x)
        sumx = numpy.sum(dat,axis=1,keepdims=True)
        return dat/sumx
    else:
        raise ValueError("Task '%s' is *not* defined for softmax."%task)

def classc(task=None,x=None,w=None):
    """
    Classifier confidence mapping

         W = classc(A)
         W = A*classc()

    Normalize the output of a classifier such that an approximate
    confidence value is obtained. Normalisation is done by just summing
    the values in each row of A, and dividing each element of this row
    by the sum. It is therefore assumed that all values are positive.
    """
    if not isinstance(task,str):
        out = prmapping(classc)
        out.mapping_type = "trained"
        if task is not None:
            out = out(task)
        return out
    if (task=='init'):
        # just return the name, and hyperparameters
        return 'Classc', ()
    elif (task=='train'):
        print("Classc: We cannot train the classc mapping.")
        return 0, x.featlab
    elif (task=='eval'):
        # we are applying to new data
        if (numpy.any(+x<0.)):
            print('classc(): Suspicious negative values in Classc.')
        sumx = numpy.sum(+x,axis=1,keepdims=True)
        if isinstance(x,prdataset):
            x.setdata( +x/sumx )
        else:
            x = x/sumx
        return x
    else:
        raise ValueError("Task '%s' is *not* defined for classc."%task)


def labeld(task=None,x=None,w=None):
    """
    Label mapping
    
           LAB = labeld(A)
           LAB = A*labeld()

    Compute the output labels from a (classified) dataset A.

    Example:
    >> a = gendatb()
    >> lab = a*ldc(a)*labeld
    >> print(lab)
    """
    if not isinstance(task,str):
        out = prmapping(labeld)
        out.mapping_type = "trained"
        if task is not None:
            out = out(task)
        return out
    if (task=='init'):
        # just return the name, and hyperparameters
        return 'Labeld', ()
    elif (task=='train'):
        print("Labeld: We cannot train the label mapping.")
        return 0, x.featlab
    elif (task=='eval'):
        # we are applying to new data
        I = numpy.argmax(+x,axis=1)
        n = x.shape[0]
        # complex way to deal with both numeric and string labels:
        out = []
        for i in range(n):
            out.append(x.featlab[I[i]])
        out = numpy.array(out)
        out = out[:,numpy.newaxis]
        return out
    else:
        raise ValueError("Task '%s' is *not* defined for labeld."%task)

def testc(task=None,x=None,w=None):
    """
    Test classifier

          E = testc(A)
          E = A*testc()
    Compute the error on dataset A.

    Example:
    >> A = gendatb()
    >> W = ldc(A)
    >> e = A*W*testc()
    """
    if not isinstance(task,str):
        out = prmapping(testc)
        out.mapping_type = "trained"
        if task is not None:
            out = out(task)
        return out
    if (task=='init'):
        # just return the name, and hyperparameters
        return 'Test classifier', ()
    elif (task=='train'):
        # nothing to train
        return None,0
    elif (task=='eval'):
        # we are classifying new data
        truelab = x.targets
        if (len(truelab.shape)<2): # robust against ill-shaped targets
            truelab = truelab[:,numpy.newaxis]
        err = (labeld(x) != truelab)*1.
        w = x.gettargets('weights')
        if w is not None:
            err *= w
        return numpy.mean(err)
    else:
        raise ValueError("Task '%s' is *not* defined for testc."%task)


def mclassc(task=None,x=None,w=None):
    """
    Multiclass classifier from two-class classifier

         W = mclassc(A,U)

    Construct a multi-class classifier using the untrained two-class
    classifier U.

    Example:
    >> a = gendats3(100)
    >> w = mclassc(a, svc([],('p',2,10)))
    >> out = a*w
    """
    if not isinstance(task,str):
        out = prmapping(mclassc,task,x)
        return out
    if (task=='init'):
        # just return the name, and hyperparameters
        if isinstance(x,prmapping):
            name = 'Multiclass '+x.name
        else:
            name = 'Multiclass'
        return name, x
    elif (task=='train'):
        # we are going to train the mapping
        c = x.nrclasses()
        lablist = x.lablist()
        orglab = x.nlab()
        f = []
        for i in range(c):
            # relabel class i to +1, and the rest to -1:
            newlab = (orglab==i)*2 - 1
            x.targets = newlab
            u = copy.deepcopy(w)
            f.append(u.train(x))
        # store the parameters, and labels:
        return f,lablist
    elif (task=='eval'):
        # we are applying to new data
        W = w.data
        c = len(W)
        pred = numpy.zeros((x.shape[0],c))
        for i in range(c):
            out = +(W[i](x))
            # which output should we choose?
            I = numpy.where(W[i].targets==+1)
            pred[:,i:(i+1)] = out[:,I[0]]
        return pred
    else:
        raise ValueError("Task '%s' is *not* defined for mclassc."%task)

def bayesrule(task=None,x=None,w=None):
    """
    Bayes rule

           W = bayesrule(A)
           W = A*bayesrule()

    Apply Bayes rule to the output of a density estimation.

    >> a = gendatb()
    >> u = parzenm()*bayesrule()
    >> w = a*u
    >> pred = +(a*w)
    """
    if not isinstance(task,str):
        out = prmapping(bayesrule)
        out.mapping_type = "trained"
        return out
    if (task=='init'):
        # just return the name, and hyperparameters
        return 'Bayes rule', ()
    elif (task=='train'):
        # nothing to train
        return None,()
    elif (task=='eval'):
        # we are classifying new data
        # make sure that when a dataset is given as input, we also do
        # dataset out, and when a data matrix in, then also a matrix out:
        outputdataset = True
        if not isinstance(x,prdataset):
            outputdataset = False
            x = prdataset(x)
        if (len(x.prior)>0):
            dat = x.data*x.prior
        else:
            dat = x.data
        Z = numpy.sum(dat,axis=1,keepdims=True)
        out = dat/(Z+1e-10)  # Or what to do here? What is eps?
        if outputdataset:
            x = x.setdata(out)
            return x
        else:
            return out
    else:
        raise ValueError("Task '%s' is *not* defined for bayesrule."%task)

def gaussm(task=None,x=None,w=None):
    """
    Gaussian density

          W = gaussm(A,(CTYPE,REG))

    Estimate a Gaussian density on each class in dataset A. The shape of
    the covariance matrix can be specified by CTYPE:
       CTYPE='full'     full covariance matrix
       CTYPE='meancov'  averaged covariance matrix over the classes
    In order to avoid numerical instabilities in the inverse of the
    covariance matrix, regularization can be applied by adding REG to
    the diagonal of the cov.matrix.

    Example:
    >> a = gendatb()
    >> w = gaussm(a,'full',0.01))
    >> scatterd(a)
    >> plotm(w)

    """
    if not isinstance(task,str):
        out = prmapping(gaussm,task,x)
        return out
    if (task=='init'):
        # just return the name, and hyperparameters
        if x is None:
            x = ('full',[0.])
        return 'Gaussian density', x
    elif (task=='train'):
        # we are going to train the mapping
        c = x.nrclasses()
        dim = x.shape[1]
        prior = x.getprior()
        mn = numpy.zeros((c,dim))
        cv = numpy.zeros((c,dim,dim))
        icov = numpy.zeros((c,dim,dim))
        Z = numpy.zeros((c,1))
        # estimate the means and covariance matrices:
        for i in range(c):
            xi = seldat(x,i)
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
            meancov = numpy.average(cv,axis=0,weights=prior) + w[1]*numpy.eye(dim)
            meanZ = numpy.sqrt(numpy.linalg.det(meancov)*(2*numpy.pi)**dim)
            meanicov = numpy.linalg.inv(meancov)
            for i in range(c):
                icov[i,:,:] = meanicov
                Z[i] = meanZ
        else:
            raise ValueError('This cov.mat is *not* defined for gaussm.')
        # store the parameters, and labels:
        return (prior,mn,icov,Z),x.lablist()
    elif (task=='eval'):
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
                out[:,i] = numpy.sum(numpy.dot(df,W[2][i,:,:])*df,axis=1)
            else:
                out[:,i] = numpy.dot(df,W[2][i,:,:])*df
            out[:,i] = W[0][i] * numpy.exp(-out[:,i]/2)/W[3][i]
        return out
    else:
        raise ValueError("Task '%s' is *not* defined for gaussm."%task)

def ldc(task=None,x=None,w=None):
    """
    Linear discriminant classifier

          W = ldc(A,REG)

    Computation of the linear classifier between the classes of the
    dataset A by assuming normal densities with equal covariance
    matrices. The covariance matrix can be regularized by adding REG to
    the diagonal of the matrix.
    """

    if x is None:  # no regularization of the cov.matrix
        x = [0.]
    u = gaussm(task,('meancov',x))*bayesrule()
    u.name = 'LDA'
    return u

def qdc(task=None,x=None,w=None):
    """
    Quadratic discriminant classifier

          W = qdc(A,REG)

    Computation of the quadratic classifier between the classes of the
    dataset A by assuming normal densities with different covariance
    matrices per class. The covariance matrices can be regularized by
    adding REG to the diagonal of the matrices.
    """
    if x is None:
        x = [0.]
    u = gaussm(task,('full',x))*bayesrule()
    u.name = 'QDA'
    return u

def nmc(task=None,x=None,w=None):
    """
    Nearest mean classifier

          W = nmc(A)

    Computation of the nearest mean classifier between the classes in
    the dataset A.
    """
    if not isinstance(task,str):
        return prmapping(nmc,task,x)
    if (task=='init'):
        # just return the name, and hyperparameters
        return 'Nearest mean', ()
    elif (task=='train'):
        # we are going to train the mapping
        c = x.nrclasses()
        mn = numpy.zeros((c,x.shape[1]))
        v = 0.
        prior = x.getprior()
        for i in range(c):
            xi = seldat(x,i)
            mn[i,:] = numpy.mean(+xi,axis=0)
            v += prior[i]*numpy.mean(numpy.var(+xi,axis=0))
        # store the parameters, and labels:
        scale = 1./(2*v)
        return (mn,scale,prior),x.lablist()
    elif (task=='eval'):
        # we are applying to new data
        mn,scale,prior = w.data
        out = numpy.exp(-scale*sqeucldist(+x,mn))
        outp = prior*out
        return outp/numpy.sum(outp,axis=1,keepdims=True)
        #return -sqeucldist(+x,mn)
    else:
        raise ValueError("Task '%s' is *not* defined for nmc."%task)

def fisherc(task=None,x=None,w=None):
    """
    Fisher's Least Square Linear Discriminant

          W = fisherc(A)

    Finds the linear discriminant function between the classes in the 
    dataset A by minimizing the errors in the least square sense.
    """
    if not isinstance(task,str):
        out = prmapping(fisherc,task,x)
        return out
    if (task=='init'):
        # just return the name, and hyperparameters
        return 'Fisher', ()
    elif (task=='train'):
        # we are going to train the mapping
        c = x.nrclasses()
        dim = x.shape[1]
        if (c>2):
            raise ValueError('Fisher classifier is defined for two classes.')
        mn = numpy.zeros((c,dim))
        cv = numpy.zeros((dim,dim))
        v0 = 0.
        for i in range(c):
            xi = seldat(x,i)
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
    elif (task=='eval'):
        # we are applying to new data
        W = w.data
        X = +x
        out = X.dot(W[0]) - W[1] 
        if (len(out.shape)<2):  # This numpy/python stuff is pathetic
            out = out[:,numpy.newaxis]
        gr = numpy.hstack((-out,out))
        if (len(gr.shape)<2):
            gr = gr[numpy.newaxis,:]
        return gr
    else:
        raise ValueError("Task '%s' is *not* defined for fisherc."%task)

def knnm(task=None,x=None,w=None):
    """
    K-Nearest Neighbor density estimation

          W = knnm(A,K)

    A density estimator is constructed based on the k-Nearest Neighbour rule
    using the objects in A.
    Default: K=1
    """
    if not isinstance(task,str):
        return prmapping(knnm,task,x)
    if (task=='init'):
        # just return the name, and hyperparameters
        if x is None:
            x = [1]
        if (isinstance(x,list) and (len(x)==0)):
            x = [1]
        if (isinstance(x,float) or (type(x) is int)):
            x = [x]
        if (x[0]<1):
            raise ValueError('kNN: Please use a positive value for K!')
        name = '%d-NN'%x[0]
        return name, x
    elif (task=='train'):
        # we only need to store the data
        # store the parameters, and labels:
        return x,x.lablist()
    elif (task=='eval'):
        # we are applying to new data
        W = w.data
        nrcl = len(w.targets)
        k = w.hyperparam[0]
        n = x.shape[0]
        lab = W.nlab()
        out = numpy.zeros((n,nrcl))
        D = sqeucldist(+x,+W)
        I = numpy.argsort(D,axis=1)
        for i in range(n):
            thislab = lab[I[i,0:k]]
            thislab = numpy.ndarray.flatten(thislab)
            out[i,:] = numpy.bincount(thislab,minlength=nrcl)/k
        return out
    else:
        raise ValueError("Task '%s' is *not* defined for knnm."%task)

def knnc(task=None,x=None,w=None):
    """
    K-Nearest Neighbor Classifier

          W = knnc(A,K)

    Computation of the K-nearest neighbor classifier for the dataset A. 
    Default: K=1
    """
    out = knnm(task,x)*bayesrule()
    return out

def parzenm(task=None,x=None,w=None):
    """
    Parzen density estimate per class
    
          W = parzenm(A,H)

    On each of the classes in dataset A, a Parzen density is estimated,
    using a width parameter of H.
    Default H=1.
    """
    if not isinstance(task,str):
        out = prmapping(parzenm,task,x)
        return out
    if (task=='init'):
        # just return the name, and hyperparameters
        if x is None:
            x = [1.]
        if (type(x) is float) or (type(x) is int):
            x = [x]
        return 'Parzen density', x
    elif (task=='train'):
        # we only need to store the data
        # store the parameters, and labels:
        return x,x.lablist()
    elif (task=='eval'):
        # we are applying to new data
        W = w.data
        nrcl = len(w.targets)
        h = w.hyperparam[0]
        n,dim = x.shape
        Z = numpy.sqrt(2*numpy.pi)*h**dim
        out = numpy.zeros((n,nrcl))
        for i in range(nrcl):
            xi = seldat(W,i)
            D = sqeucldist(+x,+xi)
            out[:,i] = numpy.sum( numpy.exp(-D/(2*h*h)), axis=1)/Z
        return out
    else:
        raise ValueError("Task '%s' is *not* defined for parzenm."%task)

def parzenc(task=None,x=None,w=None):
    """
    Parzen Classifier

          W = parzenc(A,H)

    Computation of the Parzen classifier for the dataset A, using width
    parameter H. 
    Default: H=1.
    """
    return parzenm(task,x,w)*bayesrule()

def naivebm(task=None,x=None,w=None):
    """
    Naive Bayes density estimate per class
    
          W = naivebm(A,DENS_M)

    On each of the classes in dataset A, a Naive Bayes density is
    estimated. That means that on each of the features the density
    DENS_M is estimated independently. DENS_M should be an untrained
    density mapping.

    Default DENS_M=gaussm()
    """
    if not isinstance(task,str):
        return prmapping(naivebm,task,x)
    if (task=='init'):
        # just return the name, and hyperparameters
        if x is None:
            x = [gaussm()]
        return 'Naive Bayes density', x
    elif (task=='train'):
        # we only to estimate the densities for each feature:
        c = x.shape[1]
        f = []
        for i in range(c):
            u = copy.deepcopy(w[0])
            f.append(x[:,i:(i+1)]*u)
        # store the parameters, and labels:
        return f,x.lablist()
    elif (task=='eval'):
        # we are applying to new data
        W = w.data
        nrcl = len(w.targets)
        nrfeat = len(W)
        if not isinstance(x,prdataset):
            x = prdataset(x)
        n,dim = x.shape
        out = numpy.ones((n,nrcl))
        for i in range(nrfeat):
            out *= +(x[:,i:(i+1)]*W[i])
        return out
    else:
        raise ValueError("Task '%s' is *not* defined for naivebm."%task)

def naivebc(task=None,x=None,w=None):
    """
    Naive Bayes classifier
    
          W = naivebc(A,DENS_M)

    A Naive Bayes classifier is trained on dataset A.  That means that
    on each of the features the density DENS_M is estimated
    independently. DENS_M should be an untrained density mapping.

    Default DENS_M=gaussm()
    """
    return naivebm(task,x,w)*bayesrule()

def mog(task=None,x=None,w=None):
    """
    Mixture of Gaussians 

           W = mog(A,(K,MTYPE,REG))

    Estimate the parameters of a Mixture of Gaussians density model,
    with K Gaussian clusters. The shape of the clusters can be
    specified by MTYPE:
       MTYPE = 'full'  : full covariance matrix per cluster
       MTYPE = 'diag'  : diagonal covariance matrix per cluster
       MTYPE = 'sphr'  : single value on the diagonal of cov. matrix 
    In order to avoid numerical issues, the estimation of the covariance
    matrix can be regularized by a small value REG.

    Note: the density estimate is applied to all the data in dataset A,
    regardless to what class the objects may belong to.

    Example:
    >> a = gendatb([50,50])
    >> w = mog(a,(5,'sphr',0.0001))
    >> scatterd(a)
    >> plotm(w)
    """
    if not isinstance(task,str):
        return prmapping(mog,task,x)
    if (task=='init'):
        # just return the name, and hyperparameters
        if x is None:
            x = (3,'full',0.01)  # default: k=3, full cov., small reg.
        return 'mog', x
    elif (task=='train'):
        # we are going to train the mapping
        # some basic checking:
        n,dim = x.shape
        k = w[0]
        ctype = w[1]
        reg = w[2]
        if (k>n):
            raise ValueError('More clusters than datapoints requested.')
        # some basic inits:
        nriters = 100  #DXD
        iter = 0
        LL1 = -2e6
        LL2 = -1e6
        # initialize the priors, means, cov. matrices
        iZ = (2*numpy.pi)**(-dim/2)
        covwidth = numpy.mean(numpy.diag(numpy.cov(+x)))
        largeval = 10.
        pr = numpy.ones((k,1))/k
        I = numpy.random.permutation(range(n))
        mn = +x[I[:k],:]
        cv = numpy.zeros((dim,dim,k))
        for i in range(k):
            cv[:,:,i] = numpy.eye(dim)*covwidth*largeval
        # now run the iterations
        P = numpy.zeros((n,k))
        while (abs(LL2/LL1 - 1.)>1e-6) and (iter<nriters):
            #print("Iteration %d:"%iter)
            # compute densities
            for i in range(k):
                df = +x - mn[i,:]
                icv = numpy.linalg.inv(cv[:,:,i])
                if (dim>1):
                    P[:,i] = numpy.sum(numpy.dot(df,icv)*df,axis=1)
                else:
                    P[:,i] = numpy.dot(df,icv)*df
                P[:,i] = pr[i]*iZ* numpy.exp(-P[:,i]/2.)\
                        *numpy.sqrt(numpy.linalg.det(icv))
            # next iteration
            iter += 1
            LL2 = LL1
            LL1 = numpy.sum(numpy.log(numpy.sum(P,axis=1)))
            # compute responsibilities
            sumP = numpy.sum(P,axis=1,keepdims=True)
            sumP[sumP==0.] = 1.
            resp = P/sumP
            Nk = numpy.sum(resp,axis=0)
            # re-estimate the parameters:
            for i in range(k):
                gamma = numpy.tile(resp[:,i:(i+1)],(1,dim))
                mn[i,:] = numpy.sum(+x * gamma, axis=0,keepdims=True) / Nk[i]
                df = +x - mn[i,:]
                cv[:,:,i] = numpy.dot(df.T,df*gamma) / Nk[i] \
                        + reg*numpy.diag(numpy.ones((dim,1)))
                if (ctype=='diag'):
                    cv[:,:,i] = numpy.diag(numpy.diag(cv[:,:,i]))
                elif (ctype=='sphr'):
                    s = numpy.mean(numpy.diag(cv[:,:,i]))
                    cv[:,:,i] = s * numpy.diagflat(numpy.ones((dim,1)))
                pr[i] = Nk[i]/n
            # next!

        # precompute the inverses and normalisation constants
        Z = numpy.zeros((k,1))
        for i in range(k):
            cv[:,:,i] = numpy.linalg.inv(cv[:,:,i])
            Z[i] = iZ/numpy.linalg.det(cv[:,:,i])

        # return the parameters, and feature labels
        return (pr,mn,cv,Z), range(k)  # output p(x|k) per component
    elif (task=='eval'):
        # we are applying to new data
        W = w.data   # get the parameters out
        # W[0] is the cluster prob.
        # W[1] are the cluster means
        # W[2] are the inverse cluster covariances
        # W[3] are the normalisation constants for each cluster
        n,dim = x.shape
        k = W[1].shape[0]
        out = numpy.zeros((n,k))
        for i in range(k):
            df = +x - W[1][i,:]
            if (dim>1):
                out[:,i] = numpy.sum(numpy.dot(df,W[2][:,:,i])*df,axis=1)
            else:
                out[:,i] = numpy.dot(df,W[2][:,:,i])*df
            out[:,i] = W[0][i]*numpy.exp(-out[:,i]/2.)/W[3][i]
        return out
    else:
        raise ValueError("Task '%s' is *not* defined for mog."%task)

def mogm(task=None,x=None,w=None):
    """
    Mixture of Gaussians mapping

           W = mogm(A,(K,MTYPE,REG))

    Estimate the parameters of a Mixture of Gaussians density model per
    class in dataset A, with K Gaussian clusters. The shape of the
    clusters can be specified by MTYPE:
       MTYPE = 'full'  : full covariance matrix per cluster
       MTYPE = 'diag'  : diagonal covariance matrix per cluster
       MTYPE = 'sphr'  : single value on the diagonal of cov. matrix 
    In order to avoid numerical issues, the estimation of the covariance
    matrix can be regularized by a small value REG.

    Note: the density estimate is applied *per class* in dataset A.

    Example:
    >> a = gendatb([50,50])
    >> w = mogm(a,(3,'sphr',0.0001))
    >> scatterd(a)
    >> plotm(w)
    """
    if not isinstance(task,str):
        return prmapping(mogm,task,x)
    if (task=='init'):
        # just return the name, and hyperparameters
        if x is None:
            x = (2,'full',0.01)  # default: k=3, full cov., small reg.
        return 'mogm', x
    elif (task=='train'):
        # we are going to train the mapping
        # Train a mapping per class:
        c = x.nrclasses()
        g = []
        for i in range(c):
            xi = seldat(x,i)
            g.append(mog(xi,w))

        # return the parameters, and feature labels
        return g, x.lablist()
    elif (task=='eval'):
        # we are applying to new data
        W = w.data   # get the parameters out
        n,dim = x.shape
        if not isinstance(x,prdataset):
            x = prdataset(x)
        k = len(W)
        out = numpy.zeros((n,k))
        for i in range(k):
            out[:,i] = numpy.sum(+(x*W[i]),axis=1)
        return out
    else:
        raise ValueError("Task '%s' is *not* defined for mogm."%task)

def mogc(task=None,x=None,w=None):
    """
    Mixture of Gaussians classifier

           W = mogc(A,(K,MTYPE,REG))

    Fit a Mixture of Gaussians classifier with K clusters to dataset A.
    Basically, a Mixture is estimated per class (using mogm), and with
    Bayes rule a classifier is obtained.
    The shape of the clusters can be specified by MTYPE:
       MTYPE = 'full'  : full covariance matrix per cluster
       MTYPE = 'diag'  : diagonal covariance matrix per cluster
       MTYPE = 'sphr'  : single value on the diagonal of cov. matrix 
    In order to avoid numerical issues, the estimation of the covariance
    matrix can be regularized by a small value REG.

    Example:
    >> a = gendatb([50,50])
    >> w = mogc(a,(3,'sphr',0.0001))
    >> scatterd(a)
    >> plotc(w)
    """
    return mogm(task,x,w)*bayesrule()

def baggingc(task=None,x=None,w=None):
    "Bagging"
    if not isinstance(task,str):
        return prmapping(baggingc,task,x)
    if (task=='init'):
        # just return the name, and hyperparameters
        if x is None:
            x = (nmc,100)
        return 'Baggingc', x
    elif (task=='train'):
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
    elif (task=='eval'):
        # we are applying to new data
        X = +x
        n = X.shape[0]
        W = w.data
        T = len(W)  # nr of aggregated classifiers
        c = len(W[0].targets) # nr of classes
        out = numpy.zeros((n,c))
        J = range(n)
        for i in range(T):
            # do majority voting:
            pred = W[i](X)
            I = numpy.argmax(+pred,axis=1)
            out[J,I] += 1 
        return out
    else:
        raise ValueError("Task '%s' is *not* defined for baggingc."%task)

def stumpc(task=None,x=None,w=None):
    "Decision stump classifier"
    if not isinstance(task,str):
        out = prmapping(stumpc,task,x)
        return out
    if (task=='init'):
        # just return the name, and hyperparameters
        return 'Decision stump', ()
    elif (task=='train'):
        # we are going to train the mapping
        if (x.nrclasses() != 2):
            raise ValueError('Stumpc can only deal with 2 classes.')
        # allow weights:
        n,dim = x.shape
        w = x.gettargets('weights')
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
    elif (task=='eval'):
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
        raise ValueError("Task '%s' is *not* defined for stumpc."%task)

def adaboostc(task=None,x=None,w=None):
    "AdaBoost classifier"
    if not isinstance(task,str):
        out = prmapping(adaboostc,task,x)
        return out
    if (task=='init'):
        # just return the name, and hyperparameters
        if x is None:
            x = [100]
        return 'AdaBoost', x
    elif (task=='train'):
        # we are going to train the mapping
        # setup vars
        T = w[0]
        N = x.shape[0]
        
        y = 1 - 2*x.nlab()
        h = numpy.zeros((T,3))

        tmp = prmapping(stumpc)
        w = numpy.ones((N,1))

        alpha = numpy.zeros((T,1))
        for t in range(T):
            #print("Iteration %d in Adaboost" % t)
            x.settargets('weights',w)
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
    elif (task=='eval'):
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
        raise ValueError("Task '%s' is *not* defined for adaboostc."%task)

def svc(task=None,x=None,w=None):
    """
    Support vector classifier

           w = svc(A,(K,K_par,C))

    Train the support vector classifier on dataset A, using kernel K
    with kernel parameter K_par. The tradeoff between the margin and
    training hinge loss is defined by parameter C.

    The following kernels K are defined:
    'linear'    linear kernel (default)
    'poly'      polynomial kernel with degree K_par
    'rbf'       RBF or Gaussian kernel with width K_par

    Example:
    a = gendatb()
    w = svc(a,('rbf',4,1))
    """
    if not isinstance(task,str):
        out = prmapping(svc,task,x)
        return out
    if (task=='init'):
        # just return the name, and hyperparameters
        if x is None:
            kernel = 'rbf'
            x = 1.
            C = 1.
        else:
            kernel = x[0]
            C = x[2]
            x = x[1]
        if (kernel=='linear') or (kernel=='l'):
            clf = svm.SVC(kernel='linear',degree=x,C=C,probability=True)
        elif (kernel=='poly') or (kernel=='p'):
            clf = svm.SVC(kernel='poly',degree=x,gamma='auto',coef0=1.,C=C,probability=True)
            #clf = svm.SVC(kernel='poly',gamma=x,C=C,probability=True)
        else:
            #print("Supplied kernel is unknown, use RBF instead.")
            clf = svm.SVC(kernel='rbf',gamma=1./(x*x),C=C,probability=True)
        return 'Support vector classifier', clf
    elif (task=='train'):
        # we are going to train the mapping
        X = +x
        y = numpy.ravel(x.targets)
        clf = copy.deepcopy(w)
        clf.fit(X,y)
        return clf,x.lablist()
    elif (task=='eval'):
        # we are applying to new data
        clf = w.data
        pred = clf.decision_function(+x) 
        if (len(pred.shape)==1): # oh boy oh boy, we are in trouble
            pred = pred[:,numpy.newaxis]
            pred = numpy.hstack((-pred,pred)) # sigh
        return pred
    else:
        raise ValueError("Task '%s' is *not* defined for svc."%task)

def loglc(task=None,x=None,w=None):
    """
    Logistic classifier

           w = loglc(A,lambda)

    Train the logistic classifier on dataset A, using L2 regularisation
    with regularisation parameter lambda.

    Example:
    a = gendatb()
    w = loglc(a,(0.))
    """
    if not isinstance(task,str):
        return prmapping(loglc,task,x)
    if (task=='init'):
        # just return the name, and hyperparameters
        if x is None:
            x = [0]
        if (isinstance(x,list) and (len(x)==0)):
            x = [0]
        if (isinstance(x,float)):
            x = [x]
        if (x[0]<0):
            raise ValueError('loglc: Please use a non-negative value for LAMBDA.')
        return 'Logistic classifier', x
    elif (task=='train'):
        
        nrcl = x.nrclasses()
        if (nrcl>2):
            raise ValueError('For now, only two classes.')
        # setup variables:
        lamb = w[0]
        eta = 0.01             # learning rate
        min_rel_change = 1e-6   # minimal relative change in w
        verysmall = 1e-12       # to avoid division by zero
        N,dim = x.shape
        # extend data with constant 1 to cope with bias:
        X = numpy.concatenate((+x,numpy.ones((N,1))),axis=1)
        y = (x.targets==1)*1. # we need to define 0/+1 labels
        noty = 1-y
        w = numpy.zeros((dim+1,1))

        # perform gradient descent on loglikelihood:
        LL = 4*N*numpy.log(2) # something larger than newLL:
        newLL = 2*N*numpy.log(2) # something larger than N log(2)
        t = 0
        #print('Iteration 0: LL = %f.'%newLL)
        while ((LL-newLL)>min_rel_change*LL):
            t +=1
            fx = 1./(1.+numpy.exp(-X.dot(w)))
            # gradient:
            dLLdw = numpy.sum(numpy.repeat((fx - y),dim+1,axis=1)*X,axis=0)
            # keep track of the loglikelihood:
            LL = newLL
            newLL = -y.T.dot(numpy.log(fx + verysmall))\
                    - noty.T.dot(numpy.log(1.-fx+verysmall))
            print('Iteration %d: newLL = %f.'%(t,newLL))
            # weight update:
            w -= eta*(dLLdw[:, numpy.newaxis] + 2.*lamb*w)

        return w,x.lablist()
    elif (task=='eval'):
        # we are applying to new data
        W = w.data
        # extend data with constant 1 to cope with bias:
        X = numpy.concatenate((+x,numpy.ones((x.shape[0],1))),axis=1)
        # the sigmoid:
        out = 1./(1.+numpy.exp(-X.dot(W)))
        return numpy.concatenate((1-out,out),axis=1)
    else:
        raise ValueError("Task '%s' is *not* defined for loglc."%task)



def dectreec(task=None,x=None,w=None):
    "Decision tree classifier"
    if not isinstance(task,str):
        out = prmapping(dectreec,task,x)
        return out
    if (task=='init'):
        # just return the name, and hyperparameters
        if x is None:
            max_d = None
        else:
            max_d = x[0]
        clf = tree.DecisionTreeClassifier(max_depth=max_d)
        return 'Decision tree', clf
    elif (task=='train'):
        # we are going to train the mapping
        X = +x
        y = numpy.ravel(x.targets)
        clf = copy.deepcopy(w)
        clf.fit(X,y)
        return clf,x.lablist()
    elif (task=='eval'):
        # we are applying to new data
        clf = w.data
        pred = clf.predict_proba(+x) 
        if (len(pred.shape)==1): # oh boy oh boy, we are in trouble
            pred = pred[:,numpy.newaxis]
            pred = numpy.hstack((-pred,pred)) # sigh
        return pred
    else:
        raise ValueError("Task '%s' is *not* defined for dectreec."%task)

def lassoc(task=None,x=None,w=None):
    """
    LASSO classifier

           w = lassoc(A,alpha)

    Train the LASSO classifier on dataset A, using L1 regularisation
    with regularisation parameter alpha.

    Example:
    a = gendatb()
    w = lassoc(a,(0.))
    """
    if not isinstance(task,str):
        out = prmapping(lassoc,task,x)
        return out
    if (task=='init'):
        # just return the name, and hyperparameters
        if x is None:
            alpha = 0.
        else:
            alpha = x
        clf = linear_model.Lasso(alpha=alpha,normalize=False,tol=0.0001)
        return 'LASSO classifier', clf
    elif (task=='train'):
        # we are going to train the mapping
        X = +x
        y = numpy.ravel(x.targets)
        clf = copy.deepcopy(w)
        clf.fit(X,y)
        return clf,x.lablist()
    elif (task=='eval'):
        # we are applying to new data
        clf = w.data
        pred = clf.predict(+x) 
        if (len(pred.shape)==1): # oh boy oh boy, we are in trouble
            pred = pred[:,numpy.newaxis]
            pred = numpy.hstack((-pred,pred)) # sigh
        return pred
    else:
        raise ValueError("Task '%s' is *not* defined for lassoc."%task)

def winnowc(task=None,x=None,w=None):
    """
    Winnow classifier
         w = winnowc(A,(BETA, NRITER))
    Fit a Winnow classifier on dataset A. It is comparable to a
    perceptron classifier, except the update is multiplicative instead
    of additive. More suitable for high dimensional data where you want
    to have a sparse solution. 
    """
    if not isinstance(task,str):
        out = prmapping(winnowc,task,x)
        return out
    if (task=='init'):
        # just return the name, and hyperparameters
        if x is None:
            # learning rate, nriterations
            x = [0.01,100]
        return 'Winnow', x
    elif (task=='train'):
        # we are going to train the mapping
        # setup vars
        beta,nriter = w
        N,dim = x.shape
        # define labels:
        #y = ispositive(x.targets)
        y = (x.targets==1)
        if numpy.all(numpy.logical_not(y)):
            raise ValueError("No positive objects found.")
        y = 1 - 2.*y
        #X = numpy.concatenate((+x,numpy.ones((N,1))),axis=1)
        #W = numpy.ones((dim+1,1))
        X = numpy.concatenate((+x,-(+x),numpy.ones((N,1))),axis=1)
        W = numpy.ones((2*dim+1,1))

        for i in range(nriter):
            #print("Iteration %d in Winnow" % i)
            #randomly permute the data?
            #I = numpy.random.permutation(numpy.arange(N))
            #X = X[I,:]
            # check each object if it is correctly classified
            for j in range(N):
                if (y[j]*X[j,:].dot(W)<0.):
                    #print("error in obj %d."%j)
                    W *= numpy.exp(beta*y[j]*X[j,numpy.newaxis].T)
            #print(W)
        
        # store the parameters, and labels:
        return W,x.lablist()
    elif (task=='eval'):
        # we are applying to new data
        W = w.data
        N = x.shape[0]
        #X = numpy.concatenate((+x,numpy.ones((N,1))),axis=1)
        # of course, this is a bit inefficient, but for now it works:
        X = numpy.concatenate((+x,-(+x),numpy.ones((N,1))),axis=1)
        pred = X.dot(W)
        out = numpy.hstack((pred,-pred))
        return out
    else:
        raise ValueError("Task '%s' is *not* defined for winnowc."%task)


def pcam(task=None,x=None,w=None):
    """
    Principal Component Analysis 

           w = pcam(A,K)

    Extract the K principal components from dataset A.

    Example:
    a = gendatb()
    w = pcam(a,1)
    b = a*w
    """
    if not isinstance(task,str):
        out = prmapping(pcam,task,x)
        return out
    if (task=='init'):
        # just return the name, and hyperparameters
        return 'PCA', x
    elif (task=='train'):
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
    elif (task=='eval'):
        # we are applying to new data
        dat = +x
        return dat.dot(w.data)
    else:
        raise ValueError("Task '%s' is *not* defined for pcam."%task)

def sqeucldist(a,b):
    """
    Squared Euclidean Distances

           D = sqeucldist(A,B)

    Compute the (squared) Euclidean distances between the objects in
    dataset A and B.

    Example:
    a = gendatb()
    d = sqeucldist(a,a)
    """
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

def prcrossval(a,u,k=10,nrrep=1,testfunc=testc):
    """
    Performance estimation using crossvalidation

           E = prcrossval(A,U,K,NRREP)

    Estimate the classification error E of (untrained) mapping U on
    dataset A by using K-fold (stratified) crossvalidation. If required,
    the crossvalidation can be repeated NRREP times, to get a better
    estimate.

    Example:
    a = gendatb()
    u = nmc()
    e = prcrossval(a,u,k=10)
    """
    n = a.shape[0]
    c = a.nrclasses()
    if (nrrep==1):
        # check:
        clsz = a.classsizes()
        if (min(clsz)<k):
            raise ValueError('Some classes are too small for the number of folds.')
        # randomize the data
        I = numpy.random.permutation(range(n))
        a = a[I,:]
        # now split in folds for stratified crossval
        I = numpy.zeros((n,1))
        for i in range(c):
            J = (a.nlab()==i).nonzero()
            foldnr = numpy.mod(range(clsz[i]),k)
            I[J] = foldnr
        # go!
        e = numpy.zeros((k,1))
        for i in range(k):
            J = (I!=i).nonzero()
            xtr = a[J[0],:]
            w = xtr*u
            J = (I==i).nonzero()
            e[i] = a[J[0],:]*w*testfunc()
    else:
        e = numpy.zeros((k,nrrep))
        for i in range(nrrep):
            #print("PRcrossval: iteration %d." % i)
            e[:,i:(i+1)] = prcrossval(a,u,k,1)
    return e

def cleval(a,u,trainsize=[2,3,5,10,20,30],nrreps=3,testfunc=testc):
    """
    Learning curve

           E = cleval(A,U,TRAINSIZE,NRREPS)

    Estimate the classification error E of (untrained) mapping U on
    dataset A for varying training set sizes. Default is
    trainsize=[2,3,5,10,20,30].
    To get reliable estimates, the train-test split is repeated NRREPS=3
    times.

    Example:
    a = gendatb([100,100])
    u = nmc()
    e = cleval(a,u,nrreps=10)
    """
    nrcl = a.nrclasses()
    clsz = a.classsizes()
    if (numpy.max(trainsize)>=numpy.min(clsz)):
        raise ValueError('Not enough objects per class available.')
    N = len(trainsize)
    err = numpy.zeros((N,nrreps))
    err_app = numpy.zeros((N,nrreps))
    for f in range(nrreps):
        #print("Cleval: iteration %d." % f)
        for i in range(N):
            sz = trainsize[i]*numpy.ones((1,nrcl))
            x,z = gendat(a, sz[0],seed=f)
            w = x*u
            err[i,f] = z*w*testfunc()
            err_app[i,f] = x*w*testfunc()
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

def clevalf(a,u,trainsize=0.6,nrreps=5,testfunc=testc):
    """
    Feature curve

           E = clevalf(A,U,TRAINSIZE,NRREPS)

    Estimate the classification error E of (untrained) mapping U on
    dataset A for varying feature set sizes. The nr of features is from
    1 to the nr of features in A. For K features, the classifier is
    trained on A[:,:K]
    To get reliable estimates, the train-test split is repeated NRREPS=5
    times.

    Example:
    a = gendatb([100,100])
    u = nmc()
    e = clevalf(a,u,nrreps=10)
    """
    dim = a.shape[1]
    err = numpy.zeros((dim,nrreps))
    err_app = numpy.zeros((dim,nrreps))
    for f in range(nrreps):
        # print("Clevalf: iteration %d." % f)
        for i in range(1,dim):
            x,z = gendat(a[:,:i], trainsize,seed=f)
            w = x*u
            err[i,f] = z*w*testfunc()
            err_app[i,f] = x*w*testfunc()
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

def vandermondem(task=None,x=None,w=None):
    "Vandermonde Matrix"
    if not isinstance(task,str):
        out = prmapping(vandermondem,task,x)
        out.mapping_type = "trained"
        if isinstance(task,prdataset):
            out = out(task)
        return out
    if (task=='init'):
        # just return the name, and hyperparameters
        if x is None:
            x = 1
        return 'Vandermonde', x
    elif (task=='train'):
        # nothing to train for a fixed mapping
        return None,()
    elif (task=='eval'):
        # we are applying to new data
        n = x.shape[0]
        XX = +x
        dat = numpy.hstack((numpy.ones((n,1)),XX))
        for i in range(1,w.hyperparam):
            XX *= +x
            dat = numpy.hstack((dat,XX))
        return dat
    else:
        raise ValueError("Task '%s' is *not* defined for vandermondem."%task)

def linearr(task=None,x=None,w=None):
    """
    Linear Regression 

           w = linearr(A)
           w = linearr(A,ORDER)

    Fit an ordinary least squares regression on dataset A.
    The optional second input argument, ORDER, allows for the mapping of
    the original data X to all X^N with 0<n<ORDER

    Example:
    n = 100
    x = numpy.random.rand(n,1)
    y = 0.3*x + 0.1*numpy.random.randn(n,1)
    a = gendatr(x,y)
    w = linearr(a)
    w3 = linearr(a,3)
    scatterr(a)
    plotr(w)
    plotr(w3)
    """
    if not isinstance(task,str):
        out = prmapping(linearr,task,x)
        return out
    if (task=='init'):
        # just return the name, and hyperparameters
        if x is None:
            x = 1
        return 'Ordinary least squares', x
    elif (task=='train'):
        # we are going to train the mapping
        n,dim = x.shape
        dat = +vandermondem(x,w)
        Sinv = numpy.linalg.inv(dat.T.dot(dat))
        beta = Sinv.dot(dat.T).dot(x.targets)
        # store the parameters, and labels:
        return beta,['target']
    elif (task=='eval'):
        # we are applying to new data
        dat = +vandermondem(prdataset(x),w.hyperparam)
        return dat.dot(w.data)
    else:
        raise ValueError("Task '%s' is *not* defined for linearr."%task)

def ridger(task=None,x=None,w=None):
    """
    Ridge Regression 

           w = ridger(A,LAMB)

    Train a ridge regression on dataset A with regularisation parameter
    LAMB.

    Example:
    n = 100
    x = numpy.random.rand(n,1)
    y = 0.3*x + 0.1*numpy.random.randn(n,1)
    a = gendatr(x,y)
    w = ridger(a,(0.1))
    scatterr(a)
    plotr(w)
    """
    if not isinstance(task,str):
        out = prmapping(ridger,task,x)
        return out
    if (task=='init'):
        # just return the name, and hyperparameters
        if x is None:
            x = 0.
        return 'Ridge regression', x
    elif (task=='train'):
        # we are going to train the mapping
        n,dim = x.shape
        dat = numpy.hstack((+x,numpy.ones((n,1))))
        Sinv = numpy.linalg.inv(dat.T.dot(dat) + w*numpy.eye(dim+1))
        beta = Sinv.dot(dat.T).dot(x.targets)
        # store the parameters, and labels:
        return beta,['target']
    elif (task=='eval'):
        # we are applying to new data
        n = x.shape[0]
        dat = numpy.hstack((+x,numpy.ones((n,1))))
        return dat.dot(w.data)
    else:
        raise ValueError("Task '%s' is *not* defined for ridger."%task)

def kernelr(task=None,x=None,w=None):
    """
    Kernel Regression 

           w = kernelr(A,SIGM)

    Fit a kernel regression with width parameter SIGM to regression
    dataset A.

    Example:
    a = gendatsinc(100)
    w = kernelr(a,0.4)
    scatterr(a)
    plotr(w)
    """
    if not isinstance(task,str):
        out = prmapping(kernelr,task,x)
        return out
    if (task=='init'):
        # just return the name, and hyperparameters
        if x is None:
            x = 1.
        return 'Kernel regression', x
    elif (task=='train'):
        # we only need to store the data
        return (+x,x.targets),['target']
    elif (task=='eval'):
        # we are applying to new data
        W = w.data
        X = W[0]
        y = W[1]
        gamma = -1/(w.hyperparam*w.hyperparam)
        K = numpy.exp(gamma*sqeucldist(+x,X))
        out = K.dot(y)
        out = out/numpy.sum(K,axis=1,keepdims=True)
        return out
    else:
        raise ValueError("Task '%s' is *not* defined for kernelr."%task)

def lassor(task=None,x=None,w=None):
    """
    LASSO Regression 

           w = lassor(A,LAMB)

    Train a LASSO regression on dataset A with regularisation parameter
    LAMB.

    Example:
    n = 100
    x = numpy.random.rand(n,1)
    y = 0.3*x + 0.1*numpy.random.randn(n,1)
    a = gendatr(x,y)
    w = lassor(a,(0.1))
    scatterr(a)
    plotr(w)
    """
    if not isinstance(task,str):
        out = prmapping(lassor,task,x)
        return out
    if (task=='init'):
        # just return the name, and hyperparameters
        if x is None:
            x = 1.
        # use the sklearn implementation:
        regr = linear_model.Lasso(alpha=x)
        return 'LASSO regression', regr
    elif (task=='train'):
        X = +x
        y = x.targets
        regr = copy.deepcopy(w)
        regr.fit(X,y)
        return regr,['target']
    elif (task=='eval'):
        # we are applying to new data
        regr = w.data
        out = regr.predict(+x)
        out = out[:,numpy.newaxis]  # Pfff... Python...
        return out
    else:
        raise ValueError("Task '%s' is *not* defined for lassor."%task)

def testr(task=None,x=None,w=None):
    """
        MSE for regression

               e = testr(X)

        Compute the Mean squared error error on dataset X.

        Example:
        a = pr.gendatb([20, 20])
        e = testr(a)
    """
    if not isinstance(task,str):
        out = prmapping(testr)
        out.mapping_type = "trained"
        if task is not None:
            out = out(task)
        return out
    if (task=='init'):
        # just return the name, and hyperparameters
        return 'Test regressor', ()
    elif (task=='train'):
        # nothing to train
        return None,0
    elif (task=='eval'):
        # we are comparing the output with the targets
        err = (+x - x.targets)**2.
        w = x.gettargets('weights')
        if w is not None:
            err *= w
        return numpy.mean(err)
    else:
        raise ValueError("Task '%s' is *not* defined for testc."%task)

def hclust(D, ctype, K=None):
    """
    Hierarchical Clustering clustering

           w = hclust(D, TYPE, K)

    Train the Hierarchical clustering algorithm on distance matrix D,
    using K clusters and TYPE clustering criterion. 
    When K is defined, then the list of cluster labels is returned.
    When K is not given, the full dendrogram is returned.

    The following clustering criteria TYPE are defined:
    'single'    uses the minimum of the distances between all
                observations of the two sets (default)
    'complete'  uses the maximum distances between all observations of
                the two sets
    'average'   uses the average of the distances of each observation of
                the two sets  (DXD: Sorry, not implemented yet!)
    Example:
    a = gendat()
    D = a*proxm(a,('city'))   # use city-block distance
    dendro = hclust(D, 'single')
    plotdg(dendro)

    Or:
    a = gendat()
    D = a*proxm(a,('eucl'))
    lab = hclust(D, 'single', 3)
    scatterd(prdataset(+a,lab))
    """
    
    # some input checking:
    D = +D # no prdataset is needed
    if (D.shape[0] != D.shape[1]):
        raise ValueError('Hclust expects a square distance matrix.')
    # distances to itself do not count:
    N = D.shape[0]
    for i in range(N):
        D[i,i] = numpy.inf
    # start with each point a cluster
    I = [[i] for i in range(N)]
    # how many clusters? or the full dendrogram?
    if K is None:
        K = 0
    else:
        lab = numpy.zeros((N,1))
    dendr = numpy.ndarray((N-1,3))

    # flatten the dendrogram:
    def deepflatten(I):
        for i in I:
            if isinstance(i,list):
                yield from deepflatten(i)
            else:
                yield i

    # start clustering, and store the fusion levels:
    for k in range(N-1):

        # first, when the correct nr of clusters is found, store the cluster
        # indices:
        if (k==N-K):
            for i in range(K):
                J = list(deepflatten(I[i]))
                lab[J]=i
        # find the closest clusters
        ind = numpy.unravel_index(numpy.argmin(D, axis=None), D.shape)
        if (ctype=='single'):
            newD = numpy.min(D[ind,:],axis=0)
        elif (ctype=='complete'):
            # use 'masked_invalid' to ignore the infinities on the diag
            newD = numpy.max(numpy.ma.masked_invalid(D[ind,:]),axis=0)
        dendr[k,0] = ind[0]
        dendr[k,1] = ind[1]
        dendr[k,2] = D[ind[0],ind[1]]
        #print('Merge cluster %d and %d at level %f.'%(ind[0],ind[1],dendr[k,2]))

        # merge the clusters
        # replace the distances
        D[ind[0]:(ind[0]+1),:] = newD
        D[:,ind[0]:(ind[0]+1)] = newD[:,numpy.newaxis]
        D[ind[0],ind[0]] = numpy.inf
        D = numpy.delete(D,ind[1],0)
        D = numpy.delete(D,ind[1],1)
        # update the cluster indices
        I[ind[0]].extend(I[ind[1]])
        I.pop(ind[1])

    if (K==0):
        return dendr
    else:
        return lab

def plotdg(dendr):
    """
    Plot dendrogram

          plotdg(DENDR)

    Plot dendrogram DENDR, as it is generated from hclust.

    Example:
    a = gendat()
    D = sqeucldist(a,a)
    lab = hclust(D, 'single', 3)
    dendro = hclust(D, 'single')
    plotdg(dendro)
    """
    N = dendr.shape[0]+1
    # flatten the dendrogram:
    def deepflatten(I):
        for i in I:
            if isinstance(i,list):
                yield from deepflatten(i)
            else:
                yield i
    # find the elements at each level:
    I = [[i] for i in range(N)]
    for k in range(N-1):
        ind0 = int(dendr[k,0])
        ind1 = int(dendr[k,1])
        # update the cluster indices
        I[ind0] = [I[ind0],I[ind1]]
        I.pop(ind1)
    # find the sequence of original objects for plotting:
    flatI = list(deepflatten(I))
    x = numpy.zeros(N)
    y = numpy.zeros(N)
    for i in range(N):
        x[flatI[i]] = i

    # plot:
    for k in range(N-1):
        ind0 = int(dendr[k,0])
        ind1 = int(dendr[k,1])
        H = dendr[k,2]
        plt.plot([x[ind0],x[ind0],x[ind1],x[ind1]],\
                [y[ind0],H,H,y[ind1]],\
                'b')
        x[ind0] = (x[ind0]+x[ind1])/2
        y[ind0] = H
        x = numpy.delete(x,ind1)
        y = numpy.delete(y,ind1)
    # give the object IDs on the x-axis
    plt.xticks(range(N),flatI)
    plt.xlabel('Object nr.')
    plt.ylabel('Fusion level')


def gendats(n,dim=2,delta=2.):
    """
    Generation of a simple classification data.

        A = gendats(N,DIM,DELTA)

    Generate a two-class dataset A from two DIM-dimensional Gaussian
    distributions, containing N samples. Optionally, the mean of the
    first class can be shifted by an amount of DELTA.
    """
    prior = [0.5,0.5]
    N = genclass(n,prior)
    x0 = numpy.random.randn(N[0],dim)
    x1 = numpy.random.randn(N[1],dim)
    x1[:,0] = x1[:,0] + delta  # move data from class 1
    x = numpy.concatenate((x0,x1),axis=0)
    y = genlab(N,(-1,1))
    out = prdataset(x,y)
    out.name = 'Simple dataset'
    out.prior = prior
    return out

def gendatd(n,dim=2,delta=2.):
    """
    Generation of a difficult classification data.

        A = gendatd(N,DIM,DELTA)

    Generate a two-class dataset A from two DIM-dimensional Gaussian
    distributions, containing N samples. Optionally, the mean of the
    first class can be shifted by an amount of DELTA.
    """
    prior = [0.5,0.5]
    N = genclass(n,prior)
    x0 = numpy.random.randn(N[0],dim)
    x1 = numpy.random.randn(N[1],dim)
    x0[:,1:] *= numpy.sqrt(40)
    x1[:,1:] *= numpy.sqrt(40)
    x1[:,0] += delta  # move data from class 1
    x1[:,1] += delta  # move data from class 1
    x = numpy.concatenate((x0,x1),axis=0)
    R = numpy.array([[1.,-1.],[1.,1.]])
    x[:,0:2] = x[:,0:2].dot(R)
    y = genlab(N,(-1,1))
    out = prdataset(x,y)
    out.name = 'Difficult dataset'
    out.prior = prior
    return out

def gendatb(n=(50,50),s=1.0):
    """
    Generation of a banana shaped classes

        A = gendatb(N,S)

    Generate a two-dimensional, two-class dataset A of N objects with a
    banana shaped distribution. The data is uniformly distributed along
    the bananas and is superimposed with a normal distribution with
    standard deviation S in all directions. Class priors are P(1) = P(2)
    = 0.5.
    Defaults: N = [50,50], S = 1.
    """
    r = 5
    prior = [0.5,0.5]
    N = genclass(n,prior)
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
    out.prior = prior
    return out

def gendatc(n=(50,50),dim=2,mu=0.):
    """
    Generation of two spherical classes with different variances

        A = gendatc(N,DIM,MU)

    Generation of a DIM-dimensional 2-class dataset A of N objects.  Both
    classes are spherically Gaussian distributed.

    Class 1 has the identity matrix as covariance matrix and mean MU. If
    U is a scalar then [U,0,0,..] is used as class mean.  Class 2 has
    also the identity matrix as covariance matrix, except for a variance
    of 4 for the first two features. Its mean is 0.  Class priors are
    P(1) = P(2) = 0.5.
    """
    prior = [0.5,0.5]
    N = genclass(n,prior)

    x0 = numpy.random.randn(N[0],dim)
    x0[:,0] += mu
    x1 = numpy.random.randn(N[1],dim)
    x1[:,0] *= 3.
    if (dim>1):
        x1[:,1] *= 3.

    x = numpy.concatenate((x0,x1),axis=0)
    y = genlab(N,(-1,1))
    out = prdataset(x,y)
    out.name = 'Circular dataset'
    out.prior = prior
    return out

def gendath(n=(50,50)):
    """
    Generation of Highleyman classes

        A = gendath(N)

    Generation of a 2-dimensional 2-class dataset A of N objects
    according to Highleyman. 

    The two Highleyman classes are defined by 
    1: Gauss([1 1],[1 0; 0 0.25]).
    2: Gauss([2 0],[0.01 0; 0 4]).
    Class priors are P(1) = P(2) = 0.5 
    """
    prior = [0.5,0.5]
    N = genclass(n,prior)
    x0 = numpy.random.randn(N[0],2)
    x0[:,0] = x0[:,0] + 1.     # feature 0 from class 0
    x0[:,1] = 0.5*x0[:,1] + 1  # feature 1 from class 0
    x1 = numpy.random.randn(N[1],2)
    x1[:,0] = 0.1*x1[:,0] + 2. # feature 0 from class 1
    x1[:,1] = 2.*x1[:,1]       # feature 1 from class 1
    x = numpy.concatenate((x0,x1),axis=0)
    y = genlab(N,(-1,1))
    out = prdataset(x,y)
    out.name = 'Highleyman dataset'
    out.prior = prior
    return out

def gendats3(n,dim=2,delta=2.):
    """
    Generation of three spherical classes

        A = gendats3(N,DIM,DELTA)

    Generation of a DIM-dimensional 3-class dataset A of N objects.  All
    classes are spherically Gaussian distributed.

    Class 0 has the mean at (-DELTA, 0, 0, ..].
    Class 1 has the mean at (+DELTA, 0, 0, ..].
    Class 2 has the mean at (0, +DELTA, 0, ..].
    P(1) = P(2) = P(3) = 1/3.
    """
    N = genclass(n,[1./3,1./3,1./3])
    x0 = numpy.random.randn(N[0],dim)
    x1 = numpy.random.randn(N[1],dim)
    x2 = numpy.random.randn(N[2],dim)
    x0[:,0] -= delta
    x1[:,0] += delta
    x2[:,1] += delta
    x = numpy.concatenate((x0,x1,x2),axis=0)
    y = genlab(N,(1,2,3))
    out = prdataset(x,y)
    out.name = 'Simple dataset'
    out.prior = [1./3,1./3,1./3]
    return out

def gendatsinc(n=25,sigm=0.1):
    """
    Generation of Sinc data

        A = gendatsinc(N,SIGMA)

    Generate the standard 1D Sinc data containing N objects, with Gaussian
    noise with standard deviation SIGMA. 
    """
    x = -5. + 10.*numpy.random.rand(n,1)
    y = numpy.sin(numpy.pi*x)/(numpy.pi*x) + sigm*numpy.random.randn(n,1)
    out = prdataset(x,y)
    out.name = 'Sinc'
    return out

def boomerangs(n=100):
    p = [1./2,1./2]
    N = genclass(n, p)
    t = numpy.pi * (-0.5 + numpy.random.rand(N[0],1))

    xa = 0.5*numpy.cos(t)           + 0.025*numpy.random.randn(N[0],1);
    ya = 0.5*numpy.sin(t)           + 0.025*numpy.random.randn(N[0],1);
    za = numpy.sin(2*xa)*numpy.cos(2*ya) + 0.025*numpy.random.randn(N[0],1);

    t = numpy.pi * (0.5 + numpy.random.rand(N[1],1));

    xb = 0.25 + 0.5*numpy.cos(t)    + 0.025*numpy.random.randn(N[1],1);
    yb = 0.50 + 0.5*numpy.sin(t)    + 0.025*numpy.random.randn(N[1],1);
    zb = numpy.sin(2*xb)*numpy.cos(2*yb) + 0.025*numpy.random.randn(N[1],1);

    xa = numpy.concatenate((xa,ya,za),axis=1)
    xb = numpy.concatenate((xb,yb,zb),axis=1)
    x = numpy.concatenate((xa,xb),axis=0)
    y = genlab(N,(1,2))
    a = prdataset(x,y)
    a.name = 'Boomerangs'
    a.prior = p
    return a

def prkmeans(a,K,maxiter=100):
    """
    K-Means clustering

           w = prkmeans(A, K, MAXIT)

    Train the K-Means clustering algorithm on dataset A, using K clusters,
    with maximum number of iterations MAXIT.

    Example:
    a = gendat()
    lab = kmeans(a, 3, 100)
    """
    # initialise the stuff:
    N,dim = a.shape
    a = +a
    # DXD: we may want to improve the initialisation:
    I = numpy.random.permutation(range(N))
    means = a[I[:K],:]    # use the first K objects as starting means

    # initial label assignment:
    D = sqeucldist(a,means)
    lab = numpy.argmin(D,axis=1)
    # start the iterations:
    oldlab = copy.deepcopy(lab)
    oldlab[0] = -1 # make sure that the new labels are different:
    iter = 0
    while any(lab != oldlab) and (iter<maxiter):
        # next iteration:
        iter += 1
        oldlab = lab
        # recalculate the means:
        for k in range(K):
            I = (lab==k).nonzero()[0]  # this Python is making life a hell
            means[k,:] = numpy.mean(a[I,:],axis=0,keepdims=True)
        # recalculate the assignments:
        D = sqeucldist(a,means)
        lab = numpy.argmin(D,axis=1)
        
    return lab


def dbi(a, lab):
    """
        Davies-Bouldin Index

               e = dbi(A, Y)

        Computes the Davies-Bouldin score for features A and clustering
        labels Y.
        The outcomes differ from the sklearn implementation, because
        that one is wrong (switched the order of 'mean' and 'sqrt'). 

        Example:
        a = gendat()
        lab = prkmeans(+a, (3, 150, 'random'))
        e = dbi(+a, lab)
    """
    x = prdataset(a,lab)
    # initialise variables:
    c = x.nrclasses()
    mn = numpy.zeros((c,x.shape[1]))
    sd = numpy.zeros((c,1))
    R = numpy.zeros((c,c))

    # estimate the means and average distances to cluster centers:
    for i in range(c):
        xi = seldat(x,i)
        mi = numpy.mean(+xi,axis=0,keepdims=True)
        mn[i,:] = mi
        di = sqeucldist(+xi,mi)
        sd[i] = numpy.sqrt(numpy.mean(di))
        # This (*wrong*) version is defined in sklearn: 
        #sd[i] = numpy.mean(numpy.sqrt(di))

    # construct the R matrix:
    for i in range(c):
        R[:,i:(i+1)] = sd[i]+sd
    D =  sqeucldist(mn,mn) + 1e-15 # avoid division by 0
    R /= numpy.sqrt(D)
    numpy.fill_diagonal(R,-numpy.inf) # make sure the diagonal is ignored
    # so now you can use 'masked_invalid':
    R = numpy.mean(numpy.max(numpy.ma.masked_invalid(R),axis=1))

    return R


def icam(task=None, x=None, w=None):
    """
    Independent Component Analysis

    W = icam(A, N)

    Performs independent component analysis (ICA) on dataset A, keeping
    N (by default 1) dimensions. The implementation is the FastICA from 
    https://en.wikipedia.org/wiki/FastICA, where non-gaussian components
    are found.

    Example:
    a = read_mat("cigars")
    w = icam(a, 1)
    b = a*w
    """

    if not isinstance(task, str):
        out = prmapping(icam, task, x)
        return out
    if task == 'init':
        # just return the name, and hyperparameters
        if x is None:
            x = 1
        return 'Independent Component Analysis', x

    elif task == 'train':
        # we are going to train the mapping
        K = w
        # perform fast ica:
        N,dim = x.shape
        w = numpy.random.random_sample((K,dim))
        g = numpy.tanh
        def gprime(x):
            return 1 - numpy.tanh(x)**2

        # pre-whiten the data:
        u = pcam()*scalem()
        w_p = x*u
        a = x*w_p
        X = (+x).transpose()

        # go
        for k in range(K):
            # update component k:
            for i in range(10): # DXD: need a better convergence criterion
                wX = w[k,:].dot(X)
                w[k,:] = X.dot(g(wX).transpose())/N \
                        - gprime(wX).dot(numpy.ones((N,1))*w[k,:].transpose())/N
                if (k>0):
                    dw = w[k,:].dot(w[0,:].transpose())*w[0,:]
                    for j in range(1,k):
                        dw += w[k,:].dot(w[j,:].transpose())*w[j,:]
                    w[k,:] -= dw
                wnorm = numpy.sqrt(w[k,:].dot(w[k,:].transpose()))
                w[k,:] /= wnorm
        # return both the withening as the ICs
        return (w_p,w.transpose()), range(K)

    elif task == 'eval':
        # we are applying to new data
        w_p = w.data[0]
        w_ica = w.data[1]
        out = +(x*w_p)
        return out.dot(w_ica)
    else:
        raise ValueError("Task '%s' is *not* defined for icam."%task)


def fisherm(task=None, x=None, w=None):
    """
        Linear Discriminant Analysis

        W = fisherm(A, N)

        Performs linear discriminant analysis (LDA) on dataset A,
        keeping N < C dimensions, where C is the number of classes.

        Example:
        a = read_mat("cigars")
        w = fisherm(a, 1)
        b = a*w
        """
    if not isinstance(task, str):
        out = prmapping(fisherm, task, x)
        return out
    if task == 'init':
        # just return the name, and hyperparameters
        if x is None:
            x = 1
        return 'Fisher mapping', (x)

    elif task == 'train':
        nrd = w
        c = x.nrclasses()
        dim = x.shape[1]
        # get within and between cov.matrices:
        mn = numpy.zeros((c,dim))
        cv = numpy.zeros((dim,dim))
        for i in range(c):
            xi = seldat(x,i)
            mn[i,:] = numpy.mean(xi,axis=0)
            cv += numpy.cov(xi,rowvar=False)
        # get largest eigenvectors of S_W^-1 S_B:
        C = numpy.linalg.inv(cv)
        C = C.dot(numpy.cov(mn,rowvar=False))
        D,E = numpy.linalg.eig(C)

        # store the parameters, and labels:
        return E[:,:nrd], numpy.arange(nrd)
    elif (task=='eval'):
        # we are applying to new data
        X = +x
        return X.dot(w.data)
    else:
        raise ValueError("Task '%s' is *not* defined for fisherm."%task)


def llem(task=None, x=None, w=None):
    """
            Locally Linear Embedding

            W = llem(A, (N, K, REG))

            Performs locally linear embedding (LLE) on dataset A, keeping N dimensions, using K neighbors
            and REG being the regularization parameter.

            Example:
            a = read_mat("cigars")
            w = llem(a, (1, 3, 0.001))
            b = a*w
    """

    if not isinstance(task, str):
        out = prmapping(llem, task, x)
        return out
    if task == 'init':
        # just return the name, and hyperparameters
        if x is None:
            n = 1
            neighbors = 5
            reg = 0.01
        else:
            n = x[0]
            neighbors = x[1]
            reg = x[2]
        lle = LocallyLinearEmbedding(n_components=n, n_neighbors=neighbors, reg=reg)
        return 'Locally Linear Embedding', lle

    elif task == 'train':
        # we are going to train the mapping
        X = +x
        lle = copy.deepcopy(w)
        lle.fit(X)
        return lle, range(lle.embedding_.shape[1])

    elif task == 'eval':
        # we are applying to new data
        lle = w.data
        pred = lle.transform(+x)
        if len(pred.shape) == 1: # oh boy oh boy, we are in trouble
            pred = pred[:, numpy.newaxis]
        return pred
    else:
        raise ValueError("Task '%s' is *not* defined for lle."%task)


def isomapm(task=None, x=None, w=None):
    """
            Isometric Mapping

            W = llem(A, (N, K))

            Performs isometric mapping on dataset A, keeping N dimensions, using K neighbors.

            Example:
            a = read_mat("cigars")
            w = isomapm(a, (1, 3))
            b = a*w
    """

    if not isinstance(task, str):
        out = prmapping(isomapm, task, x)
        return out
    if task == 'init':
        # just return the name, and hyperparameters
        if x is None:
            n = 1
            neighbors = 5
        else:
            n = x[0]
            neighbors = x[1]
        isomap = Isomap(n_components=n, n_neighbors=neighbors)
        return 'Isomap', isomap

    elif task == 'train':
        # we are going to train the mapping
        X = +x
        isomap = copy.deepcopy(w)
        isomap.fit(X)
        return isomap, range(isomap.embedding_.shape[1])

    elif task == 'eval':
        # we are applying to new data
        isomap = w.data
        pred = isomap.transform(+x)
        if len(pred.shape) == 1: # oh boy oh boy, we are in trouble
            pred = pred[:, numpy.newaxis]
        return pred
    else:
        raise ValueError("Task '%s' is *not* defined for isomap."%task)

def feateval(task=None,x=None,w=None):
    """
    Evaluation of a feature set

          J = feateval(A,crit)

    Evaluation of features by the criterion CRIT, using objects in the
    dataset A.  The larger J, the better. Resulting J-values are
    incomparable over the various methods.
       crit='1NN'     Leave-one-out 1-nearest neighbor performance
       crit='in-in'   inter-intra distances
       crit='eucl-s'  sum of (squared) euclidean distances between class
                      means
       crit='eucl-m'  minimum of (squared) euclidean distances between
                      class means

    When the criterion is an untrained classifier, the apparent error is
    used. One can also request the K-fold cross-validation error using:
    crit = ['crossval', ldc(), K]

    Example:
    >> A = gendatb()
    >> e = feateval(A,'in-in')
    >> e = feateval(A,ldc())
    >> e = feateval(A,['crossval',ldc(),10])
    """
    if not isinstance(task,str):
        out = prmapping(feateval,x)
        out.mapping_type = "trained"
        if task is not None:
            out = out(task)
        return out
    if (task=='init'):
        # find out if we use a prespecified criterion, on a prmapping
        if x is None:
            x = ldc()
        if isinstance(x,prmapping):
            x = ['app.err.', x]
        if isinstance(x,tuple):
            x = list(x)
        # just return the name, and hyperparameters
        return 'Feature evaluation', x
    elif (task=='train'):
        # nothing to train
        return None,0
    elif (task=='eval'):
        # we are evaluating the features:
        J = 0
        if isinstance(w.hyperparam,list):
            if (w.hyperparam[0]=='app.err.'):
                # use the apparent error of the given classifier
                classif = x*w.hyperparam[1]
                J = numpy.mean(labeld(x*classif) != x.targets)*1.
            elif (w.hyperparam[0]=='crossval'):
                J = prcrossval(x,w.hyperparam[1],w.hyperparam[2])
                J = numpy.mean(J)
        elif (w.hyperparam=='1NN'):
            # Leave-one-out 1-nearest neighbor
            lab = a.nlab()
            D = sqeucldist(+a,+a)
            for i in range(a.shape[0]):
                D[i,i] = numpy.inf
            I = numpy.argmin(D,axis=0)
            J = numpy.mean((lab == lab[I])*1.)
        elif (w.hyperparam=='eucl-m'):
            # minimum of (squared) euclidean distances between class means
            c = x.nrclasses()
            dim = x.shape[1]
            mn = numpy.zeros((c,dim))
            for i in range(c):
                xi = seldat(x,i)
                mn[i,:] = numpy.mean(+xi,axis=0)
            D = sqeucldist(mn,mn)
            for i in range(c):
                D[i,i] = numpy.inf
            J = numpy.min(D)
        elif (w.hyperparam=='eucl-s'):
            # sum of (squared) euclidean distances between class means
            c = x.nrclasses()
            dim = x.shape[1]
            mn = numpy.zeros((c,dim))
            for i in range(c):
                xi = seldat(x,i)
                mn[i,:] = numpy.mean(+xi,axis=0)
            D = sqeucldist(mn,mn)
            J = numpy.sum(D)/2.
        elif (w.hyperparam=='in-in'):
            # inter-intra distances
            c = x.nrclasses()
            dim = x.shape[1]
            mn = numpy.zeros((c,dim))
            cv = numpy.zeros((c,dim,dim))
            for i in range(c):
                xi = seldat(x,i)
                mn[i,:] = numpy.mean(+xi,axis=0)
                cv[i,:,:] = numpy.cov(+xi,rowvar=False)
            S_b = numpy.cov(mn,rowvar=False)  # between scatter
            S_w = numpy.mean(cv,axis=0) # within scatter
            J = numpy.trace(numpy.linalg.pinv(S_w).dot(S_b))
        return J
    else:
        raise ValueError("Task '%s' is *not* defined for feateval."%task)

def featseli(task=None, x=None, w=None):
    """
    Individual Feature Selector

           w = featseli(A, (K,CRIT))

    Individual feature selection of K features using the dataset A. The
    criterion is defined by CRIT; for more information for possible
    criteria, see FEATEVAL.

    Example:
    a = gendat()
    w = featseli(a, (4,ldc()))
    """
    if not isinstance(task,str):
        out = prmapping(featseli, task, x)
        return out
    if (task=='init'):
        # just return the name, and hyperparameters
        if isinstance(x,int):
            x = [x, ldc()]
        if x is None:
            x = [numpy.inf,ldc()]
        return 'Individual Feature Selector', x
    elif (task=='train'):
        # we are going to train the mapping
        K = w[0]
        if (K>x.shape[1]):
            K = x.shape[1] # we want all the features, ordered 
        # the list of features to choose from:
        I = list(range(x.shape[1]))
        perf = numpy.zeros((x.shape[1]))
        for i in I:
            perf[i] = feateval(x[:,i:(i+1)] ,w[1])
        # sort them:
        J = numpy.argsort(-perf)
        J = J[:K]
        # done, we found our K features
        featlab = [x.featlab[j] for j in J]
        return J,featlab
    elif (task=='eval'):
        # we are applying to new data
        return x[:,w.data]
    else:
        raise ValueError("Task '%s' is *not* defined for feature selection."%task)

def featself(task=None, x=None, w=None):
    """
    Sequential Forward Feature Selector

           w = featself(A, (K,CRIT))

    Forward feature selection of K features using the dataset A. The
    criterion is defined by CRIT; for more information for possible
    criteria, see FEATEVAL.

    Example:
    a = gendat()
    w = featself(a, (4,ldc()))
    """
    if not isinstance(task,str):
        out = prmapping(featself, task, x)
        return out
    if (task=='init'):
        # just return the name, and hyperparameters
        if isinstance(x,int):
            x = [x, ldc()]
        if x is None:
            x = [numpy.inf,ldc()]
        return 'Sequential Forward Feature Selector', x
    elif (task=='train'):
        # we are going to train the mapping
        K = w[0]
        if (numpy.isinf(K)):
            K = x.shape[1] # we want all the features, ordered 
        # the list of features to choose from:
        I = list(range(x.shape[1]))
        x_prev = x[:,:0]
        J = []    # the choosen features
        for k in range(K):
            # choose the k'th feature:
            # select from the candidates:
            perf = -numpy.ones((x.shape[1]))
            for i in I:
                x_new = x_prev.concatenate(x[:,i:(i+1)],axis=1)
                perf[i] = feateval(x_new,w[1])
            # select the best one:
            Jadd = numpy.argmax(perf)
            #print('The %d-th feature is feature %d.'%(k,Jadd)) 
            # update our variables
            J.append(Jadd)
            I.remove(Jadd)
            x_prev = x_prev.concatenate(x[:,Jadd:(Jadd+1)],axis=1)
        # done, we found our K features
        featlab = [x.featlab[j] for j in J]
        return J,featlab
    elif (task=='eval'):
        # we are applying to new data
        return x[:,w.data]
    else:
        raise ValueError("Task '%s' is *not* defined for feature selection."%task)

def featselb(task=None, x=None, w=None):
    """
    Sequential Backward Feature Selector

           w = featselb(A, (K,CRIT))

    Backward feature selection of K features using the dataset A. The
    criterion is defined by CRIT; for more information for possible
    criteria, see FEATEVAL.

    Example:
    a = gendat()
    w = featselb(a, (4,ldc()))
    """
    if not isinstance(task,str):
        out = prmapping(featselb, task, x)
        return out
    if (task=='init'):
        # just return the name, and hyperparameters
        if isinstance(x,int):
            x = [x, ldc()]
        if x is None:
            x = [numpy.inf,ldc()]
        return 'Sequential Backward Feature Selector', x
    elif (task=='train'):
        # we are going to train the mapping
        K = w[0] # number of features to retain
        # the choosen features:
        J = list(range(x.shape[1]))
        n = 0
        while (len(J)>K):
            # choose the k'th feature:
            # select from the candidates:
            perf = -numpy.ones((x.shape[1]))
            for j in J:
                newJ = copy.deepcopy(J) #AIAI Caramba!
                newJ.remove(j)
                perf[j] = feateval(x[:,newJ],w[1])
            # select the best one:
            Jrem = numpy.argmax(perf)
            n += 1
            #print('The %d-th feature to remove is feature %d.'%(n,Jrem)) 
            # update our variables
            J.remove(Jrem)

        # done, we found our K features
        featlab = [x.featlab[j] for j in J]
        return J,featlab
    elif (task=='eval'):
        # we are applying to new data
        return x[:,w.data]
    else:
        raise ValueError("Task '%s' is *not* defined for feature selection."%task)



