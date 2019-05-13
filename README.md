# prtools
Prtools for Python
==================

This is a bare-bones implementation of Prtools for Python. It includes the 
prdataset and prmapping objects. The main advantages are:
1. The data and the labels are stored in one prdataset. That means that when
a subset of a dataset is selected, only one operation is needed:
```
> b = a[:6,:]
```
and the corresponding labels are returned in b as well.

2. Operations on datasets can be concatenated in a sequential mapping:
```
> u = scalem()*pcam()*ldc()
```
When this mapping is trained on dataset a, like:
```
> w = a*u
```
or
```
> w = u.train(a)
```
all steps in the sequence are trained. And in the evaluation on new data, like:
```
> pred = w(b)
```
or
```
> pred = b*w
```
the full sequence is applied again. This avoids the error-prone mix of training
the first mapping on the training set, applying it to the train and test set,
then train the second mapping on the training set etc.

3. For two-dimensional data it is very easy to visualise classifiers. This can
be done by the commands:
```
> scatterd(a)
> plotc(w)
```

Mapping
=======

To define a mapping, you need to define three tasks:
1. 'untrained' mapping: here you define the name, and possibly the
   hyperparameters of the mapping,
2. 'train' the mapping: here you train the mapping parameters, given a
   dataset and possibly its hyperparameters. The output should be the
   trained parameters, and the names of the output features,
3. 'eval' the mapping: given a dataset, and the trained mapping, the
   output for the given dataset should be computed.

Next to these three operations, the definition of a mapping also
requires a call to the function `prmapping` when no task is defined. 
A bare-bones mapping is, for instance, the scale mapping:

```
def scalem(task=None,x=None,w=None):
    "Scale mapping"
    if not isinstance(task,str):
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
```

In this mapping each feature is normalised to get zero mean, and unit
variance.  As can be seen, the mean and variance are estimated from a
training set. In the 'eval' section the features are rescaled.




