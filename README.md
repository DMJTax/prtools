# prtools
Prtools for Python

This is a bare-bones implementation of Prtools for Python. It includes the 
prdataset and prmapping objects. The main advantages are:
1. The data and the labels are stored in one prdataset. That means that when
a subset of a dataset is selected, only one operation is needed:

> b = a[:6,:]

and the corresponding labels are returned in b as well.

2. Operations on datasets can be concatenated in a sequential mapping:

> u = scalem()*pcam()*ldc()

When this mapping is trained on dataset a, like:

> w = a*u

or

> w = u.train(a)

all steps in the sequence is trained. And in the evaluation on new data, like:

> pred = w(b)

or

> pred = b*w

the full sequence is applied again. This avoids the error-prone mix of training
the first mapping on the training set, applying it to the train and test set,
then train the second mapping on the training set etc.

3. For two-dimensional data it is very easy to visualise classifiers. This can
be done by the command:

> plotc(w)




