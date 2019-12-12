import numpy
import requests
import os
from prtools import dataset
from scipy.io import loadmat

def getUCIdata(name,N,dim,getOnline=False):
    if getOnline:
        link = "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        f = requests.get(link+name+"/"+name+".data")
        txt = f.text.splitlines()
    else:
        text_file = open(name+".data", "r")
        txt = text_file.readlines()
        text_file.close()
    x = numpy.zeros((N,dim))
    labx = numpy.empty(N,dtype=object)
    i = 0
    for line in txt:
        nr = line.split(',')
        for j in range(dim):
            try:
                x[i,j] = float(nr[j])
            except:
                x[i,j] = numpy.nan
        # finally get the label:
        thislab = nr[dim].rstrip()
        try:
            labx[i] = float(thislab)
        except:
            labx[i] = thislab
        i += 1
        if (i>=N):
            break
    a = dataset.prdataset(x,labx)
    return a

def missingvalues(a,val):
    if isinstance(val,str):
        if (val=='remove'):
            suma = numpy.sum(+a,axis=1)
            I = numpy.nonzero(~numpy.isnan(suma))
            a = a[I[0],:]
        elif (val=='mean'):
            dat = +a
            suma = numpy.sum(dat,axis=0)
            I = numpy.nonzero(numpy.isnan(suma))[0]
            for i in range(len(I)):
                feata = +a[:,I[i]]
                J = numpy.nonzero(numpy.isnan(feata))
                dat[J,I[i]] = numpy.nanmean(feata)
            a.data = dat
    return a

def ionosphere(getOnline=False):
    a = getUCIdata("ionosphere",351,34,getOnline)
    a.name = 'Ionosphere'
    return a

def arrythmia(getOnline=False):
    a = getUCIdata("arrhythmia",452,279,getOnline)
    a.name = 'Arrythmia'
    return a

def iris(getOnline=False):
    a = getUCIdata("iris",150,4,getOnline)
    a.featlab = ['sepal length','sepal width','petal length','petal width']
    a.name = 'Iris'
    return a

def breast(getOnline=False):
    a = getUCIdata("breast-cancer-wisconsin",699,10,getOnline)
    a.featlab = [ "Sample code number", "Clump Thickness",
            "Uniformity of Cell Size", "Uniformity of Cell Shape",
            "Marginal Adhesion", "Single Epithelial Cell Size",
            "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses"]
    a.name = 'Breast'
    a = a[:,1:]
    return a

def read_mat(file):
    """
    Reads a dataset from a .mat file and converts it into a prdataset

    :param file: name of .mat file to be read from the /data folder
    :return: a prdataset containing the features/labels read from the file
    """
    import prtools # import prtools to get its installation path
    data = loadmat(os.path.dirname(prtools.__file__) + '/data/' + file + '.mat')
    if file == 'diabetes' or file == 'mfeat_zer' or file == 'mfeat_pix':
        features = data['a'][0][0][0]
        labels = data['a'][0][0][1]
    else:
        features = data['a']['data'][0][0]
        labels = data['a']['nlab'][0][0]
    a = dataset.prdataset(features, labels)
    return a
