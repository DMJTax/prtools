import numpy
import requests
import dataset

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
        labx[i] = nr[dim]
        i += 1
        if (i>=N):
            break
    a = dataset.prdataset(x,labx)
    return a

def arrythmia(getOnline=False):
    a = getUCIdata("arrhythmia",452,279,True)
    a.name = 'Arrythmia'
    return a

def iris(getOnline=False):
    a = getUCIdata("iris",150,4)
    a.featlab = ['sepal length','sepal width','petal length','petal width']
    a.name = 'Iris'
    return a

