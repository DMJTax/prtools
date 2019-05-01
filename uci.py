def iris(getOnline=False):
    if getOnline:
        link = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
        f = requests.get(link)
        txt = f.text.splitlines()
    else:
        text_file = open("iris.data", "r")
        txt = text_file.readlines()
        text_file.close()

    N,dim = 150,4
    x = numpy.zeros((N,dim))
    labx = numpy.empty(N,dtype=object)
    i = 0
    for line in txt:
        nr = line.split(',')
        for j in range(dim):
            x[i,j] = float(nr[j])
        labx[i] = nr[dim]
        i += 1
        if (i>=N):
            break
    a = prdataset(x,labx)
    a.featlab = ['sepal length','sepal width','petal length','petal width']
    a.name = 'Iris'
    return a

