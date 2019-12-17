import numpy
import prtools as pr
import matplotlib.pyplot as plt

# Generate 2D input data
n = 100
x = numpy.random.randn(n,2)
y = numpy.sin(x[:,0])*numpy.sin(x[:,1])
a = pr.gendatr(x,y)   # store it in a regression dataset

# Fit a model:
#w = pr.ridger(a,0.5)
w = pr.kernelr(a,0.5)
print(w)

# Show:
pr.scatterr(a)
pr.plotr(w)

plt.show()
