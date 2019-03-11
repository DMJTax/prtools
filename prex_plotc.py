from prtools import *
import matplotlib.pyplot as plt

# standard prtools: make data, train classifier, plot:
a = gendatb((40,40))
u = prmapping(proxm,'rbf',1.5)*prmapping(nmc)
w = a*u
plt.figure(1)
scatterd(a)
plotc(w)

plt.show()
