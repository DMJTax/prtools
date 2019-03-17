from prtools import *
import matplotlib.pyplot as plt

# standard prtools: make data, train classifier, plot:
a = gendatb((40,40))
u = scalem()*proxm(('rbf',0.5))*nmc()
w = a*u
out = a*w
err = out*testc()
print('Classification error: %3.2f' % err)
plt.figure(1)
scatterd(a)
plotc(w)

plt.show()
