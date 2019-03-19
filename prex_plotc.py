from prtools import *
import matplotlib.pyplot as plt

# standard prtools: make data and define classifier:
#a = gendatb((30,30))
#a = gendatd((30,30))
a = gendath((30,30))
#a = gendats3((100))

#u = nmc()
#u = scalem()*proxm(('rbf',0.5))*nmc()
u = gaussm(('full',0.1))*bayesrule()
#u = qdc()
#u = knnc([1])
#u = scalem()*parzenc()

# train classifier, plot:
w = a*u
pred = a*w
print w
err = pred*testc()
print('Classification error: %3.2f' % err)
plt.figure(1)
scatterd(a)
plotc(w)
#plotm(w,gridsize=100,colors='gray')

plt.show()
