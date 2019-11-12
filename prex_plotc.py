import prtools as pr
import matplotlib.pyplot as plt

# standard prtools: make data and define classifier:
a = pr.gendatb((30,30))
#a = pr.gendatd((30,30))
#a = pr.gendath((30,30))
#a = pr.gendats3((100))
print(a)

#u = pr.nmc()
#u = pr.scalem()*pr.proxm(('rbf',0.5))*pr.nmc()
#u = pr.gaussm(('full',0.1))*pr.bayesrule()
#u = pr.ldc()
#u = pr.knnc([1])
#u = pr.scalem()*pr.parzenc()
u = pr.scalem()*pr.parzenc([0.6])

# train classifier, get labels, error and plot:
w = a*u
print(w)
pred = a*w
print(pred)
lab = pred*pr.labeld()
print(+lab)
err = pr.testc(pred)
print('Classification error: %3.2f' % err)
plt.figure(1)
pr.scatterd(a)
pr.plotc(w)

plt.show()
