import prtools as pr
import matplotlib.pyplot as plt

# make data and define classifier:
a = pr.gendatb((5,5))
print(a)

# Three ways to train:
# (0) boring prtools way
w = pr.nmc(a)
# (1) advanced prtools way
u = pr.nmc(a)
w = a*u
# (2) pythonesc:
w = pr.nmc()
w.train(a)

print(w)

# ways to evaluate:
# (0) standard prtools way
pred = a*w
# (1) or
pred = w(a)
# (2) pythonesc:
pred = w.eval(a)

# get the labels
lab = a*pr.labeld()
# or
lab = pr.labeld(pred)
print(lab)
# get the error
err = pred*pr.testc()
print('Classification error is: ',err)
