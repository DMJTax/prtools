import numpy
from prtools import *
import matplotlib.pyplot as plt
import uci

a = uci.breast(True)
a = uci.missingvalues(a,'remove')

for lamb in (0.1, 1.0, 10.0):
    w = lassor(a,lamb)
    print("Setting lambda to %f gives the weights:"%lamb)
    print(w.data.coef_)
    err = a*w*testr()
    print("MSE = %f" % err)

