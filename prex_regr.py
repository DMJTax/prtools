from prtools import lassor, testr
from prtools.uci import breast, missingvalues

# Read in one of the UCI datasets (and use the class labels as a
# regression target):
a = breast(True)
a = missingvalues(a,'remove')

# Use a few regularisation parameters and see how sparse the solutions
# become:
for alpha in (0.1, 1.0, 10.0):
    w = lassor(a,alpha)
    print("Setting alpha=%f gives the weights:"%alpha)
    print(w.data.coef_)
    err = a*w*testr()
    print("MSE = %f" % err)

