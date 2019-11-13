import prtools as pr
import matplotlib.pyplot as plt

# make data and define classifier:
a = pr.gendatb((500,500))
u = pr.nmc()

# compute and show the learning curve, containing
# the true error and apparent error:
e = pr.cleval(a,u)
# Let's add another classifier:
e2 = pr.cleval(a,pr.ldc())

plt.legend()
plt.show()

