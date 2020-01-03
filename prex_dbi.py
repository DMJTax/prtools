import numpy
import prtools as pr
import matplotlib.pyplot as plt

a = pr.read_mat("triclust")

lab = pr.kmeans(a,(3,150,'random'))

e = pr.dbi(a, lab)

