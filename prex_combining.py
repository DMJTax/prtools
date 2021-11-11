import prtools as pr
import matplotlib.pyplot as plt

# make train and test data:
x = pr.gendatb((10,10))
z = pr.gendatb((1000,1000))

# fit base classifiers:
w1 = x*pr.parzenc()
w2 = x*pr.fisherc()
w3 = x*pr.qdc()

# Define the combined classifiers:
w = pr.parallelm([w1,w2,w3])
wmean = w*pr.meanc()
wmin = w*pr.minc()
wmax = w*pr.maxc()
wmedian = w*pr.medianc()
wprod = w*pr.prodc()

# How do the different combiners perform?
print("Mean combiner: ", z*wmean*pr.testc())
print("Min combiner: ", z*wmin*pr.testc())
print("Max combiner: ", z*wmax*pr.testc())
print("Median combiner: ", z*wmedian*pr.testc())
print("Product combiner: ", z*wprod*pr.testc())
