import prtools as pr
import matplotlib.pyplot as plt

# make data and concatenate:
print("Combining datasets:")
a = pr.gendatb((5,5))
b = pr.gendats((5,5))
c = pr.gendatd((5,5))
print(a)

out = pr.concatenate([a,b,c],axis=0)
print(out)
out = pr.concatenate([a,b,c],axis=1)
print(out)
out = pr.concatenate([a,b,c])
print(out)

# make mappings and concatenate:
print("Combining mappings:")
w1 = pr.ldc(a)
w2 = pr.parzenc(a)
w3 = pr.qdc(a)

out = pr.concatenate([w1,w2,w3],axis=0)
print(out)
out = pr.concatenate([w1,w2,w3],axis=1)
print(out)
out = pr.concatenate([w1,w2,w3])
print(out)
