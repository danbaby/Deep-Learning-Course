import numpy as np
import math
x=np.array([64.3,99.6,145.45,63.75,135.46,92.85,86.97,144.76,59.3,116.03])
y=np.array([62.55,82.42,132.62,73.31,131.05,86.57,85.49,127.44,55.25,104.84])
numx=0
numy=0
for i in range(10):
    numx+=x[i]
    numy+=y[i]
numx_=numx/10.00
numy_=numy/10.00
suma=0
sumx=0
for i in range(10):
    suma+=(x[i]-numx)*(y[i]-numy)
    sumx+=math.pow(x[i]-numx,2)
w=float(suma/sumx)
b=numy-w*numx
print("w的值：%.2f\nb的值：%.2f"%(w,b))