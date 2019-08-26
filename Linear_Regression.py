#%%
import numpy as np

x=[2,4,6,8]
y=[81,93,91,97]

mx = np.mean(x)
my = np.mean(y)
print('mean of x:',mx)
print('mean of y:',my)

divisor = sum([(mx-i)**2 for i in x])
print(divisor)

def top(x, mx, y ,my):
    d = 0
    for i in range(len(x)):
        d += (x[i]-mx)*(y[i]-my)
    return d

dividend = top(x, mx, y, my)

print("Denominator is:",divisor)
print("Numerator is",dividend)

#%%
a = dividend / divisor
b = my - (mx*a)
print("Slope a:",a)
print("Intercept of Y as B:",b)
