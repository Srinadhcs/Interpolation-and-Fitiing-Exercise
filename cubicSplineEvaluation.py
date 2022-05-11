# Required Imports
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline, lagrange
from scipy.integrate import quadrature
from scipy.misc import derivative
from scipy import interpolate


# ### Q1

def langrange_interpolation(val, x, fx):
    N = len(x)
    result = 0
    for i in range(0,N):
        prod = 1 
        for j in range(0,N):
            if (j!=i):
                prod *= (val - x[j])/(x[i] - x[j])
        result += prod*fx[i]
    return result

x = np.array([0,3,5,7,13])
fx = np.array([-7,6,-1,11,18])

cs = CubicSpline(x, fx)

print("Results Lagrange:")
print("y(2)= ",langrange_interpolation(2, x, fx) )
print("y(9)= ",langrange_interpolation(9, x, fx) )
print("\nResults CubicSpline:")
print("y(2)= ",cs(2.0))
print("y(2)= ",cs(9.0))


# ### Q2

def gaussian(x,mu,sigma):
    return np.exp((-(x-mu)**2)/(2*sigma**2))*math.sqrt(1/(2*math.pi*sigma))

x = np.linspace(-4,4,10)
y = gaussian(x, 0.0, 1.0)

cs = CubicSpline(x,y)
x_i = np.linspace(-4,4,1000)

y_g = gaussian(x_i, 0.0, 1.0)
y_cs = cs(x_i)
y_l = langrange_interpolation(x_i, x, y)


plt.plot(x_i,y_g, color='b')
plt.plot(x_i,y_l, color='g')
plt.plot(x_i,y_cs, color='r')
plt.plot(x,y,",")
plt.show()


# #### Inferences
# 
# - The lagrange interpolation got distorted at the edges as bisection takes points on either side of the value and hence there are no values to fit on edges.
# - As we can see the CubicSpline gave the best fit as compared to all other methods.
# - Therefore, I would like to choose the CubicSpline interpolation over Guassian and Lagrange interpolations.

# ### Q3 

def f(x):
    return (x-2)**3 - (3.5*x) + 8

def df(x):
    return 3*(x-2)**2 - 3.5

def F(a, b):
    def integral(x):
        return (x-2)**4 + (8*x) - (7/4)*(x**2)

    return integral(b) - integral(a)

def central_diff(x):
    def f(x):
        return (x-2)**3 - (3.5*x) + 8
    h = 0.001
    return (f(x+h) - f(x))/h 


# ### Q4

x = [0, 1, 2, 3, 4]
fx = list(map(f, x))

cs = CubicSpline(x, fx)


# ### Q5


exactF = F(0, 4)
guasF, gausErr = quadrature(cs, 0, 4)

print(f"Integral using our function: {exactF}")
print(f"Integral using Guassian Quadrature function: {guasF}")


# ### Q6


exact_df = list(map(df, x))
central_df = [derivative(cs, val, dx = 0.001) for val in x] 

print(f"Exact Derivative:\n{exact_df}")
print(f"\nCentral Difference Derivative:\n{central_df}")      


# ### Q7

points = np.linspace(0, 4, 50)

exact_f = np.array(list(map(f, points)))
cs_f = np.array([i for i in list(map(cs, points))])

lg = lagrange(x, fx)
l_f = np.array(list(map(lg, points)))


print(cs_f)


print(exact_f)


print(l_f)


# - As we can see we are getting the same values for the function using Lagrange Interpolation and CubicSpline Interpolation which fully resembles with expected output of the actual function.