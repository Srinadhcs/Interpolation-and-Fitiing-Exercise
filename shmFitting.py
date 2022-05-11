# Required Imports
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline


# ### Q1  
# #### Defining Function to model SHM

def shm(t, A, omega, phi, d):
    return A*np.cos(omega*t + phi) + d


# ### Q2
# #### Read the whole text file and store in appropriate arrays


t, y = np.loadtxt(os.path.join(os.getcwd(),"SHM-200g.txt"), unpack=True)


# ### Q3 
# #### Setting up array with same number of elements of data points and filling it with 1 mm = 0.001 m.

sig = np.zeros(len(y))
sig.fill(0.001)


# ### Q4
# #### Estimating A

A = (y.max() - y.min())/2.0
print(f"A: {A}")


# ### Q5
# #### Estimating W from T

cs = CubicSpline(t, y)
roots = cs.roots()

adj_diff = np.zeros(len(roots) - 1) 

for i in range(len(roots)-1):
    adj_diff[i] = roots[i+1] - roots[i]

T = np.mean(adj_diff) * 2

w = 2.0*np.pi/T

print(f"T: {T} \nw: {w}")


# ### Q6
# #### Estimating phi

phi = np.arccos(y[0]/A)

print(f"phi: {phi}")


# ### Q7
# #### Fitting Curve


theta_0 = [A, w, phi, 0.0]
theta, cov_mat = curve_fit(shm, t, y, p0=theta_0, sigma=sig)
print("Theta\n",theta)
print("\nCovariance Matrix\n",cov_mat)


# ### Q8
# #### Plotting the data points


t_m = np.linspace(t.min(), t.max(), 1000)
y_m = shm(t_m, theta[0], theta[1], theta[2], theta[3])

plt.errorbar(t, y, sig, fmt=".")
plt.plot(t_m, y_m)
plt.xlabel("t")
plt.ylabel("y")

plt.show()


# ### Q9
# #### Generating Correlation

cor_mat = np.zeros_like(cov_mat)
for i in range(0,len(theta)):
    for j in range(0,len(theta)):
        cor_mat[i][j] = cov_mat[i][j]/np.sqrt(cov_mat[i][i]*cov_mat[j][j])

print(cor_mat)


# ### Q10
# #### Comparing standard deviations


std_dev = np.sqrt(np.diag(cov_mat))

std_A, std_w, std_phi, std_d = std_dev

print(f"Std Deviation of A\t: {std_A}")
print(f"Std Deviation of w\t: {std_w}")
print(f"Std Deviation of phi\t: {std_phi}")
print(f"Std Deviation of d\t: {std_d}")


# - Parameter phi has the largest Standard Deviation
