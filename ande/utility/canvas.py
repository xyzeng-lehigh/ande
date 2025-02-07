import cmath
import numpy as np
import matplotlib.pyplot as plt

# Given the coefficient a_k, k >= -offset compute and plot the order star of
#   sigma(z) = \sum_k coef[k+offset] exp(kz) - z
def plot_os_exp_z(offset,coef,range_x,range_y,nsample):
    if nsample <= 0:
        nsample = 2001
    x = np.linspace(range_x[0],range_x[1],nsample)
    y = np.linspace(range_y[0],range_y[1],nsample)
    [X,Y] = np.meshgrid(x,y)
    # Start with W = -Z = -X-iY
    W = -X
    l = offset
    r = len(coef)-1-l
    for k in range(-l,r+1):
        if k == 0:
            W = W + float(coef[offset])*np.ones((nsample,nsample))
        else:
            W = W + float(coef[offset+k])*np.exp(k*X)*np.cos(k*Y)
    # This will thicken the countour lines
    #plt.contour(X,Y,W,levels=[0],colors=['k'],linewidths=2)
    plt.contourf(X,Y,W,levels=[np.min(np.min(W))-1,0],cmap=plt.cm.bone)
    plt.plot([0,0],[range_y[0],range_y[1]],'k-',linewidth=2)
    plt.show()
