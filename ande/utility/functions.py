import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Given the coefficients a_k, k >= -offset
# The next two functions compute:
#   sum_{k>=-offset}a_k cos(k theta)
# and
#   sum_{k>=-offset}a_k sin(k theta)
# respectively
def func_cos_node(offset,coef):
    l = offset
    r = len(coef)-1-l
    theta = sp.symbols('theta')
    fcos = coef[offset]
    for k in range(-l,0):
        fcos = fcos + coef[offset+k] * sp.cos((-k)*theta)
    for k in range(1,r+1):
        fcos = fcos + coef[offset+k] * sp.cos(k*theta)
    return fcos

def func_sin_node(offset,coef):
    l = offset
    r = len(coef)-1-l
    theta = sp.symbols('theta')
    fsin = sp.Rational(0,1)
    for k in range(-l,0):
        fsin = fsin - coef[offset+k] * sp.sin((-k)*theta)
    for k in range(1,r+1):
        fsin = fsin + coef[offset+k] * sp.sin(k*theta)
    return fsin

def plot(var_name,val0,val1,func,plotZero,nsample):
    if nsample <= 0:
        nsample = 2001
    var = np.linspace(val0,val1,nsample)
    val = np.zeros(nsample)
    for i in range(0,nsample):
        val[i] = sp.Float(func.subs(sp.symbols(var_name),var[i]))
    plt.xlim(val0,val1)
    plt.plot(var,val,'b-')
    if plotZero:
        plt.plot([val0, val1],[0, 0],'r-')
    plt.xlabel(r'$'+'\\'+var_name+'$')
    plt.show()

def plot_theta(val0,val1,func,plotZero,nsample):
    plot('theta',val0,val1,func,plotZero,nsample)
