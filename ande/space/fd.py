import sys, os.path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),'../')))
import utility.constants
import utility.functions

import numpy as np
import sympy as sp

# compute coefficients 
#   beta_k -l \le k \le r 
# such that
#   du/dx_j = (1/h) \sum_k beta_k u_{j+k}
# has optimal order p=l+r
# 
# coef[offset+k] = beta_k, -l \le k \le r
def calc_coef_dx(l,r):
    offset = l
    coef   = []
    for k in range(-l,0):
        coef.append( utility.constants.power_sign(k-1) * utility.constants.binomial_normal(l,r,k) / sp.Integer( k ) )
    coef.append( utility.constants.harmonic_diff(l,r) )
    for k in range(1,r+1):
        coef.append( utility.constants.power_sign(k-1) * utility.constants.binomial_normal(l,r,k) / sp.Integer( k ) )
    return offset, coef

# Let the solution to the semi-discretized system with initial data exp(-i kappa x) be
#   u_j = A(t)exp(-ij theta), A'(t)/A(t) = -(c/h)mu
# The next two functions plot Re mu and Im mu as functions of theta = kappa h
#   Re mu = \sum_k beta_k cos(k\theta)
#   Im mu = - \sum_k beta_k sin(k\theta)
# func_* return function handles
# plot_* plot the functions on the interval [0, 2 pi]
def func_dx_real(l,r):
    offset, coef = calc_coef_dx(l,r)
    real_mu = utility.functions.func_cos_node(offset,coef)
    return real_mu

def func_dx_imag(l,r):
    offset, coef = calc_coef_dx(l,r)
    imag_mu = - utility.functions.func_sin_node(offset,coef)
    return imag_mu

def plot_dx_real(l,r,plotZero=True,nsample=-1):
    real_mu = func_dx_real(l,r)
    utility.functions.plot_theta(0,2*np.pi,real_mu,plotZero,nsample)

def plot_dx_imag(l,r,plotZero=False,nsample=-1):
    imag_mu = func_dx_imag(l,r)
    utility.functions.plot_theta(0,2*np.pi,imag_mu,plotZero,nsample)
