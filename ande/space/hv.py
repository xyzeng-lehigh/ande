import sys, os.path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),'../')))
import utility.constants
import utility.functions
import utility.canvas

import numpy as np
import sympy as sp
import math

# compute four-digit stencil lc, rc, ln, rn 
# given l = lc+ln and r = rc+rn
def calc_stencil(stencil):
    l = stencil[0]
    r = stencil[1]
    ln = math.floor(l/2)
    lc = l - ln
    rn = math.floor(r/2)
    rc = r - rn
    return [lc,rc,ln,rn]

# compute coefficients
#   alpha_k -lc \le k \le lc-1 and beta_k -ln \le k \le rn
# such that
#   du/dx_j =   (1/h) \sum_k \alpha_k \overline{u}_{j+k+1/2} 
#             + (1/h) \sum_k \beta_k u_{j+k}
# has the optimal order p=bl+br+l+r

# coef_c[offset_c+k] = alpha_k, -lc \le k \le rc-1
# coef_n[offset_n+k] = beta_k, -ln \le k \le rn
def calc_coef_dx(stencil):
    offset_c, coef_c = calc_coef_dx_cell(stencil)
    offset_n, coef_n = calc_coef_dx_node(stencil)
    return offset_c, coef_c, offset_n, coef_n

def calc_coef_dx_cell(stencil):
    if len(stencil)==4:
        [lc,rc,ln,rn] = stencil
    else:
        [lc,rc,ln,rn] = calc_stencil(stencil)
    offset = lc
    coef = []
    # Part 1: alpha_{-lc,...,-1}
    if lc != ln:
        val = - sp.Rational(2,lc*lc) * utility.constants.binomial_normal(lc,rc,-lc) * utility.constants.binomial_normal(lc,rn,-lc) 
        coef.append( val )
    else:
        val = sp.Integer(0)
    for k in range(-ln,0):
        coef.append( val - sp.Rational(2,k*k)*(sp.Integer(1)+sp.Integer(k)*(utility.constants.harmonic_diff(lc+k,rc-k)+utility.constants.harmonic_diff(ln+k,rn-k)))*utility.constants.binomial_normal(lc,rc,k)*utility.constants.binomial_normal(ln,rn,k) )
        val = coef[-1]
    # Part 2: alpha_{0,...,lc-1}
    # Part 2-1: alpha_rev{0,...,lc-1} such that
    #   alpha_rev_0 = alpha_{lc-1} ... alpha_rev_{lc-1} = alpha_{0}
    coef_rev = []
    if rc != rn:
        val = sp.Rational(2,rc*rc) * utility.constants.binomial_normal(lc,rc,rc) * utility.constants.binomial_normal(ln,rc,rc)
        coef_rev.append( val )
    else:
        val = sp.Integer(0)
    for k in range(rn,0,-1):
        coef_rev.append( val + sp.Rational(2,k*k)*(sp.Integer(1)+sp.Integer(k)*(utility.constants.harmonic_diff(lc+k,rc-k)+utility.constants.harmonic_diff(ln+k,rn-k)))*utility.constants.binomial_normal(lc,rc,k)*utility.constants.binomial_normal(ln,rn,k) )
        val = coef_rev[-1]
    # Part 2-2: copy from alpha_rev to alpha
    for k in range(0,len(coef_rev)):
        coef.append( coef_rev[-1-k] )
    return offset, coef

def calc_coef_dx_node(stencil):
    if len(stencil)==4:
        [lc,rc,ln,rn] = stencil
    else:
        [lc,rc,ln,rn] = calc_stencil(stencil)
    offset = ln
    coef = []
    for k in range(-ln,0):
        coef.append( - sp.Rational(2,k) * utility.constants.binomial_normal(lc,rc,k) * utility.constants.binomial_normal(ln,rn,k) )
    coef.append( sp.Integer(2) * (utility.constants.harmonic_diff(lc,rc) + utility.constants.harmonic_diff(ln,rn)) )
    for k in range(1,rn+1):
        coef.append( - sp.Rational(2,k) * utility.constants.binomial_normal(lc,rc,k) * utility.constants.binomial_normal(ln,rn,k) )
    return offset, coef

# For stability analysis, we define:
#   beta(z) = \sum_k beta_k z^k
# and 
#   alpha(z) = \sum_k alpha_k z^k
# Then stability of the semi-discretized method is equivalent to two conditions:
#   (1) Re beta >= 0
#   (2) Re alpla Re \bar{beta}(z-1)alpha + [Im (z-1)alpha]^2 <= 0
# when z=e^{i theta}
def func_dx_real_beta(stencil):
    offset, coef = calc_coef_dx_node(stencil)
    real_beta = utility.functions.func_cos_node(offset,coef)
    return real_beta

def func_dx_imag_beta(stencil):
    offset, coef = calc_coef_dx_node(stencil)
    imag_beta = utility.functions.func_sin_node(offset,coef)
    return imag_beta

def plot_dx_real_beta(stencil,plotZero=True,nsample=-1):
    real_beta = func_dx_real_beta(stencil)
    utility.functions.plot_theta(0,2*np.pi,real_beta,plotZero,nsample)

def plot_dx_imag_beta(stencil,plotZero=False,nsample=-1):
    imag_beta = func_dx_imag_beta(stencil)
    utility.functions.plot_theta(0,2*np.pi,imag_beta,plotZero,nsample)
