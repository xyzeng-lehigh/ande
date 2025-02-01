import numpy as np
import sympy as sp

# compute coefficients 
#   a_k -l \le k \le r 
# such that
#   du/dx_j = \sum_k a_k u_{j+k}
# has optimal order p=l+r
# 
# coef[offset+k] = a_k, -l \le k \le r
def calc_coef_dx(l,r):
    offset = l
    coef   = np.zeros(l+r+1)
    return offset, coef
