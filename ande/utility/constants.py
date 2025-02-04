import sympy as sp

# Harmonic number
#   H_n = 1 + 1/2 + ... + 1/n
def harmonic_num(n):
    H = sp.Rational(0,1)
    for i in range(1,n+1):
        H = H + sp.Rational(1,i)
    return H

# Harmonic difference
#   H_m - H_n
def harmonic_diff(m,n):
    H = sp.Rational(0,1)
    if m > n:
        for i in range(n+1,m+1):
            H = H + sp.Rational(1,i)
    elif m < n:
        for i in range(m+1,n+1):
            H = H - sp.Rational(1,i)
    return H

# Normalized binomial coefficients
#   C^{l,r}_k = l!r!/(l+k)!(r-k)!
def binomial_normal(l,r,k):
    C = sp.Integer(1)
    if k > 0:
        for i in range(1,k+1):
            C = C * sp.Rational(r-k+i,l+i)
    elif k < 0:
        for i in range(1,-k+1):
            C = C * sp.Rational(l+k+i,r+i)
    return C

# Powers of sign
#   (-1)^n
def power_sign(n):
    S = sp.Integer(1)
    if n%2 == 1:
        S = -S
    return S
