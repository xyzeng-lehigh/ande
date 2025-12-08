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

# Given the coefficients a_k, offset <= k < offset+num
# The next two functions compute:
#   sum_{k=0}^{num-1} a_{k+offset} cos((phase + k * stride) * theta)
# and
#   sum_{k=0}^{num-1} a_{k+offset} sin((phase + k * stride) * theta)
# respectively
def func_cos_node_gen(offset,num,coef,phase=0,stride=1):
  theta = sp.symbols('theta')
  fcos = 0*theta
  for k in range(0,num):
    fcos = fcos + coef[offset+k] * sp.cos((phase+k*stride)*theta)
  return fcos

def func_sin_node_gen(offset,num,coef,phase=0,stride=1):
  theta = sp.symbols('theta')
  fsin = 0*theta
  for k in range(0,num):
    fsin = fsin + coef[offset+k] * sp.sin((phase+k*stride)*theta)
  return fsin

def plot(var_name,val0,val1,func,scale=1.0,plotZero=True,nsample=-1,color='b'):
  if nsample <= 0:
    nsample = 2001
  var = np.linspace(val0,val1,nsample)
  val = np.zeros(nsample)
  for i in range(0,nsample):
    val[i] = sp.Float(func.subs(sp.symbols(var_name),scale*var[i]))
  plt.xlim(val0,val1)
  plt.plot(var,val,color+'-')
  if plotZero:
    plt.plot([val0, val1],[0, 0],'r-')
  plt.xlabel(r'$'+'\\'+var_name+'$',fontsize=18)
  #plt.show()

def plot_theta(val0,val1,func,scale=1.0,plotZero=True,nsample=-1,color='b'):
  plot('theta',val0,val1,func,scale,plotZero,nsample,color)

def plot_theta_products(val0,val1,funcs,scale=1.0,plotZero=True,nsample=-1,color='b'):
  func = sp.Integer(1)
  for k in range(0,len(funcs)):
    func = func * funcs[k]
  plot('theta',val0,val1,func,scale,plotZero,nsample,color)

def plot_theta_sum_products(val0,val1,group_funcs,scale=1.0,plotZero=True,nsample=-1,color='b'):
  sfunc = sp.Integer(0)
  for n in range(0,len(group_funcs)):
    funcs = group_funcs[n]
    func = sp.Integer(1)
    for k in range(0,len(funcs)):
      func = func * funcs[k]
    sfunc = sfunc + func
  plot('theta',val0,val1,sfunc,scale,plotZero,nsample,color)

def plot_coef(index,coefs):
  plt.plot(index,coefs,'k.',markersize=12)
  #plt.show()

def add_vertical_lines(vMarkers):
  [y_min, y_max] = plt.gca().get_ylim()
  for xval in vMarkers:
    plt.plot([xval, xval],[y_min, y_max],'g--')

def sum_sequence(seq,limit=-1):
  val = 0.0
  if limit < 0:
    limit = len(seq)
  else:
    limit = min(limit,len(seq))
  for k in range(0,limit):
    val = val + float(seq[k])
  return val
