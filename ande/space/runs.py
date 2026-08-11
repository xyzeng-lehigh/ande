import hv
import fd
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import math

def list_coef_reH_at_first_min(k,t,d,toNormalize=True,useDouble=True,toPlot=True):
    # k should be a number between 0 and floor((d-1)/2)
    stencil = [t+2*d,t,t+2*d,t]
    offset, coef = hv.calc_coef_dx_node(stencil)
    multiplier = sp.Integer(1)
    if toNormalize:
        multiplier = abs(1/coef[offset])
    sequence = []
    part_sum = []
    if k == 0:
        M = math.floor(t/d)+1
        sequence.append( multiplier * coef[offset] )
        part_sum.append( sequence[-1] )
        for m in range(1,M):
            sequence.append( multiplier * hv.utility.constants.power_sign(m) * (coef[offset+m*d]+coef[offset-m*d]) )
            part_sum.append( part_sum[-1] + sequence[-1] )
        for m in range(M,M+2):
            sequence.append( multiplier * hv.utility.constants.power_sign(m) * coef[offset-m*d] )
            part_sum.append( part_sum[-1] * sequence[-1] )
    double_seq = []
    double_p_s = []
    if useDouble or toPlot:
        for m in range(0,len(sequence)):
            double_seq.append(float(sequence[m]))
        for m in range(0,len(part_sum)):
            double_p_s.append(float(part_sum[m]))
        if useDouble:
            sequence = double_seq
            part_sum = double_p_s
    # Print to screen
    # Plot to canvas
    return sequence, part_sum

def plot_reH_fix_stencil(L):
    tmax = math.floor((L-1)/2)
    tmin = 0
    for t in range(0,tmax+1):
        s = L-2*t
        real_beta = hv.func_dx_real_beta([2*t+2*s,2*t])
        hv.utility.functions.plot_theta(0,2*np.pi,real_beta,plotZero=True)
    plt.show()

def compute_scaled_reH_zero(T,normalize=True):
    # T = t + d, compute scaled Re H(0) for d ranging from 1 to T
    # val[d-1] = d * Re H(0) * (binom{2t+2d}{t+2d})^2
    rat = []
    val = []
    for k in range(0,T):
        d = k + 1
        t = T - d
        offset, coef = hv.calc_coef_dx_node([t+2*d,t,t+2*d,t]) 
        rat.append( sp.Integer(0) )
        for j in range(0,len(coef)):
            rat[-1] = rat[-1] + coef[j]
        rat[-1] = d * rat[-1]
        for j in range(0,t):
            rat[-1] *= ((t+2*d+1+j)/(j+1)) * ((t+2*d+1+j)/(j+1))
        val.append( float(rat[-1]) )
    if normalize:
        mag = rat[0]
        for j in range(0,len(rat)):
            rat[j] = rat[j]/mag
            val[j] = val[j]/float(mag)
    return rat, val

def plot_reH_fix_dnw(t,dmax,normalize=True,toScale=True):
    colors=['b','c','g','m','r']
    for d in range(1,dmax+1):
        real_beta = hv.func_dx_real_beta([t+2*d,t,t+2*d,t])
        if normalize:
            real_beta = real_beta/sp.Float(real_beta.subs(sp.symbols('theta'),0))
        if toScale:
            hv.utility.functions.plot_theta(0,2*np.pi,real_beta,1.0/d,True,-1,colors[(d-1)%len(colors)])
        else:
            hv.utility.functions.plot_theta(0,2*np.pi,real_beta,1.0,True,-1,colors[(d-1)%len(colors)])
    plt.show()

def compute_reH_cheby_coef(t,s):
    offset, hv_coef = hv.calc_coef_dx_node([t+s,t,t+s,t])
    coef = []
    for n in range(0,t+s+1):
        coef.append( sp.Integer(0) )
    for n in range(0,len(hv_coef)):
        coef[0] = coef[0] + hv_coef[n]
    for k in range(1,t+1):
        cheby_coef = hv.utility.constants.cheby_coef(k)
        for n in range(1,k+1):
            coef[n] = coef[n] + cheby_coef[n]*(hv_coef[offset+k]+hv_coef[offset-k])
    for k in range(t+1,t+s+1):
        cheby_coef = hv.utility.constants.cheby_coef(k)
        for n in range(1,k+1):
            coef[n] = coef[n] + cheby_coef[n]*hv_coef[offset-k]
    real_coef = []
    for n in range(0,t+s+1):
        real_coef.append(float(coef[n]))
    return coef, real_coef

def plot_reH_alt(t,s):
    coefs = []
    coefs.append( sp.Integer(0) )
    for k in range(1,t+s+1):
        temp = hv.utility.constants.binomial_normal(t+s,t,-k) 
        temp = temp * temp
        coefs.append( sp.Rational(2,k) * temp )
        coefs[0] = coefs[0] - 4*(hv.utility.constants.harmonic_num(t+s-k)-hv.utility.constants.harmonic_num(t+k))*temp
    for k in range(1,t+1):
        temp = hv.utility.constants.binomial_normal(t+s,t,k) 
        temp = temp * temp
        coefs[k] = coefs[k] - sp.Rational(2,k) * temp
        coefs[0] = coefs[0] - 4*(hv.utility.constants.harmonic_num(t+s+k)-hv.utility.constants.harmonic_num(t-k))*temp
    cos_fun = hv.utility.functions.func_cos_node(0,coefs)
    hv.utility.functions.plot_theta(0,2*np.pi,cos_fun,1.0,True,-1)
    plt.show()

def plot_reH_diff_fd(t,d):
    s = 2*d
    coefs = []
    coefs.append( sp.Integer(0) )
    for k in range(1,t+1):
        val_p = hv.utility.constants.binomial_normal(t+s,t,k)
        val_m = hv.utility.constants.binomial_normal(t+s,t,-k)
        coefs.append( (hv.utility.constants.power_sign(k)*(val_m-val_p)/sp.Integer(k)) )
        coefs[-1] = coefs[-1] * (2-hv.utility.constants.power_sign(k)*(val_m+val_p))
    for k in range(t+1,t+s+1):
        val_m = hv.utility.constants.binomial_normal(t+s,t,-k)
        coefs.append( (hv.utility.constants.power_sign(k)*val_m/sp.Integer(k)) )
        coefs[-1] = coefs[-1] * (2-hv.utility.constants.power_sign(k)*val_m)
    cos_fun = hv.utility.functions.func_cos_node(0,coefs)
    hv.utility.functions.plot_theta(0,2*np.pi,cos_fun,1.0,True,-1)
    hv.utility.functions.add_vertical_lines([np.pi/d])
    val = sp.Float(cos_fun.subs('theta',np.pi/d))
    plt.show()
    real_coefs = []
    for k in range(0,len(coefs)):
        real_coefs.append( float(coefs[k]) )
    return real_coefs, val

def calc_reH_diff_hv_min(t,d):
    s = 2*d
    offset, coef = hv.calc_coef_dx_node([t+s,t,t+s,t])
    val0 = sp.Float(coef[offset])
    reH = hv.func_dx_real_beta([t+s,t,t+s,t])
    val1 = sp.Float(reH.subs('theta',np.pi/d))
    return val0-val1

def calc_reH_diff_fdm_min(t,d):
    s = 2*d
    offset, coef = fd.calc_coef_dx([t+s,t])
    val0 = sp.Float(coef[offset])
    reH = fd.func_dx_real([t+s,t])
    val1 = sp.Float(reH.subs('theta',np.pi-np.pi/d))
    return val0-val1

def calc_reH_diff_fd_multiple(m,d):
    # stencil = [md+2d,md,md+2d,md]
    # these coefficients multiply the original by d
    coefs_sm = []
    val = hv.utility.constants.rf_ratio(m*d+d+1,m*d+1,d)
    coefs_sm.append( val*(val-2*hv.utility.constants.power_sign(d)) )
    for k in range(2,m+3):
        val = hv.utility.constants.rf_ratio((m+2-k)*d+1,(m+2)*d+1,(k-2)*d)
        coefs_sm.append( val*(val-2*hv.utility.constants.power_sign(k*d)) )
        coefs_sm[-1] = hv.utility.constants.power_sign(k+1) * coefs_sm[-1]/k
    coefs_sp = []
    for k in range(1,m+1):
        val = hv.utility.constants.rf_ratio((m-k)*d+1,(m+2)*d+1,k*d)
        coefs_sp.append( val*(val-2*hv.utility.constants.power_sign(k*d)) )
        coefs_sp[-1] = hv.utility.constants.power_sign(k) * coefs_sp[-1]/k
    real_coef_sm = []
    for k in range(0,len(coefs_sm)):
        real_coef_sm.append( float(coefs_sm[k]) )
    real_coef_sp = []
    for k in range(0,len(coefs_sp)):
        real_coef_sp.append( float(coefs_sp[k]) )
    return real_coef_sm, real_coef_sp

def calc_constant_1(l,r):
    var = sp.Integer(0)
    for k in range(-l,r+1):
        var = var + hv.utility.constants.harmonic_diff(l+k,r-k)*hv.utility.constants.binomial_normal(l,r,k)*hv.utility.constants.binomial_normal(l,r,k)
    return var

def plot_diff_hweno_curves(l,r):
    offset, coef_hv = hv.calc_coef_dx_node([l,r,l,r])
    real_beta = hv.func_dx_real_beta([l,r,l,r])
    hv.utility.functions.plot_theta(0,2*np.pi,real_beta,1.0,True,-1,color='b')

    b0 = sp.Integer(0)
    for k in range(0,len(coef_hv)):
        b0 = b0 + coef_hv[k]

    denom = hv.utility.constants.binomial(2*l+2*r,l+r)

    coef1 = []
    for k in range(0,l+1):
        val = hv.utility.constants.binomial(l+r,l-k)
        coef1.append(-b0*val*val/denom)
    for k in range(1,r+1):
        val = hv.utility.constants.binomial(l+r,l+k)
        tmp = -b0*val*val/denom
        if k > l:
            coef1.append(tmp)
        else:
            coef1[k] = coef1[k] + tmp

    coef2 = []
    for k in range(0,l+1):
        val = hv.utility.constants.binomial(l+r,l-k)
        coef2.append(2*hv.utility.constants.harmonic_diff(l-k,r+k)*val*val/denom)
    for k in range(1,r+1):
        val = hv.utility.constants.binomial(l+r,l+k)
        tmp = 2*hv.utility.constants.harmonic_diff(l+k,r-k)*val*val/denom
        if k > l:
            coef2.append(tmp)
        else:
            coef2[k] = coef2[k] + tmp

    cos_fun1 = hv.utility.functions.func_cos_node(0,coef1)
    cos_fun2 = hv.utility.functions.func_cos_node(0,coef2)
    hv.utility.functions.plot_theta(0,2*np.pi,cos_fun1,1.0,True,-1,color='c')
    hv.utility.functions.plot_theta(0,2*np.pi,cos_fun2,1.0,True,-1,color='m')

    hv.utility.functions.plot_theta_sum_products(0,2*np.pi,[[real_beta],[cos_fun1],[cos_fun2]],1.0,True,-1,color='k')
    plt.show()

def calc_diff_hweno_pi(l,r):
    val1 = sp.Integer(0)
    for k in range(-l,r+1):
        tmp = hv.utility.constants.binomial_normal(l,r,k)
        val1 = val1 + hv.utility.constants.harmonic_diff(l+k,r-k)*tmp*tmp*hv.utility.constants.power_sign(k)

    val2 = sp.Integer(0)
    for k in range(-l,r+1):
        tmp = hv.utility.constants.binomial_normal(l,r,k)
        val2 = val2 + 2*hv.utility.constants.harmonic_num(l+k)*tmp*tmp*hv.utility.constants.power_sign(k)
    return [val1, val2], [float(val1), float(val2)]

def calc_binom_square_harmonic_odd(n,scale=False):
    val = sp.Integer(0)
    for k in range(1,2*n+2):
        tmp = hv.utility.constants.binomial(2*n+1,k)
        val = val + hv.utility.constants.power_sign(k) * hv.utility.constants.harmonic_num(k) * tmp * tmp
    multiplier = sp.Integer(1)
    if scale:
        multiplier = hv.utility.constants.binomial(2*n+1,n)*(n+1)
    return val * multiplier

def calc_binom_harmonic(n):
    val = sp.Integer(0)
    for k in range(1,n+1):
        tmp = hv.utility.constants.binomial(n,k)
        val = val + hv.utility.constants.power_sign(k) * hv.utility.constants.harmonic_num(k) * tmp
    multiplier = sp.Integer(1)
    return val * multiplier

def calc_weno_poly_coef(n):
    x = sp.symbols('x')
    p = sp.Integer(0)
    for k in range(1,n+1):
        p = p+sp.Rational(1,k)*sp.Pow(1-x*x,n-k)*sp.Pow(1-x,k)
    p = sp.expand(p)
    #p = sp.Poly(p)
    #return p.nth(n)
    return p

def calc_weno_twopoly(n):
    x = sp.symbols('x')
    y = sp.symbols('y')
    p = sp.Pow(1-x,n)*sp.Pow(1+x-y,n)
    p = sp.expand(p)
    p = sp.Poly(p)
    val = sp.Integer(0)
    for l in range(1,n+1):
        val = val + p.nth(n-l,l)/l
    return val

def calc_weno_sum_twopoly(n):
    x = sp.symbols('x')
    p = -hv.utility.constants.harmonic_num(n)*sp.Pow(1-x*x,n)
    for k in range(1,n+1):
        p = p + sp.Pow(1-x*x,n-k)*sp.Pow(1-x,k)/k
    val = 0
    for m in range(0,n):
        w = sp.exp(2*m*sp.I*np.pi/n)
        val = val + p.subs(x,w)
    val = sp.simplify(val)
    return val/n-hv.utility.constants.harmonic_num(n)

def plot_ade_dx_cent_diff(lc,ln):
    L = lc+ln
    offset_c, coef_c, offset_n, coef_n = hv.calc_coef_dx([L,L])
    theta = sp.symbols('theta')
    fun_rhs = sp.sin(theta)*hv.utility.functions.func_sin_node_gen(offset_c,lc,coef_c,1,2)
    fun_lhs = -hv.utility.functions.func_sin_node_gen(offset_n+1,ln,coef_n,2,2)
    hv.utility.functions.plot_theta(0,np.pi,fun_lhs,1.0,True,-1,'b');
    hv.utility.functions.plot_theta(0,np.pi,fun_rhs,1.0,True,-1,'r');
    plt.show()

def plot_ade_dxx_cent_diff(lc,ln):
    L = lc+ln
    offset_c, coef_c, offset_n, coef_n = hv.calc_coef_dxx([L])
    theta = sp.symbols('theta')
    fun_rhs = -sp.sin(theta/2)*hv.utility.functions.func_cos_node(offset_n,coef_n)
    fun_lhs = hv.utility.functions.func_sin_node_gen(offset_c,offset_c,coef_c,1,1)-hv.utility.functions.func_sin_node_gen(offset_c+1,offset_c-1,coef_c,1,1)
    #hv.utility.functions.plot_theta(0,2*np.pi,fun_lhs,1.0,True,-1,'b');
    #hv.utility.functions.plot_theta(0,2*np.pi,fun_rhs,1.0,True,-1,'r');
    hv.utility.functions.plot_theta(0,np.pi,fun_rhs-fun_lhs,1.0,True,-1,'g');
    plt.show()

def calc_ade_dxx_cent_diff(lc,ln):
    L = lc+ln
    offset_c, coef_c, offset_n, coef_n = hv.calc_coef_dxx([L])
    coef_c.append( sp.Integer(0) )
    coef_n.append( sp.Integer(0) )
    seq_orig = []
    for k in range(1,max(2*lc,2*ln+1)+1):
        if k%2 == 1:
            m=int((k-1)/2)
            seq_orig.append( float(coef_n[offset_n+m+1]-coef_n[offset_n+m]) )
        else:
            m=int(k/2)
            seq_orig.append( float(coef_c[offset_c+m-1]-coef_c[offset_c+m]) )
    seq_viet = []
    for k in range(0,len(seq_orig)):
        seq_viet.append( (k+1)*seq_orig[k] )
    seq_c    = []
    for k in range(0,len(seq_orig)):
        seq_c.append( seq_orig[k] )
    for k in range(1,len(seq_orig)):
        seq_c[len(seq_orig)-1-k] = seq_c[len(seq_orig)-1-k] - seq_c[len(seq_orig)-k]
    seq_b = []
    for k in range(0,ln+2):
        seq_b.append( float(coef_n[offset_n+k]) )
    seq_a = []
    for k in range(0,lc+1):
        seq_a.append( float(coef_c[offset_c+k]) )
    return seq_a, seq_b, seq_orig, seq_viet, seq_c

def calc_analysis_dx_cent(L,toPlot=True): # stencil = [L,L]
  [lc,rc,ln,rn] = hv.calc_stencil([L,L])
  offset_c, coef_c, offset_n, coef_n = hv.calc_coef_dx([L,L])
  theta = sp.symbols('theta')
  func_dx_alpha = hv.utility.functions.func_sin_node_gen(offset_c,lc,coef_c,phase=1,stride=2)
  func_dx_beta  = -hv.utility.functions.func_sin_node_gen(offset_n+1,ln,coef_n,phase=2,stride=2)
  if toPlot:
    hv.utility.functions.plot_theta(0,np.pi,func_dx_alpha,plotZero=True,color='b')
    hv.utility.functions.plot_theta(0,np.pi,func_dx_beta,plotZero=True,color='g')
    plt.show()
  return func_dx_alpha, func_dx_beta

def calc_analysis_dxx_cent(L,toPlot=True): # stencil = [L]
  [lc,ln] = hv.calc_stencil_dxx([L])
  offset_c, coef_c, offset_n, coef_n = hv.calc_coef_dxx([L])
  theta = sp.symbols('theta')
  func_dxx_alpha = hv.utility.functions.func_sin_node_gen(offset_c,lc,coef_c,phase=2,stride=2) - hv.utility.functions.func_sin_node_gen(offset_c+1,lc-1,coef_c,phase=2,stride=2)
  func_dxx_beta  = - coef_n[offset_n] - 2 * hv.utility.functions.func_cos_node_gen(offset_n+1,ln,coef_n,phase=2,stride=2)
  if toPlot:
    hv.utility.functions.plot_theta(0,np.pi,func_dxx_alpha,plotZero=True,color='b')
    hv.utility.functions.plot_theta(0,np.pi,func_dxx_beta,plotZero=True,color='g')
    plt.show()
  return func_dxx_alpha, func_dxx_beta

def verify_analysis_cent(Lx,Lxx):
  func_dx_alpha,  func_dx_beta  = calc_analysis_dx_cent( Lx, toPlot=False)
  func_dxx_alpha, func_dxx_beta = calc_analysis_dxx_cent(Lxx,toPlot=False)
  theta = sp.symbols('theta')
  func_to_plot = sp.sin(theta) * func_dx_alpha * func_dxx_beta - func_dx_beta * func_dxx_alpha
  hv.utility.functions.plot_theta(0,np.pi,func_to_plot,plotZero=True)
  plt.show()

def plot_analysis_cent(Lx,Lxx,plt_option=0):
  func_dx_alpha,  func_dx_beta  = calc_analysis_dx_cent( Lx, toPlot=False)
  func_dxx_alpha, func_dxx_beta = calc_analysis_dxx_cent(Lxx,toPlot=False)
  theta = sp.symbols('theta')
  match plt_option:
    case 1:
      func_to_plot_dx  = func_dx_alpha-theta*func_dx_beta
      func_to_plot_dxx = sp.sin(theta)*func_dxx_beta-func_dxx_alpha
    case 2:
      func_to_plot_dx  = sp.sin(theta)*func_dx_alpha-theta*func_dx_beta
      func_to_plot_dxx = func_dxx_beta-func_dxx_alpha
    case 3:
      func_to_plot_dx  = func_dx_alpha-sp.cos(theta)*func_dx_beta
      func_to_plot_dxx = sp.sin(theta)*sp.cos(theta)*func_dxx_beta-func_dxx_alpha
    case 4:
      func_to_plot_dx  = sp.cos(theta)*func_dx_alpha-func_dx_beta
      func_to_plot_dxx = sp.sin(theta)*func_dxx_beta-sp.cos(theta)*func_dxx_alpha
    case _:
      func_to_plot_dx  = sp.sin(theta)*func_dx_alpha-theta*func_dx_beta
      func_to_plot_dxx = theta*func_dxx_beta-func_dxx_alpha
  hv.utility.functions.plot_theta(0,np.pi,func_to_plot_dx,plotZero=True,color='b')
  hv.utility.functions.plot_theta(0,np.pi,func_to_plot_dxx,plotZero=False,color='g')
  plt.show()

# (B.7)
def calc_analysis_cent_seq_dxx_cvx(L,plotTrigSum=False):
  offset_c_dxx, coef_c_dxx, offset_n_dxx, coef_n_dxx = hv.calc_coef_dxx([L])
  lc, ln = hv.calc_stencil_dxx([L]) # ln <= lc
  coef_n_dxx.append(sp.Integer(0))
  coef_c_dxx.append(sp.Integer(0))
  seq = []
  vietSeq = []
  cvxSeq = []
  momSeq1 = 0.0
  momSeq3 = 0.0
  momSeqSign = 1.0
  for k in range(0,max(2*lc,2*ln+1)):
    seq.append(0.0)
  for k in range(0,ln+1):
    seq[2*k] = float( - coef_n_dxx[offset_n_dxx+k] + coef_n_dxx[offset_n_dxx+k+1])
  for k in range(0,lc):
    seq[2*k+1] = float(coef_c_dxx[offset_c_dxx+k] - coef_c_dxx[offset_c_dxx+k+1])
  for k in range(0,len(seq)):
    vietSeq.append((k+1)*seq[k])
  if len(seq)>2:
    for k in range(0,len(seq)-2):
      cvxSeq.append(seq[k]-2*seq[k+1]+seq[k+2])
    cvxSeq.append(seq[-2]-4*seq[-1])
  for k in range(0,len(seq)):
    momSeq1 = momSeq1 - momSeqSign*(k+1)*seq[k]
    momSeq3 = momSeq3 + momSeqSign*(k+1)*(k+1)*(k+1)*seq[k]/6.0
    momSeqSign = -momSeqSign
  if plotTrigSum:
    fun_sin = hv.utility.functions.func_sin_node_gen(0,len(seq),seq,1,1)
    hv.utility.functions.plot_theta(0,np.pi,fun_sin,1.0,True,-1,'r');
    plt.show()
  return seq, vietSeq, cvxSeq, momSeq1, momSeq3

# (B.8)
def calc_analysis_cent_seq_dx_cvx(L,plotTrigSum=False):
  offset_c_dx,  coef_c_dx,  offset_n_dx,  coef_n_dx  = hv.calc_coef_dx([L,L])
  lc, ln = hv.calc_stencil_dxx([L]) # ln <= lc
  seq = []
  vietSeq = []
  cvxSeq = []
  for k in range(0,max(2*lc-1,2*ln)):
    seq.append(0.0)
  for k in range(0,lc):
    seq[2*k] = float(coef_c_dx[offset_c_dx+k])
  for k in range(0,ln):
    seq[2*k+1] = float(-coef_n_dx[offset_n_dx+k+1])
  for k in range(0,len(seq)):
    vietSeq.append((k+1)*seq[k])
  if len(seq)>2:
    for k in range(0,len(seq)-2):
      cvxSeq.append(seq[k]-2*seq[k+1]+seq[k+2])
    cvxSeq.append(seq[-2]-4*seq[-1])
  if plotTrigSum:
    fun_sin = hv.utility.functions.func_sin_node_gen(0,len(seq),seq,1,1)
    hv.utility.functions.plot_theta(0,np.pi,fun_sin,1.0,True,-1,'r');
    plt.show()
  return seq, vietSeq, cvxSeq

# (4.27)
def calc_analysis_cent_seq_dx(L,plotTrigSum=False):
  offset_c_dx,  coef_c_dx,  offset_n_dx,  coef_n_dx  = hv.calc_coef_dx([L,L])
  lc, ln = hv.calc_stencil_dxx([L]) # ln <= lc
  coef_c_dx.append(sp.Integer(0))
  if ln < lc:
    coef_n_dx.append(sp.Integer(0))
  seq = []
  for k in range(0,lc):
    seq.append(0.0)
  for k in range(0,lc):
    seq[k] = seq[k] + coef_c_dx[offset_c_dx+k] + coef_c_dx[offset_c_dx+k+1] + 2*coef_n_dx[offset_n_dx+k+1]
  if plotTrigSum:
    fun_sin = hv.utility.functions.func_sin_node_gen(0,len(seq),seq,1,1)
    hv.utility.functions.plot_theta(0,np.pi,fun_sin,1.0,True,-1,'r');
    plt.show()
  return seq

def calc_coef_cent_ade(aLx,dLx,dLxx):
  fc = -hv.func_dx_real_alpha([dLx,dLx])
  f  = -hv.func_dx_real_alpha([aLx,aLx])
  hc = -hv.func_dx_imag_beta([dLx,dLx])
  h  = -hv.func_dx_imag_beta([aLx,aLx])
  c  =  hv.func_dxx_imag_alpha([dLxx])
  b  = -hv.func_dxx_real_beta([dLxx])
  delta = f*b-c*h
  coef0 = (fc+b)*(fc+b)*(fc*b-c*hc)
  coef2_1 = f*(fc+b)*(fc+b)
  coef2_2 = (f*hc+c+h*b)*(h*fc-c-f*hc)
  coef2 = coef2_1+coef2_2
  coef2_2_2 = h*fc-f*hc
  #coef2 = f*(fc+b)*(fc+b)+(f*hc+c+h*b)*(h*fc-c-f*hc)
  #hv.utility.functions.plot_theta(0,np.pi,coef0,plotZero=False,color='b')
  #hv.utility.functions.plot_theta(0,np.pi,coef2,plotZero=False,color='g')
  #hv.utility.functions.plot_theta(0,np.pi,coef2_1,plotZero=False,color='y')
  #hv.utility.functions.plot_theta(0,np.pi,coef2_2,plotZero=False,color='c')
  #hv.utility.functions.plot_theta(0,np.pi,coef2_2_2,plotZero=False,color='m')
  hv.utility.functions.plot_theta(0,np.pi,f*f*f,plotZero=False,color='b')
  hv.utility.functions.plot_theta(0,np.pi,2*delta*f,plotZero=False,color='g')
  hv.utility.functions.plot_theta(0,np.pi,delta*b,plotZero=False,color='c')
  hv.utility.functions.plot_theta(0,np.pi,f*c*h,plotZero=False,color='m')
  hv.utility.functions.plot_theta(0,np.pi,c*c,plotZero=False,color='r')
  plt.show()
  return coef0, coef2

def calc_coef_ratio_dx(L):
  f = -hv.func_dx_real_alpha([L,L])
  h = -hv.func_dx_imag_beta([L,L])
  theta = sp.symbols('theta')
  gamma = theta*(1-theta*theta/8/np.pi/np.pi)
  #hv.utility.functions.plot_theta(0,np.pi,f-h*theta,plotZero=True,color='k')
  #hv.utility.functions.plot_theta(0,np.pi,theta*(theta+h)-f,plotZero=True,color='b')
  hv.utility.functions.plot_theta(0,np.pi,gamma*(gamma+h)-f,plotZero=True,color='b')
  plt.show()
  return

def calc_coef_ratio_dxx(L):
  c =  hv.func_dxx_imag_alpha([L])
  b = -hv.func_dxx_real_beta([L])
  theta = sp.symbols('theta')
  gamma = theta*(1-theta*theta/8/np.pi/np.pi)
  #hv.utility.functions.plot_theta(0,np.pi,b*theta-c,plotZero=True,color='k')
  hv.utility.functions.plot_theta(0,np.pi,b*gamma-c,plotZero=True,color='b')
  plt.show()
  return
