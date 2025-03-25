import hv
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
