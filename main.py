# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib
from scipy import interpolate
from scipy.optimize import curve_fit
from scipy.special import erf, erfc
from scipy.interpolate import UnivariateSpline
from scipy.optimize import fsolve
from scipy.integrate import simps, quad
from scipy.stats import linregress
from decimal import Decimal
import pandas as pd
from uncertainties import ufloat, unumpy
from uncertainties.umath import *
from subprocess import call
from datetime import date, datetime
import os
import helpers as hp
import sys
import re
import copy
#from simplegeneric import generic

#font = {'family' : 'normal',
#        'size'   :  12}
#matplotlib.rc('font', **font)

# Custom code starts here
hp.replace("Name", "Roman Gruber")
hp.replace("Experiment", "Semiconductors")

# constants
kB = 1.38064852*10**-23 # m^2 kg s^-2 K^-1
eV = 1.6021766208*10**-19

###################
# Operating point #
###################

delta_V_4 = hp.fetch2('data/operating_point.xlsx', 'err V_4 [mV]')
V_4 = hp.fetch2('data/operating_point.xlsx', 'V_4 [mV]', delta_V_4)/1000
delta_I = hp.fetch2('data/operating_point.xlsx', 'err I [mu A]')
I = hp.fetch2('data/operating_point.xlsx', 'I [mu A]', delta_I)/10**6
T = hp.fetch2('data/operating_point.xlsx', 'T [°C]', 0.1)

R_S = V_4/I

fig, ax1 = plt.subplots()

ax1.errorbar(hp.nominal(I)*10**3, hp.nominal(T), fmt='b.', xerr=hp.stddev(I)*10**3, yerr=hp.stddev(T))
ax1.set_xlabel('Current $I \,  [mA]$')
ax1.set_ylabel('Temperature $T \, [^{\circ}C]$', color='b')
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
ax2.errorbar([100000], [1000000], fmt='b.', xerr=0, yerr=0, label="measured temperature") # dummy for the label
ax2.errorbar(hp.nominal(I)*10**3, hp.nominal(R_S)/1000, fmt='r.', xerr=hp.stddev(I)*10**3, yerr=hp.stddev(R_S)/1000, label="measured resistance")
ax2.set_ylabel('Resistance $R_S \, [k\Omega]$', color='r')
ax2.tick_params('y', colors='r')

I_Op = 0.4007
V_2_Op = 3.250
hp.replace("I_Op", I_Op)
hp.replace("V_2_Op", V_2_Op)

ax2.axvline(I_Op, linewidth=2, color='black')
ax2.text(I_Op+0.025, 3.00, r'Operating point $V_{Op} = ' + hp.fmt_number(V_2_Op) + r' \, V$', rotation=90, verticalalignment='center')

ax2.fill_between([100000], [1000000], [1000000], facecolor="lightyellow", alpha=0.5, linewidth=0.0, label="safe zone") # dummy for the label
x = np.linspace(0.05, 0.6, 100)
ax1.fill_between(hp.nominal(x), np.ones(x.size)*30, np.ones(x.size)*0.0, facecolor="lightyellow", alpha=0.5, linewidth=0.0)
ax1.plot([0.05, 0.05], [0.0, 30.0], 'g--', linewidth=0.5)
ax1.plot([0.6, 0.6], [0.0, 30.0], 'g--', linewidth=0.5)

ax1.set_ylim(23.5, 26.0)
ax2.set_ylim(2.94, 3.06)
plt.xlim(-0.05, 2.0)
ax2.grid(True)
ax2.legend(loc="best")
ax1.grid(True)
fig.tight_layout()

plt.savefig('plots/operating_point.eps')

arr = hp.to_table(
	r'$T \, [^{\circ}C]$', T,
	r'$I \, [\mu A]$', I*10**6,
	r'$V \, [mV]$', V_4*10**3,
	r'$R_S \, [\Omega]$', R_S
)

hp.replace("table:operatingPoint", arr)

###############
# Temperature #
###############

delta_Tc = hp.fetch2('data/temperature_cooling.xlsx', 'err T [°C]')
Tc = hp.fetch2('data/temperature_cooling.xlsx', 'T [°C]', delta_Tc) + 273.15 # °K

sf_Ic = hp.fetch2('data/temperature_cooling.xlsx', 'I sf').astype(int)
delta_Ic = hp.fetch2('data/temperature_cooling.xlsx', 'err I [mA]')
Ic = hp.fetch2('data/temperature_cooling.xlsx', 'I [mA]', delta_Ic, 0, sf_Ic)/1000

sf_V_4c = hp.fetch2('data/temperature_cooling.xlsx', 'V_4 sf').astype(int)
delta_V_4c = hp.fetch2('data/temperature_cooling.xlsx', 'err V_4 [mV]')
V_4c = hp.fetch2('data/temperature_cooling.xlsx', 'V_4 [mV]', delta_V_4c, 0, sf_V_4c)/1000

R_Sc = V_4c/Ic

delta_Th = hp.fetch2('data/temperature_heating.xlsx', 'err T [°C]')
Th = hp.fetch2('data/temperature_heating.xlsx', 'T [°C]', delta_Th) + 273.15 # °K

sf_Ih = hp.fetch2('data/temperature_heating.xlsx', 'I sf').astype(int)
delta_Ih = hp.fetch2('data/temperature_heating.xlsx', 'err I [mA]')
Ih = hp.fetch2('data/temperature_heating.xlsx', 'I [mA]', delta_Ih, 0, sf_Ih)/1000

sf_V_4h = hp.fetch2('data/temperature_heating.xlsx', 'V_4 sf').astype(int)
delta_V_4h = hp.fetch2('data/temperature_heating.xlsx', 'err V_4 [mV]')
V_4h = hp.fetch2('data/temperature_heating.xlsx', 'V_4 [mV]', delta_V_4h, 0, sf_V_4h)/1000

R_Sh = V_4h/Ih

# T-R plot

plt.figure(0)
plt.errorbar(hp.nominal(Tc), hp.nominal(R_Sc/1000), xerr=hp.stddev(Tc), yerr=hp.stddev(R_Sc/1000), fmt='.', label=r'measured data cooling')
plt.errorbar(hp.nominal(Th), hp.nominal(R_Sh/1000), xerr=hp.stddev(Th), yerr=hp.stddev(R_Sh/1000), fmt='.', label=r'measured data heating')

plt.ylabel(r"$R \, [k\Omega]$")
plt.xlabel(r"$T \, [^{\circ}K]$")
#plt.xlim(580, 760) # intrinsic regime
#plt.ylim(0, 0.015) # intrinsic regime
plt.grid(True)
plt.legend(loc="best")
plt.savefig("plots/temperature_resistance.eps")

# energy gap plot

# cooling

Xc = 1/(2*kB*Tc) * eV # unitless
Xh = 1/(2*kB*Th) * eV # unitless
Yc = copy.copy(R_Sc)
Yh = copy.copy(R_Sh)

# probing the intrinsic regime (values were found through try and error with the residual plot, until the data looked random)
lowerc = np.min(Xc).n
upperc = 10.0
lowerh = 7.6
upperh = 12.5

Tlowerh = 1/(2*kB*upperh) * eV
Tupperh = 1/(2*kB*lowerh) * eV
Tlowerc = 1/(2*kB*upperc) * eV
Tupperc = 1/(2*kB*lowerc) * eV

intrinsic_regime_c = eV/(2*kB*np.array([upperc, lowerc]))
intrinsic_regime_h = eV/(2*kB*np.array([upperh, lowerh]))

X_polyfitc = Xc[np.logical_and(Xc >= lowerc, Xc <= upperc)]
Y_polyfitc = Yc[np.logical_and(Xc >= lowerc, Xc <= upperc)]
X_polyfith = Xh[np.logical_and(Xh >= lowerh, Xh <= upperh)]
Y_polyfith = Yh[np.logical_and(Xh >= lowerh, Xh <= upperh)]

coeffsc = hp.phpolyfit(X_polyfitc, hp.pnumpy.log(Y_polyfitc), 1)
E_gc = coeffsc[0]
A_c = hp.pnumpy.exp(-copy.copy(coeffsc[1]))
coeffsh = hp.phpolyfit(X_polyfith, hp.pnumpy.log(Y_polyfith), 1)
E_gh = coeffsh[0]
A_h = hp.pnumpy.exp(-copy.copy(coeffsh[1]))

pc = lambda x: np.polyval(copy.copy(coeffsc), x)
xc = np.linspace(6, 16, 10)
ph = lambda x: np.polyval(copy.copy(coeffsh), x)
xh = np.linspace(6, 20, 10)

slope_c, intercept_c, r_value_c, p_value_c, std_err_c = linregress(hp.nominal(X_polyfitc), hp.nominal(hp.pnumpy.log(Y_polyfitc)))
slope_h, intercept_h, r_value_h, p_value_h, std_err_h = linregress(hp.nominal(X_polyfith), hp.nominal(hp.pnumpy.log(Y_polyfith)))

def ticks(y, pos):
    return r'$e^{' + r'{:.0f}'.format(np.log(y)) + r'}$'

plt.figure(2)
ax3 = plt.subplot(111)
plt.errorbar(hp.nominal(Xc), hp.nominal(Yc), xerr=hp.stddev(Xc), yerr=hp.stddev(Yc), fmt='.', label=r'measured data')
plt.plot(hp.nominal(xc), hp.nominal(hp.pnumpy.exp(pc(xc))), '-r', label=r'regression line $p_c(x)$')
ax3.axvline(lowerc, linewidth=2, color='black')
ax3.text(lowerc+0.1, 1000, r'lower bound $B_{\downarrow} = ' + hp.fmt_number(lowerc, 2) + r' \, eV$', rotation=90, verticalalignment='center')
ax3.axvline(upperc, linewidth=2, color='black')
ax3.text(upperc+0.1, 1000, r'upper bound $B_{\uparrow} = ' + hp.fmt_number(upperc, 3) + r' \, eV$', rotation=90, verticalalignment='center')
ax3.set_yscale("log", basey=np.e, nonposy='clip')
ax3.yaxis.set_major_formatter(mtick.FuncFormatter(ticks))
plt.xlim(6, 16)
plt.ylabel(r"$\ln{\left( R_S \right)} \, [\Omega]$")
plt.xlabel(r"$\left(2 k_B T \right)^{-1} \, [eV]$")
plt.title("Band gap energy for cooling down the sample")
plt.grid(True)
plt.legend(loc="best")
plt.savefig("plots/energy_gap_cooling.eps")

plt.figure(3)
ax4 = plt.subplot(111)
plt.errorbar(hp.nominal(Xh), hp.nominal(Yh), xerr=hp.stddev(Xh), yerr=hp.stddev(Yh), fmt='.', label=r'measured data')
plt.plot(hp.nominal(xh), hp.nominal(hp.pnumpy.exp(ph(xh))), '-r', label=r'regression line $p_h(x)$')
ax4.axvline(lowerh, linewidth=2, color='black')
ax4.text(lowerh+0.1, 12000, r'lower bound $B_{\downarrow} = ' + hp.fmt_number(lowerh, 2) + r' \, eV$', rotation=90, verticalalignment='center')
ax4.axvline(upperh, linewidth=2, color='black')
ax4.text(upperh+0.1, 12000, r'upper bound $B_{\uparrow} = ' + hp.fmt_number(upperh, 3) + r' \, eV$', rotation=90, verticalalignment='center')
ax4.set_yscale("log", basey=np.e, nonposy='clip')
ax4.yaxis.set_major_formatter(mtick.FuncFormatter(ticks))
#plt.xlim(6, 16)
plt.ylabel(r"$\ln{\left( R_S \right)} \, [\Omega]$")
plt.xlabel(r"$\left(2 k_B T \right)^{-1} \, [eV]$")
plt.grid(True)
plt.title("Band gap energy for heating up the sample")
plt.legend(loc="best")
plt.savefig("plots/energy_gap_heating.eps")

# 3 steps in finding the bounds
data_y_meas_h = [None] * 3
data_y_model_h = [None] * 3
data_x_h = [None] * 3
data_y_meas_c = [None] * 3
data_y_model_c = [None] * 3
data_x_c = [None] * 3

data_y_meas_h[0] = hp.pnumpy.log(Yh)
data_y_meas_c[0] = hp.pnumpy.log(Yc)
data_y_model_h[0] = ph(Xh)
data_y_model_c[0] = pc(Xc)
data_x_h[0] = Xh
data_x_c[0] = Xc

X_2c = Xc[np.logical_and(Xc >= 6.0, Xc <= 14.0)]
Y_2c = Yc[np.logical_and(Xc >= 6.0, Xc <= 14.0)]
X_2h = Xh[np.logical_and(Xh >= 6.0, Xh <= 14.0)]
Y_2h = Yh[np.logical_and(Xh >= 6.0, Xh <= 14.0)]

data_y_meas_h[1] = hp.pnumpy.log(Y_2h)
data_y_meas_c[1] = hp.pnumpy.log(Y_2c)
data_y_model_h[1] = ph(X_2h)
data_y_model_c[1] = pc(X_2c)
data_x_h[1] = X_2h
data_x_c[1] = X_2c

data_y_meas_h[2] = hp.pnumpy.log(Y_polyfith)
data_y_meas_c[2] = hp.pnumpy.log(Y_polyfitc)
data_y_model_h[2] = ph(X_polyfith)
data_y_model_c[2] = pc(X_polyfitc)
data_x_h[2] = X_polyfith
data_x_c[2] = X_polyfitc

fignr = 4
for step in [0, 1, 2]:
	res_h = data_y_meas_h[step] - data_y_model_h[step]
	res_c = data_y_meas_c[step] - data_y_model_c[step]
	sum_h = np.sum(hp.nominal(res_h))
	mean_h = np.mean(hp.nominal(res_h))
	sum_c = np.sum(hp.nominal(res_c))
	mean_c = np.mean(hp.nominal(res_c))

	plt.figure(fignr)
	plt.plot(hp.nominal(data_x_c[step]), hp.nominal(res_c), '.', label=r'residuals step ' + str(step+1) + ' (cooling)')
	plt.plot(hp.nominal(data_x_h[step]), hp.nominal(res_h), '.', label=r'residuals step ' + str(step+1) + ' (heating)')
	plt.ylabel(r"Residuals $e_{"+str(step+1)+"} = y_{observed} - y_{predicted}$")
	plt.xlabel(r"$x = \left(2 k_B T \right)^{-1}$")
	plt.grid(True)
	plt.title("Step " + str(step + 1))
	plt.legend(loc="best")
	plt.savefig("plots/residuals_step" + str(step+1) + ".eps")
	fignr = fignr + 1

	hp.replace("res:sum:h:step"+str(step+1), sum_h)
	hp.replace("res:sum:c:step"+str(step+1), sum_c)
	hp.replace("res:mean:h:step"+str(step+1), mean_h)
	hp.replace("res:mean:c:step"+str(step+1), mean_c)

hp.replace("E_g_c", E_gc)
hp.replace("B_lower_c", hp.fmt_number(lowerc, 2))
hp.replace("B_upper_c", hp.fmt_number(upperc, 3))
hp.replace("T_lower_c", hp.fmt_number(Tlowerc, 3))
hp.replace("T_upper_c", hp.fmt_number(Tupperc, 3))
hp.replace("linear_fit_c", hp.fmt_fit(coeffsc))

hp.replace("E_g_h", E_gh)
hp.replace("B_lower_h", hp.fmt_number(lowerh, 2))
hp.replace("B_upper_h", hp.fmt_number(upperh, 3))
hp.replace("T_lower_h", hp.fmt_number(Tlowerh, 3))
hp.replace("T_upper_h", hp.fmt_number(Tupperh, 3))
hp.replace("linear_fit_h", hp.fmt_fit(coeffsh))

Tc_int_linspace = np.linspace(intrinsic_regime_c[0], intrinsic_regime_c[1], 500)
Th_int_linspace = np.linspace(intrinsic_regime_h[0], intrinsic_regime_h[1], 500)
Tc_int = Tc[np.logical_and(Xc >= lowerc, Xc <= upperc)]
Th_int = Th[np.logical_and(Xh >= lowerh, Xh <= upperh)]
R_Sc_int = R_Sc[np.logical_and(Xc >= lowerc, Xc <= upperc)]
R_Sh_int = R_Sh[np.logical_and(Xh >= lowerh, Xh <= upperh)]

#print(E_gh*eV)
#print(kB*T)
#print(E_gh, E_gc)
n_c = lambda T: T**(3/2)*hp.pnumpy.exp(-copy.copy(E_gc)*eV/(2*kB*T))
n_h = lambda T: T**(3/2)*hp.pnumpy.exp(-copy.copy(E_gh)*eV/(2*kB*T))
mu_c = 1/(R_Sc_int*eV*n_c(Tc_int))
mu_h = 1/(R_Sh_int*eV*n_h(Th_int))
#print(mu)

#print(intrinsic_regime_h)
#print(Th[19:])

#print(mu_c)

coeff_c = np.polyfit(hp.nominal(Tc_int**(-3/2)), hp.nominal(mu_c), 1)
coeff_h = np.polyfit(hp.nominal(Th_int**(-3/2)), hp.nominal(mu_h), 1)
c1_c, c2_c = copy.copy(coeff_c[0]), copy.copy(coeff_c[1])
c1_h, c2_h = copy.copy(coeff_h[0]), copy.copy(coeff_h[1])
mu_fit_c = lambda x: c1_c*x**(-3/2) + c2_c
mu_fit_h = lambda x: c1_h*x**(-3/2) + c2_h
Ts_c = np.linspace(450, 900, 500)
Ts_h = np.linspace(450, 900, 500)

#print(c1, c2, coeff)

#print(Ts)
#print(mu_fit(Ts**(-3/2)))

plt.figure(7)
fig, ax = plt.subplots(1)
plt.errorbar(hp.nominal(Tc_int), hp.nominal(mu_c), xerr=hp.stddev(Tc_int), yerr=hp.stddev(mu_c), fmt='.', label=r'cooling (intrinsic regime)')
plt.errorbar(hp.nominal(Th_int), hp.nominal(mu_h), xerr=hp.stddev(Th_int), yerr=hp.stddev(mu_h), fmt='.', label=r'heating (intrinsic regime)')
plt.plot(Ts_c, mu_fit_c(Ts_c), 'r-', label=r'theory $\propto T^{-3/2}$')
plt.plot(Ts_h, mu_fit_h(Ts_h), 'r-')

plt.xlabel(r"Temperature $T$ $[^{\circ}K]$")
plt.ylabel(r"Mobility $\mu$ $[a.u.]$")
plt.xlim(450, 900)
plt.ylim(1.0*10**18, 2.6*10**18)
ax.set_yticklabels([])
plt.grid(True)
plt.legend(loc="best")
plt.savefig("plots/mobility_vs_temp.eps")

#exit()

plt.figure(8)
fig, ax = plt.subplots(1)
plt.plot(1/Tc_int_linspace, hp.nominal(n_c(Tc_int_linspace)), '.', label=r'cooling (intrinsic regime)')
plt.plot(1/Th_int_linspace, hp.nominal(n_h(Th_int_linspace)), '.', label=r'heating (intrinsic regime)')

plt.xlabel(r"$1/T$ $[^{\circ}K^{-1}]$")
plt.ylabel(r"Carrier Concentration $n_i$ $[a.u.]$")
ax.set_yticklabels([])
plt.grid(True)
plt.legend(loc="best")
plt.savefig("plots/n_vs_temp_inverse.eps")
#plt.show()

arrc = hp.to_table(
	r'$T \, [^{\circ}C]$', Tc - 273.15,
	r'$I \, [mA]$', Ic*10**3,
	r'$V_4 \, [mV]$', V_4c*10**3,
	r'$R_S \, [\Omega]$', R_Sc
)

arrh = hp.to_table(
	r'$T \, [^{\circ}C]$', Th - 273.15,
	r'$I \, [mA]$', Ih*10**3,
	r'$V_4 \, [mV]$', V_4h*10**3,
	r'$R_S \, [\Omega]$', R_Sh
)

hp.replace("table:data:cooling", arrc)
hp.replace("table:data:heating", arrh)

hp.compile()