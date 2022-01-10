### Muhammad Sadegh Esmaeilian
### Started October 1st, 2019
### Sub-functions used in other codes

####################

### importing Libraries that we need in this code
import camb
from camb import model, initialpower
from scipy.interpolate import interp1d
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors
from matplotlib import rc
import re
import matplotlib.pyplot as plt
from scipy.integrate import quad as qd
import numpy as np
import time as t
import scipy.special as sp

####################

class Main_fn():
	def __init__(self):
		self.h = 0.6766
		self.H0 = 100*self.h
		self.epsilon = 0.001
		self.c = 2.99*100000
		self.ombh2 = 0.02242
		self.omch2 = 0.11931
		self.ns = 0.9665
		self.k0 = 0.05
		self.As = 2.105 * 1e-9
		self.mnu = 0.06
		self.omk = 0
		self.tau = 0.06
		self.gamma_0 = 0.545
		self.gamma_1 = 0
		self.etha = 0
		self.pars = camb.CAMBparams()
		self.pars.set_cosmology(H0=self.H0, ombh2=self.ombh2, omch2=self.omch2, mnu=self.mnu, omk=self.omk, tau=self.tau)
		self.results = camb.get_results(self.pars)

	def delta(self, i,j):
		if i == j:
			return 1
		else:
			return 0

	def H(self, z):
		return self.results.hubble_parameter(z)

	def D(self, z):
		return self.results.angular_diameter_distance(z)

	def Omega_m(self, z):
		return self.results.get_Omega('baryon', z) + self.results.get_Omega('cdm', z)

	def E(self, z):
		return self.H(z)/self.H0

	def x(self, z):
		return self.results.comoving_radial_distance(z)

	def gamma(self, z):
		return self.gamma_0 + self.gamma_1 * (z / 1+z)

	def f_g(self, z):
		return (self.Omega_m(z)**self.gamma(z)) * (1+self.etha)


class Euclid():
	def __init__(self, phase):
		self.mfs = Main_fn()
		self.sigma_z = 0.001
		self.sigma_v_0 = 300
		self.f_sky = 0.3636
		self.delta_z = 0.1
		self.z_min = 0.65
		self.z_max = 2.05
		self.z_med = 0.9
		self.z_0 = self.z_med/(np.sqrt(2))
		self.gammav = 0.22
		self.n_theta = 30
		self.N_bin = 12
		self.sigzWL = 0.05
		self.npwr = 3/2
		self.hble_unt = self.mfs.h
		self.nhi = 3600*((180/np.pi)**2)*self.n_theta/self.N_bin
		self.tmu = np.arange(-1, 1.01, 0.2)
		self.tz = np.round(np.arange(0.7, 2.01, 0.1), 1)[::-1]
		self.shzs = np.linspace(0.03, 3, 100)[::-1]
		self.dict_of_bin_indx = {
			'0':'0.6',
			'1':'0.7',
			'2':'0.8',
			'3':'0.9',
			'4':'1.0',
			'5':'1.1',
			'6':'1.2',
			'7':'1.3',
			'8':'1.4',
			'9':'1.5',
			'10':'1.6',
			'11':'1.7',
			'12':'1.8',
			'13':'1.9',
			'14':'2.0',
			'15':'2.1'
		}
		self.dict_of_numb_den = {
			'0.7':'1750',
			'0.8':'2680',
			'0.9':'2560',
			'1.0':'2350',
			'1.1':'2120',
			'1.2':'1880',
			'1.3':'1680',
			'1.4':'1400',
			'1.5':'1120',
			'1.6':'810',
			'1.7':'530',
			'1.8':'490',
			'1.9':'290',
			'2.0':'160'
		}

	def b(self, z):
		return np.sqrt(1+z)

	def sigma_r(self, z):
	    return self.sigma_z*(1+z)*self.mfs.c/self.mfs.H(z)

	def sigma_v(self, z):
	    return self.sigma_v_0/self.mfs.H(z)

	def sigzW(self, z):
		return self.sigzWL*(1+z)


class SKA():
	def __init__(self, phase):
		self.mfs = Main_fn()
		self.gamma_0 = 0.545
		self.gamma_1 = 0
		self.etha = 0
		self.sigma_v_0 = 300
		self.delta_z = 0.1
		self.gammav = 0.3
		self.shzs = np.linspace(0.03, 6, 150)[::-1]
		self.tmu = np.arange(-1, 1.01, 0.2)
		self.nu0 = 1420*1e6
		self.npwr = 5/4
		self.hble_unt = 1
		self.binindxdctph1 = {
			'0':'0.05',
			'1':'0.15',
			'2':'0.25',
			'3':'0.35',
			'4':'0.45',
			'5':'0.55'
		}
		self.binindxdctph2 = {
			'0':'0.15',
			'1':'0.25',
			'2':'0.35',
			'3':'0.45',
			'4':'0.55',
			'5':'0.65',
			'6':'0.75',
			'7':'0.85',
			'8':'0.95',
			'9':'1.05',
			'10':'1.15',
			'11':'1.25',
			'12':'1.35',
			'13':'1.45',
			'14':'1.55',
			'15':'1.65',
			'16':'1.75',
			'17':'1.85',
			'18':'1.95',
			'19':'2.05'
		}
		self.numbdenph1 = {
			'0.05':'0.0273',
			'0.15':'0.00493',
			'0.25':'0.000949',
			'0.35':'0.000223',
			'0.45':'0.0000644'
		}
		self.numbdenph2 = {
			'0.15':'0.0620',
			'0.25':'0.0363',
			'0.35':'0.0216',
			'0.45':'0.0131',
			'0.55':'0.00807',
			'0.65':'0.00511',
			'0.75':'0.00327',
			'0.85':'0.00211',
			'0.95':'0.00136',
			'1.05':'0.000870',
			'1.15':'0.000556',
			'1.25':'0.000353',
			'1.35':'0.000222',
			'1.45':'0.000139',
			'1.55':'0.0000855',
			'1.65':'0.0000520',
			'1.75':'0.0000312',
			'1.85':'0.0000183',
			'1.95':'0.0000105'
		}
		self.biasfactph1 = {
			'0.05':'0.657',
			'0.15':'0.714',
			'0.25':'0.789',
			'0.35':'0.876',
			'0.45':'0.966'
		}
		self.biasfactph2 = {
			'0.15':'0.623',
			'0.25':'0.674',
			'0.35':'0.730',
			'0.45':'0.790',
			'0.55':'0.854',
			'0.65':'0.922',
			'0.75':'0.996',
			'0.85':'1.076',
			'0.95':'1.163',
			'1.05':'1.257',
			'1.15':'1.360',
			'1.25':'1.472',
			'1.35':'1.594',
			'1.45':'1.726',
			'1.55':'1.870',
			'1.65':'2.027',
			'1.75':'2.198',
			'1.85':'2.385',
			'1.95':'2.588'
		}
		if phase == 1:
			self.f_sky = 0.1212
			self.z_min = 0
			self.z_max = 0.5
			self.z_med = 1.1
			self.n_theta = 30
			self.N_bin = 2.7
			self.dltnu = 12.7*1e3
			self.sigzWL = 0.05
			self.dict_of_bin_indx = self.binindxdctph1
			self.dict_of_numb_den = self.numbdenph1
			self.dict_of_bias_fact = self.biasfactph1
		elif phase == 2:
			self.f_sky = 0.7272
			self.z_min = 0.1
			self.z_max = 2
			self.z_med = 1.3
			self.n_theta = 30
			self.N_bin = 10
			self.dltnu = 12.8*1e3
			self.sigzWL = 0.03
			self.dict_of_bin_indx = self.binindxdctph2
			self.dict_of_numb_den = self.numbdenph2
			self.dict_of_bias_fact = self.biasfactph2
		self.sig_nu = self.dltnu / np.sqrt(8*np.log(2))
		self.z_0 = self.z_med/(np.sqrt(2))
		self.nhi = 3600*((180/np.pi)**2)*self.n_theta/self.N_bin
		self.tz = np.round(np.arange(self.z_min, self.z_max, 0.1), 1)[::-1]+0.05

	def b(self, z):
		return float(self.dict_of_bias_fact[str(round(z, 2))])

	def sigma_r(self, z):
	    return ((1+z)**2)*(self.sig_nu/self.nu0)*self.mfs.c/self.mfs.H(z)

	def sigma_v(self, z):
	    return self.sigma_v_0/self.mfs.H(z)

	def sigzW(self, z):
		return self.sigzWL*(1+z)


class S4():
	def __init__(self, phase=1):
		self.mfs = Main_fn()
		self.freqchnls = {
		"LF1":"21",
		"LF2":"29",
		"LF3":"40",
		"MF1":"95",
		"MF2":"150",
		"HF1":"220",
		"HF2":"270"
		}
		self.conf1 = {
		"LF1":[8.4, 10.4, 7.3, 23.1],
		"LF2":[6.1, 7.5, 5.3, 16.7],
		"LF3":[6.1, 5.5, 5.3, 16.8],
		"MF1":[1.5, 2.3, 1.3, 4.1],
		"MF2":[1.7, 1.5, 1.5, 4.6],
		"HF1":[6.5, 1.0, 5.7, 17.9],
		"HF2":[10.8, 0.8, 9.4, 29.7]
		}
		self.conf2 = {
		"LF1":[9.2, 125.0, 8.0, 25.2],
		"LF2":[6.4, 90.5, 5.5, 17.5],
		"LF3":[6.7, 65.6, 5.8, 18.3],
		"MF1":[1.6, 27.6, 1.4, 4.4],
		"MF2":[1.8, 17.5, 1.5, 4.8],
		"HF1":[6.8, 11.9, 5.9, 18.7],
		"HF2":[11.6, 9.7, 10.0, 31.8]
		}
		self.fs = {
		"1":0.05,
		"2":0.50
		}
		self.cvrt = {
		"arcmin-rad": (np.pi/180)*(1/60)
		}
		self.MDX = {
		"TT":1,
		"EE":0.5,
		"BB":0.5
		}
		self.ls = np.arange(5, 5006)
		self.conf = {}

	def N_l(self, XX, channels, fsindx, confindx=1):
		if confindx == 1: self.conf = self.conf1
		else: self.conf = self.conf2
		N_nus = [(self.MDX[XX]*(self.conf[i][fsindx+1]*self.cvrt["arcmin-rad"])**-2)*np.exp(-self.ls*(self.ls+1)*(((self.conf[i][1]*self.cvrt["arcmin-rad"])**2)/(8*np.log(2)))) for i in channels]
		return self.fs[str(fsindx)], 1/sum(N_nus)


class Params():
	def __init__(self, survey, phase):
		self.prms = eval(survey)(phase)

	def bin(self, i):
		return float(self.prms.dict_of_bin_indx[str(int(i))])

	def betta(self, z):
	    return self.prms.mfs.f_g(z) / self.prms.b(z)

	def n_gc(self, z):
		return (self.prms.hble_unt**3) * float(self.prms.dict_of_numb_den[str(z)])

	def n(self, z):
		return (self.prms.npwr/(sp.gamma(3/self.prms.npwr)*self.prms.z_0**3)) * (z**2) * np.exp(-(z/self.prms.z_0)**(self.prms.npwr))

	def ni(self, i, z):
		erv = 1/(np.sqrt(2)*self.prms.sigzW(z))
		return (0.5) * self.n(z) * (sp.erf((self.bin(i+1)-z)*erv) - sp.erf((self.bin(i)-z)*erv))

	# def A(self, z):
	# 	a = 0
	# 	if z < self.prms.z_min or z > self.prms.z_max : return a
	# 	else:
	# 		for i in range(len(self.prms.tz)-1):
	# 			a += self.ni(i, z)
	# 		if a != 0 : return self.n(z)/a
	# 		else:
	# 			return 0

	def A(self, i):
		func = lambda zpr : self.ni(i, zpr)
		integ = qd(func, self.prms.z_min, self.prms.z_max)[0]
		return 1/integ

	# def W(self, i, z):
	# 	func = lambda z_prime : (1 - (self.prms.mfs.x(z)/self.prms.mfs.x(z_prime))) * (self.A(z_prime)*self.ni(i, z_prime))
	# 	integ = qd(func, z, np.inf)[0]
	# 	return integ

	def W(self, i, z, A):
		func = lambda z_prime : (1 - (self.prms.mfs.x(z)/self.prms.mfs.x(z_prime))) * (A*self.ni(i, z_prime))
		integ = qd(func, z, np.inf)[0]
		return integ

	# def WVs(self, i, zlst):
	# 	Ws = []
	# 	for q, j in enumerate(zlst):
	# 		wv = self.W(i, j)
	# 		Ws.append(wv)
	# 	return np.array(Ws)

	def WVs(self, i, zlst, A):
		Ws = []
		for q, j in enumerate(zlst):
			wv = self.W(i, j, A)
			Ws.append(wv)
		return np.array(Ws)

	def vr_z(self, z):
	    return (4*np.pi/3) * (self.prms.f_sky) * ((self.prms.mfs.x(z+(self.prms.delta_z/2))**3) - (self.prms.mfs.x(z-(self.prms.delta_z/2))**3))





