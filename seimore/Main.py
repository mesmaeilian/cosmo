### Eigen-Reconstruction of Scaler Perturbations
### Script Created on November 27th, 2019

### Libraries
from scipy.integrate import quad as qd
import time as t
from BasicFuncs import *

### Main Part of Code
def pwr(number, root):
	return np.log(number)/np.log(root)

### Smooth Function
def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise (ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise (ValueError, "Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise (ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=numpy.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='valid')
    return y


class base_func():
	def __init__(self, k_min, k_max, N, npoints, modified=False):
		self.k_min = k_min
		self.k_max = k_max
		self.N = N
		self.npoints = npoints
		self.mdfflg = modified
		self.k = np.log(np.logspace(pwr(self.k_min,10), pwr(self.k_max, 10), self.npoints, endpoint=True))
		self.ki = np.log(np.logspace(pwr(self.k_min,10), pwr(self.k_max, 10), self.N, endpoint=True))
		if self.mdfflg == True:
			self.ki = np.log(np.logspace(pwr(self.k_min,10), pwr(self.k_max, 10), int(self.N+self.N/10), endpoint=True))[int(self.N/20):-int(self.N/20)]
		# self.dlt_k = (np.log(self.k_max)-np.log(self.k_min))/self.npoints
		self.sig = (self.ki[1] - self.ki[0])/2
		self.flg = False
		self.chsn_bf = ''
		self._dict_basefuncs = {
			"G": 'basefunc_1',
			"T": 'basefunc_2',
			"H": 'basefunc_3_b',
			"F": 'basefunc_4',
			"C": 'basefunc_5'
			}
		self.eps = self.sig
		self.norm_flg = [False for i in range(5)]

	def basefunc_1(self, k, i):
		print("<<Gaussian>> Base Function is being Used !!!")
		return np.exp( -((k-self.ki[i])**2) / (2*(self.sig**2)))

	def basefunc_2(self, k, i):
		print("<<Triangular>> Base Function is being Used !!!")
		ovl_param = 1.5
		sig = self.sig
		o_index = np.where(abs(k-self.ki[i]) < ovl_param*(sig))
		res = np.zeros(len(k))
		res[o_index] = 1 - (abs(k[o_index]-self.ki[i])/(ovl_param*sig))
		return res

	def basefunc_3(self, k, i):
		print("<<Top Hat>> Base Function is being Used !!!")
		sig = self.sig
		n_sig = (len(k)/(self.N))
		n_eps = int(n_sig*0.2)
		if n_eps < 1 :
			n_eps = 1
		o_index = np.where(np.abs(k-self.ki[i]) < sig)
		res = np.zeros(len(k))
		res[o_index[0]] = 1
		res[o_index[0][:n_eps]-n_eps] = np.linspace(0, 1, len(o_index[0][:n_eps]-n_eps))
		res[o_index[0][-n_eps:]+n_eps] = np.linspace(0, 1, len(o_index[0][:n_eps]-n_eps))[::-1]
		return res

	def basefunc_3_b(self, k, i):
		print("<<Top Hat>> Base Function is being Used !!!")
		sig = self.sig
		o_index = np.where(np.abs(k-self.ki[i]) < sig)
		res = np.zeros(len(k))
		res[o_index[0]] = 1
		return res

	def basefunc_4(self, k, i):
		print("<<Fourier>> Base Function is being Used !!!")
		k_min = np.log(self.k_min)
		k_max = np.log(self.k_max)
		k_mid = (k_min + k_max) / 2
		var = 2 * (i+1) * np.pi * (k - k_mid)/(k_max-k_min)
		if i >= self.N/2 :
			var = np.pi/2 - (i+1-self.N/2) * 2 * np.pi * (k - k_mid)/(k_max-k_min)			
		return np.sin(var)

	def basefunc_5(self, k, i):
		print("<<Chebyshev>> Base Function is being Used !!!")
		k = ( ( (k-k[0])/(k[-1]-k[0]) ) * 2 ) - 1
		if i == 0:
			return 1
		elif i == 1:
			return k
		else:
			cheb = [1, k]
			print(i,">>>>>>>>")
			for j in range(i-1):
				cheb.append(2*k*cheb[-1]-cheb[-2])
		return cheb[-1]

	def normalzr(self, typ):
		integ = []
		for i in [int(self.N/2)]:
			f = interp1d(self.k, self.bsfnc(self.k, i, name=typ))
			fl = lambda k : f(k)**2
			mid_p = self.ki[int(self.N/2)]
			i_v = qd(fl, np.log(self.k_min), np.log(self.k_max), points=np.linspace(mid_p-self.sig, mid_p+self.sig, 100), limit=500)[0]
			integ.append(i_v)
		self.norm_flg[list(self._dict_basefuncs.keys()).index(typ)] = sum(integ)/len(integ)
		# self.norm_flg[list(self._dict_basefuncs.keys()).index(typ)] = 1

	def bsfnc(self, k, i, name):
		self.chsn_bf = name
		return getattr(self, self._dict_basefuncs[name])(k, i)

	def bsf(self, k, i, name=None):
		if name == None:
			if self.flg == False:
				bf_names = self._dict_basefuncs.keys()
				bf_names = [str(i) for i in bf_names]
				chsn_bf = str(input("Choose Type of Base Function to Start :\n %s \n"%str(bf_names)))
				self.chsn_bf = name
				self.flg = True
				return self.bsfnc(k, i, chsn_bf)
			else:
				return self.bsfnc(k, i, self.chsn_bf)
		else:
			if self.norm_flg[list(self._dict_basefuncs.keys()).index(name)] == False:
				self.normalzr(name)
			return np.sqrt(1/self.norm_flg[list(self._dict_basefuncs.keys()).index(name)]) * self.bsfnc(k, i, name)

class PS():
	def __init__(self, k_min, k_max, N, npoints, survphas, wnd):
		self.k_min = k_min
		self.k_max = k_max
		self.npoints = npoints
		self.N = N
		self.wnd =  wnd
		self.kmaxeff = k_max
		self.base_func = base_func(k_min=self.k_min, k_max=self.k_max, N=self.N, npoints=self.npoints)
		self.bsi = Params(survphas[:-1], int(survphas[-1]))
		self.l = []
		self.knlcutcoff = []
		self.Ws = []
		if wnd == 1:
			self.nden_dct = self.bsi.prms.dict_of_numb_den
		if wnd == 3:
			self.knlcutcoff = np.array([(1+i)**(2/(2+self.bsi.prms.mfs.ns)) for i in self.bsi.prms.shzs])
			# self.Ws = np.array([self.bsi.WVs(i, self.bsi.prms.shzs) for i in range(len(self.bsi.prms.dict_of_bin_indx)-1)])
			self.As = np.array([self.bsi.A(i) for i in range(len(self.bsi.prms.dict_of_bin_indx)-1)])
			self.Ws = np.array([self.bsi.WVs(i, self.bsi.prms.shzs, self.As[i]) for i in range(len(self.bsi.prms.dict_of_bin_indx)-1)])
			self.kmaxeff = 1
		if wnd == 4:
			self.s4fs, self.NlTT = self.bsi.prms.N_l(XX="TT", channels=["MF1", "MF2"], fsindx=1)
			self.s4fs, self.NlEE = self.bsi.prms.N_l(XX="EE", channels=["MF1", "MF2"], fsindx=1)
			# print(self.s4fs)

	def init_params(self):
		pars = camb.CAMBparams()
		pars.set_cosmology(H0=self.bsi.prms.mfs.H0, ombh2=self.bsi.prms.mfs.ombh2, omch2=self.bsi.prms.mfs.omch2)
		pars.InitPower.set_params(As = self.bsi.prms.mfs.As, ns = self.bsi.prms.mfs.ns)
		return pars

	def set_pf(self, i, alpha, base, nonlinear, f=None):
		pars = self.init_params()
		if f == None:
			PK = lambda k, i, alpha : self.bsi.prms.mfs.As * ((k/self.bsi.prms.mfs.k0)**(self.bsi.prms.mfs.ns-1)) * ( 1 + (alpha*self.base_func.bsf(np.log(k), i, base)) )
		else:
			PK = lambda k, i, alpha : self.bsi.prms.mfs.As * ((k/self.bsi.prms.mfs.k0)**(self.bsi.prms.mfs.ns-1)) * ( 1 + (alpha*f(np.log(k))))
		efns = None
		if nonlinear == True:
			efns = self.bsi.prms.mfs.ns
		pars.set_initial_power_function(PK, effective_ns_for_nonlinear=efns, N_min=600, args=(i, alpha));
		return pars

	def set_prmps(self, i, alpha, base, nonlinear, f):
		pars = self.set_pf(i, alpha, base, nonlinear, f)
		if nonlinear == True: pars.NonLinear = model.NonLinear_both
		else:
			pars.NonLinear = model.NonLinear_none
		return pars

	def pws(self, z, i, alpha, base, nonlinear, f=None):
		pars = self.set_prmps(i, alpha, base, nonlinear, f)
		pars.set_matter_power(redshifts=z)
		results = camb.get_results(pars)
		kh, zs, pk = results.get_matter_power_spectrum(minkh=self.k_min/self.bsi.prms.mfs.h, maxkh=self.kmaxeff/self.bsi.prms.mfs.h, npoints=self.npoints)
		return kh, zs, pk/(self.bsi.prms.mfs.h**3)

	def cls(self, i, alpha, base, nonlinear, f=None, clmode='total', nonlinlens=False, lmin=5, lmax=5000):
		pars = self.set_prmps(i, alpha, base, nonlinear, f)
		pars.set_nonlinear_lensing(nonlinear=nonlinlens)
		pars.set_for_lmax(lmax, lens_potential_accuracy=0)
		results = camb.get_results(pars)
		powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True)[clmode];
		Cl = np.array([[powers[:, 0][lmin:lmin+lmax+1]+self.NlTT, powers[:, 3][lmin:lmin+lmax+1]], [powers[:, 3][lmin:lmin+lmax+1], powers[:, 1][lmin:lmin+lmax+1]+self.NlEE]])
		return np.arange(lmin, lmin+lmax+1), Cl

	def mtr_pws_o(self, kh, zs, pk, z, mu):
		pk_o = [(self.bsi.prms.b(z)**2) * ((1+self.bsi.betta(z)*mu**2)**2) * np.exp(-(j**2)*(mu**2)*((self.bsi.prms.sigma_r(z)**2)+(self.bsi.prms.sigma_v(z)**2))) * pk[i] for i,j in enumerate(kh)]
		return np.array(kh), np.array(pk_o)

	def obs21(self):
		return 1

	def sh_ps(self, kh, zs, pk, i, j):
		l = np.arange(5, 3505)
		self.l = l
		intgrd = np.zeros(len(l))
		for a, z in enumerate(zs):
			lloc = l.copy()
			pkz = interp1d(kh, pk[a], kind="cubic")
			lind1, lind2 = self.k_min/self.bsi.prms.mfs.h < l/(self.bsi.prms.mfs.x(z)), l/(self.bsi.prms.mfs.x(z)) < 0.15*self.knlcutcoff[a]/self.bsi.prms.mfs.h
			lind = lind1 * lind2
			lloc[lind==False] = 5
			pl = pkz(lloc/(self.bsi.prms.mfs.x(z)))
			s = self.Ws[i][a] * self.Ws[j][a] * (1/(1+z**4)) * (self.bsi.prms.mfs.H(z)**3) * (self.bsi.prms.mfs.Omega_m(z)**2) * pl * lind
			intgrd += s

		P = intgrd * 0.05 * (9/4) * (1/self.bsi.prms.mfs.c**3)
		C = P + ((self.bsi.prms.gammav**2)*(self.bsi.prms.nhi**-1)*self.bsi.prms.mfs.delta(i,j))
		return l, P, C

class fish_mtr():
	def __init__(self, k_min, k_max, N, npoints, survphas, wnd):
		self.k_min = k_min
		self.k_max = k_max
		self.npoints = npoints
		self.N = N
		self.epsilon = 0.01
		self.Cls = eval("PS")(self.k_min, self.k_max, self.N, self.npoints, survphas, wnd)
		self.blc = '-'*20
		self.ptr1 = '>'*1
		self.ptr2 = '<'*1
		self.times = []
		self.kh = []
		self.zs = []
		self.pkf = []
		self.clf = []
		self.ls = []
		self.shpsf = []
		self.pkt = []
		self.pkot = []
		self.shps = []
		self.covshps = []
		self.Cl = []
		self.gtu = []

	def Pkf(self, zs, nonlinear):
		self.kh, self.zs, self.pkf = self.Cls.pws(zs, 0, 0, "F", nonlinear)

	def CLf(self, nonlinear):
		self.ls, self.clf = self.Cls.cls(0, 0, "F", nonlinear)

	def Pk_t(self, zs, a, base, nonlinear):
		print(self.blc, "CAMB CALCS STARTED", self.blc)
		self.pkt = []
		for i in range(self.N):
			print(self.blc, "PK{}".format(str(i)), self.blc)
			a1 = t.time()
			pk = self.Cls.pws(zs, i, a+self.epsilon, base, nonlinear)[2]
			self.pkt.append(pk)
			a2 = t.time()
			print(self.blc, "TIME : ", a2-a1, self.blc)
		print(self.blc, "CAMB CALCS FINISHED", self.blc)

	def PkO_t(self):
		print(self.blc, "OBS CALCS STARTED", self.blc)
		self.pkot = []
		pko_matr = np.zeros((len(self.Cls.bsi.prms.tz), len(self.Cls.bsi.prms.tmu), len(self.kh)))
		print(len(self.Cls.bsi.prms.tz))
		for j, z in enumerate(self.Cls.bsi.prms.tz):
			for p, mu in enumerate(self.Cls.bsi.prms.tmu):
				print(self.blc, "PKO(z={},mu={})".format(str(round(z,1)), str(round(mu, 1))), self.blc)
				a3 = t.time()
				pko_matr[j][p] = self.Cls.mtr_pws_o(self.kh, self.zs, self.pkf[j], z, mu)[1]
				a4 = t.time()
				print(self.blc, "TIME : ", a4-a3, self.blc)
		self.pkot = pko_matr
		print(self.blc, "OBS CALCS FINISHED", self.blc)

	def ShPf(self):
		P_t = np.zeros((len(self.Cls.bsi.prms.tz), len(self.Cls.bsi.prms.tz), 3500))
		print(len(self.Cls.bsi.prms.tz))
		print(len(self.Cls.Ws))
		for i in range(len(self.Cls.bsi.prms.tz)):
			for j in range(i, len(self.Cls.bsi.prms.tz)):
				print("CALCS FOR FIDUCIAL PS AT ij = {}{}".format(str(i+1), str(j+1)))
				P_t[i][j] = self.Cls.sh_ps(self.kh, self.zs, self.pkf, i, j)[1]
				P_t[j][i] = P_t[i][j]
		self.shpsf = P_t

	def ShP_t(self):
		print(self.blc, "SH PS CALCS STARTED", self.blc)
		self.shps = []
		self.covshps = []
		for n, pk in enumerate(self.pkt):
			print(self.blc, "CALCS FOR BASE NO. : %s"%str(n+1), self.blc)
			P_t = np.zeros((len(self.Cls.bsi.prms.tz), len(self.Cls.bsi.prms.tz), 3500))
			C_t = np.zeros((len(self.Cls.bsi.prms.tz), len(self.Cls.bsi.prms.tz), 3500))
			for i in range(len(self.Cls.bsi.prms.tz)):
				for j in range(i, len(self.Cls.bsi.prms.tz)):
					P_t[i][j], C_t[i][j] = self.Cls.sh_ps(self.kh, self.zs, pk, i, j)[1:]
					P_t[j][i], C_t[j][i] = P_t[i][j], C_t[i][j]
			self.shps.append(P_t)
			self.covshps.append(C_t)
		print(self.blc, "SH PS CALCS FINISHED", self.blc)

	def Cl_t(self, a, base, nonlinear):
		self.Cl = []
		for i in range(self.N):
			print("CL {}".format(str(i)))
			ls, cl = self.Cls.cls(i, a+self.epsilon, base, nonlinear)
			self.Cl.append(cl)

	def pger(self, gamma):
		err = np.array([np.exp(-(gamma)*((i-0.1)/0.1)**2) for i in self.kh])
		indx = np.where(self.kh<0.1)
		err[indx] = 1
		self.gtu = err
		return err

	def fishel_mps(self, i, j):
		F_time = t.time()
		t_lst = []
		for x, z in enumerate(self.Cls.bsi.prms.tz):
			nz = self.Cls.bsi.n_gc(z)
			lst = [] 
			for b, mu in enumerate(self.Cls.bsi.prms.tmu):
				fmp = np.array(self.pkf[x])
				m2 = np.array(self.pkt[i][x])
				m4 = np.array(self.pkt[j][x])
				omp = np.array(self.pkot[x][b])
				pd1 = (m2 - fmp)/self.epsilon
				pd2 = (m4 - fmp)/self.epsilon
				# print("karim")
				intgrtd_func = [(j**3) * pd1[i] * pd2[i] * ((1/fmp[i])**2) * self.gtu[i] * ((nz*omp[i])/(nz*omp[i]+1))**2 for i, j in enumerate(self.kh)]
				f_integ = sum(intgrtd_func)*(np.log(self.kh[1])-np.log(self.kh[0]))
				lst.append(f_integ)
			t_lst.append(self.Cls.bsi.vr_z(z)*sum(lst))
		t_integ = sum(t_lst)
		Se_time = t.time()
		self.times.append(Se_time - F_time)
		print(57*self.ptr1, " Total Calc time : ", round(Se_time-F_time, 2) , 57*self.ptr2)
		rm = (((self.N*self.N/2)+self.N/2 - len(self.times)) * sum(self.times)/len(self.times))/60
		minute, sec = int(rm), int((rm-int(rm))*60)
		print(46*self.ptr1, "Remaining Time : {} Minute(s) and {} Second(s)".format(minute, sec), 46*self.ptr2)
		print(7*self.blc)
		return (1/(8*np.pi)**2)*t_integ


	# def fishel_mp(self, i, j):
	# 	F_time = t.time()
	# 	t_lst = []
	# 	for x, z in enumerate(self.Cls.bsi.prms.tz):
	# 		m2 = np.array(self.pkt[i][x])
	# 		m4 = np.array(self.pkt[j][x])
	# 		fmp = np.array(self.pkf[x])
	# 		pd1 = (m2 - fmp)/self.epsilon
	# 		pd2 = (m4 - fmp)/self.epsilon
	# 		intgrtd_func = [(j**3) * pd1[i] * pd2[i] for i, j in enumerate(self.kh)]
	# 		f_integ = sum(intgrtd_func)*(np.log(self.kh[1])-np.log(self.kh[0]))
	# 		t_lst.append(f_integ)
	# 	t_integ = sum(t_lst)
	# 	Se_time = t.time()

	# 	self.times.append(Se_time - F_time)
	# 	print(57*self.ptr1, " Total Calc time : ", round(Se_time-F_time, 2) , 57*self.ptr2)
	# 	rm = (((self.N*self.N/2)+self.N/2 - len(self.times)) * sum(self.times)/len(self.times))/60
	# 	minute, sec = int(rm), int((rm-int(rm))*60)
	# 	print(46*self.ptr1, "Remaining Time : {} Minute(s) and {} Second(s)".format(minute, sec), 46*self.ptr2)
	# 	print(7*self.blc)
	# 	return t_integ

	def fishel_shps(self, i, j):
		P1, C1 = np.array(self.shps[i]), np.array(self.covshps[i])
		P2, C2 = np.array(self.shps[j]), np.array(self.covshps[j])
		inr_sumtn = []
		for i in range(3500):
			pd1 = P1[:, :, i] - self.shpsf[:, :, i]
			pd2 = P2[:, :, i] - self.shpsf[:, :, i]
			C1inv = np.linalg.inv(C1[:, :, i])
			C2inv = np.linalg.inv(C2[:, :, i])
			s = ((2*i+1)/2) * np.trace(pd1 @ C1inv @ pd2 @ C2inv)
			inr_sumtn.append(s)
		return self.Cls.bsi.prms.f_sky * sum(inr_sumtn)

	def fishel_cl(self, i, j):
		CLN1 = np.array(self.Cl[i])
		CLN2 = np.array(self.Cl[j])
		inr_sumtn = []
		for i, j in enumerate(self.ls):
			pd1 = (CLN1[:,:,i] - self.clf[:,:,i])/self.epsilon
			pd2 = (CLN2[:,:,i] - self.clf[:,:,i])/self.epsilon
			Cinv = np.linalg.inv(self.clf[:,:,i])
			s = ((2*j+1)/2) * np.trace(pd1 @ Cinv @ pd2 @ Cinv)
			inr_sumtn.append(s)
		inr_sumtn = np.array(inr_sumtn)
		inr_sumtn[np.isnan(inr_sumtn)] = 0
		return self.Cls.s4fs * np.sum(inr_sumtn)

	def fish_matr(self, base, nonlinear, a, T):
		fm = np.zeros((self.N,self.N))
		if T == 1:
			self.Pkf(self.Cls.bsi.prms.tz, nonlinear)
			self.Pk_t(self.Cls.bsi.prms.tz, a, base, nonlinear)
			for i in range(len(fm)):
				for j in range(i, len(fm)):
					print(7*self.blc)
					print(60*self.ptr1, "Calculating F{%s,%s}"%(str(i+1), str(j+1)), 60*self.ptr2)
					fm[i][j] = self.fishel_mp(i, j)
					fm[j][i] = fm[i][j]
			return fm
		elif T == 2:
			self.Pkf(self.Cls.bsi.prms.tz, nonlinear)
			self.Pk_t(self.Cls.bsi.prms.tz, a, base, nonlinear)
			self.PkO_t()
			_ = self.pger(2)
			for i in range(len(fm)):
				for j in range(i, len(fm)):
					print(7*self.blc)
					print(60*self.ptr1, "Calculating F{%s,%s}"%(str(i+1), str(j+1)), 60*self.ptr2)
					fm[i][j] = self.fishel_mps(i, j)
					fm[j][i] = fm[i][j]
			return fm
		elif T == 3:
			self.Pkf(self.Cls.bsi.prms.shzs, nonlinear)
			self.Pk_t(self.Cls.bsi.prms.shzs, a, base, nonlinear)
			self.ShPf()
			self.ShP_t()
			for i in range(len(fm)):
				for j in range(i, len(fm)):
					print(7*self.blc)
					print(60*self.ptr1, "Calculating F{%s,%s}"%(str(i+1), str(j+1)), 60*self.ptr2)
					fm[i][j] = self.fishel_shps(i, j)
					fm[j][i] = fm[i][j]
			return fm
		elif T == 4:
			self.CLf(nonlinear)
			self.Cl_t(a, base, nonlinear)
			for i in range(len(fm)):
				for j in range(i, len(fm)):
					print(7*self.blc)
					print(60*self.ptr1, "Calculating F{%s,%s}"%(str(i+1), str(j+1)), 60*self.ptr2)
					fishc = self.fishel_cl(i, j)
					fm[i][j] = fishc
					fm[j][i] = fm[i][j]
			return fm
		else:
			print("Something Went Wrong!")
		return fm
