from Main import base_func, fish_mtr, pwr, interp1d
import numpy as np
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class Mod_Cnstr():
	def __init__(self, k_min, k_max, n_base, npoints, nonlinear, survphas, mode, n_typ, l_typ):
		self.mode = mode
		self.k_min = k_min
		self.k_max = k_max
		self.n_base = n_base
		self.n_typ = n_typ
		self.l_typ = list(l_typ)
		self.nonlin_flg = nonlinear
		self.npoints = npoints
		self.survphas = survphas
		self.base_func = base_func(k_min=self.k_min, k_max=self.k_max, N=self.n_base, npoints=self.npoints)
		self.fish_class = fish_mtr(k_min=self.k_min, k_max=self.k_max, N=self.n_base, npoints=self.npoints, survphas=self.survphas, wnd=self.mode)
		self.tblc = '~'*140 + '\n' + '~'*140
		self.Gtyp_dct = {}
		self.Etyp_dct = {}
		self.Mtyp_dct = {}
		self.fishmtr_dct = {}
		self.egn_dct = {}
		self.mds_dct = {}
		self.mf = {}
		self.mpresp = {}
		self.clresp = {}
		self.apresp = {}
		self.dicts = dict((key, {}) for key in ["Info","FishMatr", "EigenValVec", "Modes", "MF"])
		self.mode_dct = {"1":"MatterPS","2":"GalaxyClustering", "3":"WeakLensing", "4":"CMBTemperature"}
		self.nlft = {"True":"NL", "False":"L"}
		self.resp = {"KH":[], "LS":[] ,"MP":{}, "CT":{}, "AP": {}}

	def ChsnMs(self):
		if self.n_typ == "W":
			self.l_typ = list("GTHFC")
		self.Gtyp_dct = dict((key, "self.%s_fish"%key) for key in self.l_typ)
		self.Etyp_dct = dict((key, "self.%s_egns"%key) for key in self.l_typ)
		self.Mtyp_dct = dict((key, "self.%s_mds"%key) for key in self.l_typ)
		self.fishmtr_dct = dict((key, []) for key in self.l_typ)
		self.egn_dct = dict((key, []) for key in self.l_typ)
		self.egn_mds = dict((key, []) for key in self.l_typ)
		self.mf = dict((key, []) for key in self.l_typ)
		self.mpresp = dict((key, []) for key in self.l_typ)
		self.clresp = dict((key, []) for key in self.l_typ)
		self.apresp = dict((key, 0) for key in self.l_typ)
		return self.l_typ

	# def Resps(self, alpha=0.1, base="G", nonlinear=False, f=None):
	# 	regs = self.n_base
	# 	if f : regs = 1
	# 	mpr = []
	# 	clr = []
	# 	for i in range(regs):
	# 		_, pkc, _, clc = self.fish_class.Cls.pws(self.fish_class.Cls.tz, i, alpha, base, nonlinear=nonlinear, f=f)
	# 		mpr.append((pkc-self.pkf)/alpha)
	# 		clr.append((clc-self.clf)/alpha)
	# 		print("Calc For Bin {} are Done ! \n{}".format(str(i+1), 50*'-'))
	# 	if len(mpr) == 1 :
	# 		mpr = mpr[0]
	# 		clr = clr[0]
	# 	self.mpresp[base] = np.array(mpr)
	# 	self.clresp[base] = np.array(clr)
	# 	self.apresp[base] = alpha
	# 	self.resp["MP"] = self.mpresp
	# 	self.resp["CT"] = self.clresp
	# 	self.resp["KH"] = self.kh
	# 	self.resp["LS"] = self.ls
	# 	self.resp["AP"] = self.apresp

	def evl_fish(self):
		if self.n_typ == "W" or self.n_typ == "D": 
			for i in self.ChsnMs():
				print(self.tblc)
				self.fishmtr_dct[i] = (self.fish_class.fish_matr(i, nonlinear=self.nonlin_flg, a=0, T=self.mode))
				print(self.tblc)
				self.fish_class.times = []
		else:
			try:
				self.n_typ = input("W Mode : Execute Code for all of Base Functions\nD Mode : Execute Code for Chosen Base Functions\nPlease Choose between W or D Mode : ")
				if self.n_typ == 'D':
					self.l_typ = input("Please make a string from base function names something like 'GT'. Types : ").upper()
				return self.evl_fish()
			except ValueError:
				print("You Enter Something Wrong!")

	def evl_egn(self):
		self.fishmtr_dct = dict((key, np.array(self.fishmtr_dct[key])) for i, key in enumerate(self.fishmtr_dct.keys()))
		for i in self.Gtyp_dct.keys():
			try:
				self.egn_dct[i] = np.linalg.eigh(self.fishmtr_dct[i])
			except ValueError:
				print("Process of Calc Eigs Failed !!!")

	def Mod2Func(self, mode):
		bmk = np.log(np.logspace(pwr(9e-7,10), pwr(self.k_min, 10), self.npoints/10, endpoint=False))
		amk = np.log(np.logspace(pwr(self.k_max+0.2,10), pwr(2e+2, 10), self.npoints/10, endpoint=True))
		k = np.append(bmk, self.base_func.k)
		k = np.append(k, amk)
		mod = np.append((int(self.npoints/10)*[0]), mode)
		mod = np.append(mod, (int(self.npoints/10)*[0]))
		f = interp1d(k, mod, kind='cubic')
		return f

	def Cstr_M(self):
		self.base_func._dict_basefuncs["H"] = "basefunc_3_b"
		for i in self.egn_dct.keys():
			try:
				modes = []
				# MF = []
				vect = self.egn_dct[i][1].transpose()
				for j in vect:
					md_val = np.zeros(self.npoints)
					for p, q in enumerate(j):
						md_val += q * self.base_func.bsf(self.base_func.k, p, i)
					modes.append(md_val)
					# MF.append(self.Mod2Func(md_val))
				modes.reverse()
				# MF.reverse()
				self.mds_dct[i] = np.array(modes)
				# self.mf[i] = np.array(MF)
			except ValueError:
				print("Process of Calc Modes Failed !!!")

	def CM_Main(self, respi=False, svdes=False):
		if respi != False:
			self.Resps(alpha=respi[0], f=respi[1])
			date = str(datetime.date(datetime.now()))
			nm = "MPCL"
			f_name = "/" + date + '--' + nm + '--' + svdes + ".pickle"
			file = open(f_name,"wb")
			pickle.dump(self.resp, file)
			file.close()
		else:
			self.dicts["Info"] = {"Survey":self.survphas, "Probe":self.mode_dct[str(self.mode)], "KMin":self.k_min, "KMax":self.k_max, "N":self.n_base, "npoints":self.npoints, "Nonlin_flg":self.nonlin_flg, "BaseF":self.l_typ, "KH":self.base_func.k}
			self.evl_fish()
			self.dicts["FishMatr"] = self.fishmtr_dct
			self.evl_egn()
			self.dicts["EigenValVec"] = self.egn_dct
			self.Cstr_M()
			self.dicts["Modes"] = self.mds_dct
			# self.dicts["MF"] = self.mf
			if svdes == None:
				pass
			else:
				if svdes == True:
					date = str(datetime.date(datetime.now()))
					time = str(datetime.time(datetime.now()))[:5]
					nm = "".join(self.l_typ)
					nb = str(self.n_base)
					nl = self.survphas
					nk = "NL%s"%self.nonlin_flg + str(self.k_max)
					f_name = "/" + date + '--' + time + '--' + nm + '-' + nb + '-' + nl + '-' + nk + '-' + self.mode_dct[str(self.mode)] + ".pickle"
				else:
					name = svdes
					f_name = "/" + name + ".pickle"
				file = open(f_name,"wb")
				pickle.dump(self.dicts, file)
				file.close()
