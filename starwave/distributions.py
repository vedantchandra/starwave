import numpy as np
import sys
import os 

path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)
sys.path.append(dir_path)

from generalrandom import GeneralRandom

l_logm = np.log(0.05)
u_logm = np.log(8)

def set_GR_spl(slope):
		x = np.linspace(l_logm,u_logm,1000)
		y = np.exp(x*(slope+1))
		GR_spl = GeneralRandom(x,y,1000)
		return GR_spl

def set_GR_bpl(alow,ahigh,bm,**kwargs):

	x = np.linspace(l_logm,u_logm,1000)
	lkm = np.log(bm)*(alow-ahigh)
	y = np.where(x<np.log(bm), x*(alow+1), +lkm+x*(ahigh+1))
	GR_bpl = GeneralRandom(x,np.exp(y),1000)

	return GR_bpl

def set_GR_ln10full(mc,sm,mt,sl):

	x = np.linspace(l_logm,u_logm,1000)
	BMtr = x>=np.log(mt)
	lkm = np.exp(np.log(mt)*(sl+1)) / np.exp(-0.5* ((np.log(mt)/np.log(10)-np.log10(mc))/sm)**2)
	y = np.empty_like(x)        
	y[~BMtr] = np.exp(-0.5* ((x[~BMtr]/np.log(10)-np.log10(mc))/sm)**2)           
	y[BMtr]  =  np.exp(x[BMtr]*(sl+1))/lkm
	GR_ln10full = GeneralRandom(x,y,1000)

	return GR_ln10full

def set_GR_unif(bf):

	x = np.array([-1,-1e-6,0.,1])
	y = np.array([1-bf,1-bf,bf,bf])
	GR_unif = GeneralRandom(x,y,1000)

	return GR_unif