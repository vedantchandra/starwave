import sys
import os 
import numpy as np
import numpy.linalg as la

path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)
sys.path.append(dir_path)

from generalrandom import GeneralRandom

l_logm = np.log(0.05) # lower limit on log stellar mass
u_logm = np.log(8) # upper limit on log stellar mass

l_age = 8  # lower limit on log stellar age in Gyr
u_age = 10.1249 # upper limit on log stellar age in Gyr

l_feh = -4 # lower limit on [Fe/H]
u_feh = 1 # upper limit on [Fe/H]

# def get_near_psd(A):
#     C = (A + A.T)/2
#     eigval, eigvec = np.linalg.eig(C)
#     eigval[eigval < 0] = 1e-5
s
#     return eigvec.dot(np.diag(eigval)).dot(eigvec.T)

from numpy import linalg as la

def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    :param A: input array
    :type A: array
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky
    :param B: input array
    :type B: bytearray
    """
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False



class SW_SFH:
    '''
    wraps around a scipy distribution to give it a sample() method
    '''

	def __init__(self, scipy_dist):
		self.scipy_dist = scipy_dist

	def sample(self, N):
		sfh = self.scipy_dist.rvs(N)
		age, feh = sfh.T
		within = (age > l_age) * (age < u_age) * (feh > l_feh) * (feh < u_feh)
		age[~within] = np.nan
		feh[~within] = np.nan
		#age = np.log10(age * 1e9) # CONVERT TO LOG AGE FOR ISOCHRONE
		return np.vstack((age, feh)).T

class GridSFH:
    '''
    loads and samples a grid-based SFH from a dictionary of ages, metallicities, and weights
    '''
    
    def __init__(self, sfh_grid):
        '''
        Assumes sfh_grid is a dictionary with the following keys:
        'mets' : array of M [Fe/H] grid points
        'ages' : array of A age (Gyr) grid points
        'probabilities' : M x A matrix with probability (or some weight) of each SFH bin
        :param sfh_grid: SFH grid
        :type sfh_grid: dict
        '''

        self.sfh_grid = sfh_grid
        mets, ages, probs = sfh_grid['mets'], sfh_grid['ages'], sfh_grid['probabilities']
        MM, AA = np.meshgrid(mets[:-1], ages[:-1])
        
        self.mm = MM.ravel()
        self.aa = AA.ravel()
        self.pp = probs.ravel() / np.sum(probs)
        self.idxs = np.arange(len(self.pp))

        self.dm = np.diff(mets)[0]
        self.da = np.diff(ages)[0]
        
        self.rng = np.random.default_rng()
    
    def sample(self, N): 
        sel_idx = self.rng.choice(self.idxs, p = self.pp, size = N)
        sampled_m = self.mm[sel_idx] + self.rng.uniform(0, self.dm, size = N)
        sampled_a = self.aa[sel_idx] + self.rng.uniform(0, self.da, size = N)
        
        return np.vstack((sampled_a, sampled_m)).T

def set_GR_spl(slope):
    '''
    defines a GeneralRandom object for a single power-law (Salpeter) IMF
    :param slope: slope of the IMF
    :type slope: float
    :return: GR_spl
    :rtype: GeneralRandom object
    '''
		x = np.linspace(l_logm,u_logm,1000)
		y = np.exp(x*(slope+1))
		GR_spl = GeneralRandom(x,y,1000)
		return GR_spl

def set_GR_bpl(alow,ahigh,bm):
    '''
    defines a GeneralRandom object for a double power-law (Kroupa) IMF
    :param alow: low-mass IMF slope
    :type alow: float
    :param ahigh: high-mass IMF slope
    :type ahigh: float
    :param bm: break mass
    :type bm: float
    :return: GR_bpl
    :rtype: GeneralRandom object
    '''
	x = np.linspace(l_logm,u_logm,1000)
	lkm = np.log(bm)*(alow-ahigh)
	y = np.where(x<np.log(bm), x*(alow+1), +lkm+x*(ahigh+1))
	GR_bpl = GeneralRandom(x,np.exp(y),1000)

	return GR_bpl

def set_GR_ln10full(mc,sm,mt,sl):
    '''
    defines a GeneralRandom object for a log-normal (Chabrier) IMF
    :param mc: mean mass
    :type mc: float
    :param sm: sigma mass
    :type sm: float
    :param mt: transition mass to power-law
    :type mt: float
    :param sl: power-law slope after transition mass
    :type sl: float
    :return: GR_ln10full
    :rtype: GeneralRandom object
    '''

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