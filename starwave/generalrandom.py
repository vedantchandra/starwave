import matplotlib.pyplot as plt
import numpy as np
#from numba import jit
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz

class GeneralRandom:
  """This class enables us to generate random numbers with an arbitrary 
  distribution. Interpolation between provided samples is linear"""
  
  def __init__(self, x, p, Nrl):
    """Initialize the lookup table (with default values if necessary)
    Inputs:
    x = random number values
    p = probability density profile at that point
    Nrl = number of reverse look up values between 0 and 1"""  
    self.set_pdf(x, p, Nrl)
 # @jit
  def set_pdf(self, x, p, Nrl):
    """Generate the lookup tables. 
    x is the value of the random variate
    pdf is its probability density
    cdf is the cumulative pdf
    inversecdf is the inverse look up table
    
    """
    
    self.x = x
    self.pdf = p/np.trapz(p,self.x) #normalize it
    self.cdf = cumtrapz(self.pdf,self.x,initial=0.)
    self.inversecdfbins = Nrl
    self.Nrl = Nrl
    y = np.arange(Nrl)/float(Nrl)
    delta = 1.0/Nrl
    self.inversecdf = np.zeros(Nrl)    
    self.inversecdf[0] = self.x[0]
    cdf_idx = 0
    for n in range(1,self.inversecdfbins):
      #      while self.cdf[cdf_idx] < y[n] and cdf_idx < Nrl:
      while ((self.cdf[cdf_idx] < y[n]) & (cdf_idx < Nrl)):
        cdf_idx += 1
      self.inversecdf[n] = self.x[cdf_idx-1] + (self.x[cdf_idx] - self.x[cdf_idx-1]) * (y[n] - self.cdf[cdf_idx-1])/(self.cdf[cdf_idx] - self.cdf[cdf_idx-1]) 
      if cdf_idx >= Nrl:
        break
    self.delta_inversecdf = np.concatenate((np.diff(self.inversecdf), [0]))
    self.intp_pdf = interp1d(self.x,self.pdf,fill_value=0.,bounds_error=False)

  #@jit
  def random(self, N = 1):
    """Give us N random numbers with the requested distribution"""

    idx_f = np.random.uniform(size = N, high = self.Nrl-1)
    idx = np.array([idx_f],'i')
    y = self.inversecdf[idx] + (idx_f - idx)*self.delta_inversecdf[idx]

    return y.T

  def sample(self, N = 1):
    return self.random(N)[:,0]

  #@jit
  def getpdf(self, x):
    """Get pdf at position x"""

    return self.intp_pdf(x)
  
  def plot_pdf(self):
    plt.plot(self.x, self.pdf)
    
  def self_test(self, N = 1000):
    plt.figure()
    #The cdf
    plt.subplot(2,2,1)
    plt.plot(self.x, self.cdf)
    #The inverse cdf
    plt.subplot(2,2,2)
    y = np.arange(self.Nrl)/float(self.Nrl)
    plt.plot(y, self.inversecdf)
    
    #The actual generated numbers
    plt.subplot(2,2,3)
    y = self.random(N)
    plt.hist(y, bins = 50,
           range = (self.x.min(), self.x.max()), 
           normed = True)
    plt.plot(self.x, self.pdf)
