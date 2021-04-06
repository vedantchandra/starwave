'''
Author: Mario Gennaro

Define a class that can be used to interpolate isocrones
in mass, age and metallicity.
Initialization requires a pandas dataframe of isocrones,
with multiindex ('[Fe/H]', 'age','mass') and
a photband object
'''

import findNN_arr
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm

class intNN:
    
    def __init__(self, isoPD, photbands):

        self.isoages = np.unique(np.asarray([isoPD.index.get_level_values('age')]).T)

        self.logages = np.log10(self.isoages * 1e9)

        self.l_logage = np.min(self.logages)
        self.u_logage = np.max(self.logages)

        self.isomets = np.unique(np.asarray([isoPD.index.get_level_values('[Fe/H]')]).T)
        self.iso_intp = {};
        for band in photbands:
            self.iso_intp[band] = [[0 for x in range(len(self.isomets))] for y in range(len(self.isoages))]
        self.iso_mrng = [[0 for x in range(len(self.isomets))] for y in range(len(self.isoages))]
        self.photbands = photbands

        print('interpolating %i ages and %i metallicities...' % (len(self.isoages), len(self.isomets)))

        for aa, age in tqdm(enumerate(self.isoages)):

            for zz, met in enumerate(self.isomets):
                isomass = np.asarray(isoPD.loc[met].loc[age].index.get_level_values('mass'))
                self.iso_mrng[aa][zz] = ([np.amin(isomass),np.amax(isomass)])

                isomags  = isoPD.loc[met].loc[age][self.photbands]

                for band in self.photbands:
                    self.iso_intp[band][aa][zz] = interp1d(isomass, isomags[band], kind='linear', assume_sorted=True,
                                                 bounds_error=False, fill_value=np.nan)
    def __call__(self,mss,logage,met):

        if logage < self.l_logage or logage > self.u_logage:
            print('intNN warning: age out of bounds')

        age = 10**logage * 1e-9
        nage_idx = findNN_arr.find_nearest_idx(self.isoages,age)
        nmet_idx = findNN_arr.find_nearest_idx(self.isomets,met)
        mags = [];

        isointp  = {band: self.iso_intp[band][nage_idx][nmet_idx](mss) for band in self.photbands}

        return isointp

   
