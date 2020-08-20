import pandas
import bz2
import pickle
import matplotlib.pyplot as plt
import numpy as np

class MagData:
    def __init__(self, **kwargs):

        if len(kwargs) > 0:
            print(kwargs)
            self.mags = kwargs['magnitudes']
            self.names = kwargs['names']
            self.data = pandas.DataFrame(self.mags, columns = self.names);

    def to_cmd(self, band1, band2):
        mag = self.data[band2]
        color = self.data[band1] - self.data[band2]
        return np.asarray([color,mag]).T

    def plot_cmd(self, band1 = None, band2 = None):
        
        if band1 is None:
            band1 = self.names[0]
        if band2 is None:
            band2 = self.names[1]
        
        mag = self.data[band2]
        color = self.data[band1] - self.data[band2]

        f = plt.figure(figsize = (8,5))
        plt.scatter(color, mag, s = 10, alpha = 0.75, color = 'k')
        plt.gca().invert_yaxis()
        plt.ylabel(band1)
        plt.xlabel(band1 + '$-$' + band2)
        return f

    def load_catalog(self, filename):
        with bz2.BZ2File(filename,'rb') as f: 
            catalog = pickle.load(f) 

        mask = catalog['dat_det']
        del catalog['dat_det']

        mags = [];
        names = [];
        for key in catalog.keys():
            mags.append(catalog[key])
            names.append(key)

        mags = np.asarray(mags).T[mask]
        self.mags = mags
        self.names = names
        self.data = pandas.DataFrame(self.mags, columns = self.names);
        print('catalog loaded!')
        
    def rename(self, newnames):
        self.names = newnames
        self.data.columns = newnames

class SimTable:
    def __init__(self, **kwargs):
        pass

    def load_simdict(self, sim_dict):
        if isinstance(sim_dict, str):
            with bz2.BZ2File(sim_dict, 'rb') as f: 
                print('loading dictionary...')
                simdict = pickle.load(f) 
    
        elif isinstance(sim_dict, list):
            with bz2.BZ2File(sim_dict,'rb') as f: 
                print('loading first dictionary...')
                simdict = pickle.load(f)

            simdict['outmag1'] = simdict['Output Mags'][:,0]
            simdict['outmag2'] = simdict['Output Mags'][:,1]

            for kk in range(len(sim_dict) - 1):
                with bz2.BZ2File(sim_dict[kk + 1]) as f: 
                    t_simdict = pickle.load(f)
                    t_simdict['outmag1'] = t_simdict['Output Mags'][:,0]
                    t_simdict['outmag2'] = t_simdict['Output Mags'][:,1]
                for key in simdict.keys():
                    simdict[key] = np.append(simdict[key], t_simdict[key])
                print('adding additional dictionary...')

            simdict['Output Mags'] = np.vstack((simdict['outmag1'],simdict['outmag2'])).T

            del simdict['outmag1']
            del simdict['outmag2']

        simdict['output_mag1'] = simdict['Output Mags'][:, 0]
        simdict['output_mag2'] = simdict['Output Mags'][:, 1]
        
        simdict['input_mag1'] = simdict['Input Mags'][:, 0]
        simdict['input_mag2'] = simdict['Input Mags'][:, 1]

        del simdict['Output Mags']
        del simdict['Input Mags']

        self.simdf = pandas.DataFrame.from_dict(simdict)
        print('simulation dataframe loaded!')