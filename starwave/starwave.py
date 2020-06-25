import matplotlib.pyplot as plt
import numpy as np
import pyabc
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import os
import sys
import functools

path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)
sys.path.append(dir_path)

from generalrandom import GeneralRandom
from distributions import *
from plot import *
from parameters import *

class StarWave:
    """
    StarWave: fitting the stellar birth function of resolved stellar populations 
    with approximate Bayesian computation. 
    
    """

    def __init__(self, simdf = None, imf_type = 'spl'):
        
        if simdf is not None:
            self.simdf = simdf
            self.base_weights = 1 / self.simdf['MassProb'] / self.simdf['BinProb']
            
        if isinstance(imf_type, (list, np.ndarray, tuple)):
            pass
        else:
            imf_type = [imf_type]
        self.imf_type = imf_type
        self.params = [make_params(imf) for imf in imf_type]
        
        print('initalized starwave with %s IMF and default priors' % imf_type)

    def init_scaler(self, observed_cmd, gamma = 0.5):
        self.cmd_scaler = MinMaxScaler()
        self.cmd_scaler.fit(observed_cmd);
        scaled_observed_cmd = self.cmd_scaler.transform(observed_cmd)
        Phi_approx = Nystroem(kernel = 'rbf', n_components=50, gamma = gamma) 
        Phi_approx.fit(scaled_observed_cmd)
        self.mapping = Phi_approx.transform
        print('scaler initialized and mapping defined!')
        return scaled_observed_cmd

    def get_cmd(self, nstars, gr_dict, simdf):
        
        weights = self.base_weights

        keys = ['logM', 'BinQ']

        for key in keys:
            new_prob = gr_dict[key].getpdf(self.simdf[key])
            weights = new_prob * weights
        
        weights = weights / np.nansum(weights)
        
        idx = np.random.choice(len(self.simdf), size = nstars, replace=True, p=weights)
        
        sel_df = self.simdf.iloc[idx]
        
        in_mags = sel_df[['input_mag1', 'input_mag2']].dropna().to_numpy()
        out_mags = sel_df[['output_mag1', 'output_mag2']].dropna().to_numpy()

        return in_mags, out_mags
    
    def make_cmd(self, mags):
        return np.asarray( [mags[:,0] - mags[:,1], mags[:,0]] ).T

    def sample_cmd(self, params, model = 'spl'):
            
        if model == 'spl':
            gr_dict = {'logM':set_GR_spl(params['slope']), 'BinQ': set_GR_unif(params['bf'])}
        elif model == 'bpl':
            gr_dict = {'logM':set_GR_bpl(params['alow'], params['ahigh'], params['bm']),\
                       'BinQ': set_GR_unif(params['bf'])}
        elif model == 'ln':
            gr_dict = {'logM':set_GR_ln10full(params['mean'], params['sigma'], params['bm'], \
                    params['slope']), 'BinQ': set_GR_unif(params['bf'])}
        else:
            print('Unrecognized model!')
        
        intensity = 10**params['log_int']
        nstars = int(stats.poisson.rvs(intensity))
        
        j_in, j_out = self.get_cmd(nstars, gr_dict, self.simdf)
        cmd_in = np.asarray([j_in[:,0] - j_in[:,1], j_in[:,0]]).T
        cmd_out = np.asarray([j_out[:,0] - j_out[:,1], j_out[:,0]]).T
        return cmd_in, cmd_out

    def sample_norm_cmd(self, params, model = 'spl'):
        in_cmd, out_cmd = self.sample_cmd(params, model)
        if len(in_cmd) == 0 or len(out_cmd) == 0:
            return np.zeros((1000,2))
        return self.cmd_scaler.transform(in_cmd), self.cmd_scaler.transform(out_cmd)

    def kernel_representation(self, P, mapping):
        Phi_P = mapping(P).sum(axis=0)
        return Phi_P

    def approx_kernel_distance(self, P, Q, mapping):
        Phi_P = self.kernel_representation(P, mapping)
        Phi_Q = self.kernel_representation(Q, mapping)
        return np.sqrt(np.sum((Phi_P - Phi_Q)**2))

    def exact_kernel_distance(self, P, Q, gamma):
        P = P[np.max(np.isfinite(P),1)]
        Q = Q[np.max(np.isfinite(Q),1)]
        PP = np.exp(- gamma*np.sum((P[:, None, :] - P[None, :, :])**2, axis=-1)).sum()
        QQ = np.exp(- gamma*np.sum((Q[:, None, :] - Q[None, :, :])**2, axis=-1)).sum()
        PQ = np.exp(- gamma*np.sum((P[:, None, :] - Q[None, :, :])**2, axis=-1)).sum()
        return np.sqrt(PP + QQ - 2 * PQ)

    def cmd_sim(self, params, imf_type):
        in_cmd, out_cmd = self.sample_norm_cmd(params, model = imf_type)
        return {'input': self.kernel_representation(in_cmd, self.mapping),
                'output': self.kernel_representation(out_cmd, self.mapping)}
    
    def fit_cmd(self, observed_cmd, pop_size = 1000, max_n_pop = np.Inf, savename = 'starwave', min_acceptance_rate = 0.0001, gamma = 0.5, 
                    cores = 1, accept = 'uniform', alpha = 0.5, population_strategy = 'constant'):


        if cores == 1:
            pyabc_sampler = pyabc.sampler.SingleCoreSampler()
        elif cores > 1:
            pyabc_sampler = pyabc.sampler.MulticoreEvalParallelSampler(n_procs = cores)
        else:
            print('invalid number of cores. defaulting to 1 core.')
            pyabc_sampler = pyabc.sampler.SingleCoreSampler()

        if population_strategy == 'constant':
            population_strategy = pyabc.populationstrategy.ConstantPopulationSize(pop_size)
        elif population_strategy == 'adapt':
            population_strategy = pyabc.populationstrategy.AdaptivePopulationSize(pop_size)


        scaled_observed_cmd = self.init_scaler(observed_cmd, gamma = gamma)

        # if not isinstance(gamma, str):
        #     gamma = gamma
        # elif gamma == 'heuristic':
        #     # KDT = KDTree(scaled_observed_cmd)
        #     # dd, ind = KDT.query(scaled_observed_cmd, k=2)
        #     # avmindist = np.mean(dd[:,1])
        #     # sigma = sigmacorr*avmindist
        #     # gamma = 0.5/(sigma**2)
        #     # print('setting kernel gamma = %.1f'%gamma)
        #     gamma = 0.5
        # sigmacorr = 3

        # plt.scatter(observed_cmd[:, 0], observed_cmd[:,1])

        # R = np.random.uniform(0, 1, (len(observed_cmd),2))

        obs = dict(input = self.kernel_representation(scaled_observed_cmd, self.mapping),
                   output = self.kernel_representation(scaled_observed_cmd, self.mapping))

        dummy_cmd = np.zeros(observed_cmd.shape)
        
        def simcmd(imf_type):
            return lambda params: self.cmd_sim(params, imf_type = imf_type)
        
        simulator = [];
        prior = [];
        for idx,imf in enumerate(self.imf_type):
            simulator.append(simcmd(imf))
            prior.append(self.params[idx].to_pyabc())        


        if accept == 'uniform':
            acceptor = pyabc.acceptor.UniformAcceptor()
            eps = pyabc.epsilon.QuantileEpsilon(alpha = alpha)
            def distance(cmd1, cmd2):
                return np.sqrt(np.sum((cmd1['output'] - cmd2['output'])**2))

        elif accept == 'stochastic':
            acceptor = pyabc.StochasticAcceptor()
            eps = pyabc.Temperature()
            
            base_params = make_params(self.imf_type[0]).get_values()

            sim_rep = np.asarray([self.cmd_sim(base_params, imf_type = self.imf_type[0])['output'] for ii in range(25)])

            var = np.var(sim_rep, 0)

            distance = pyabc.IndependentNormalKernel(var = var, keys = ['input'])

        abc = pyabc.ABCSMC(simulator, 
                            prior,
                            distance, 
                            sampler = pyabc_sampler,
                            population_size = pop_size, 
                            eps = eps,
                            acceptor = acceptor)

        db_path = ("sqlite:///" + savename + ".db")

        abc.new(db_path, obs);

        self.history = abc.run(min_acceptance_rate = min_acceptance_rate, max_nr_populations = max_n_pop)

        return self.history

    def gof_lf(self, df, w, observed_cmd, imf_type, n_samples = 25, kde = False, n_bins = 35, color = True):

        if imf_type == 'spl':
            simulator = self.cmd_sim_spl
        elif imf_type == 'bpl':
            simulator = self.cmd_sim_bpl
        elif imf_type == 'ln':
            simulator = self.cmd_sim_ln

        idxs = np.arange(len(df))
        post_samples = df.iloc[np.random.choice(idxs, size = n_samples, p = w)]

        self.cmd_scaler = MinMaxScaler()
        self.cmd_scaler.fit(observed_cmd)

        cmds = [self.cmd_scaler.inverse_transform(simulator(sample)['data']) for _,sample in post_samples.iterrows()]
        
        if kde:
            return plot_lfs_kde(cmds)
        else:
            if color:
                return plot_lfs(cmds, n_bins = n_bins, axis = 1), plot_lfs(cmds, n_bins = n_bins, axis = 0)
            return plot_lfs(cmds, n_bins = n_bins)

    #def gof_lf(self, df, w, observed_cmd, imf_type, n_samples = 25, n_bins = 35):


    def load_history(self, dbpath, id):
        def fakesim(p):
            return dict(null = p)

        dummy_abc = pyabc.ABCSMC(fakesim, None, None, sampler = pyabc.sampler.SingleCoreSampler())

        return dummy_abc.load("sqlite:///" + dbpath, id)
    
if __name__ == '__main__':
    sw = StarWave()
    sw.params.pretty_print()