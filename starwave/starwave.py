import matplotlib.pyplot as plt
import numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import os
import sys
import functools
from sklearn.neighbors import KDTree,NearestNeighbors

path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)
sys.path.append(dir_path)

from generalrandom import GeneralRandom
from distributions import *
from plot import *
from parameters import *
from getmags import *
import intNN

import torch
import sbi
from sbi import utils as utils
from sbi import user_input
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.utils.get_nn_models import posterior_nn

class StarWave:
    """
    StarWave: fitting the stellar birth function of resolved stellar populations 
    with approximate Bayesian computation. 
    
    """

    def __init__(self, isodf, asdf, bands, imf_type, sfh_type = 'gaussian'):

        self.imf_type = imf_type
        self.sfh_type = sfh_type
        self.params = make_params(imf_type, sfh_type)
        self.make_prior(self.params) ## INITIALIZE FIXED PARAMS VECTOR
        self.bands = bands
        self.iso_int = intNN.intNN(isodf, self.bands)
        self.asdf = asdf
        self.return_inputmags = False

        self.bands_in = [band + '_in' for band in bands]
        self.bands_out = [band + '_out' for band in bands]

        self.asdf_noise = self.asdf[self.bands_out].to_numpy() - self.asdf[self.bands_in].to_numpy()

        self.kdtree = KDTree(asdf[self.bands_in])
        self.trgb = -100
        self.lim_logmass = np.log(0.1)

        self.debug = False
        
        print('initalized starwave with %s bands, %s IMF, and default priors' % (str(bands), imf_type))
        print_prior_summary(self.params)

    def init_scaler(self, observed_cmd, gamma = 0.5):
        self.cmd_scaler = MinMaxScaler()
        self.cmd_scaler.fit(observed_cmd);
        scaled_observed_cmd = self.cmd_scaler.transform(observed_cmd)
        Phi_approx = Nystroem(kernel = 'rbf', n_components=50, gamma = gamma) 
        Phi_approx.fit(scaled_observed_cmd)
        self.mapping = Phi_approx.transform
        print('scaler initialized and mapping defined!')
        return scaled_observed_cmd

    def get_cmd(self, nstars, gr_dict):

        input_mags = np.empty((nstars, len(self.bands)))
        input_mags[:] = np.nan

        masses = gr_dict['logM'].sample(nstars)
        binqs = gr_dict['BinQ'].sample(nstars)
        sfhs = gr_dict['SFH'].sample(nstars)
        dms = gr_dict['DM'].sample(nstars)


        for ii in range(nstars):

            mass = masses[ii]
            binq = binqs[ii]
            age, feh = sfhs[ii]
            dm = dms[ii]

            if mass < self.lim_logmass or np.isnan(age) or np.isnan(feh):
                continue

            input_mag = get_absolute_mags(mass, age, feh, binq, self.iso_int, self.bands)

            input_mags[ii, :] = input_mag + dm

        nans = (np.isnan(input_mags) + (input_mags < self.trgb)).any(axis = 1)

        input_mags = input_mags[~nans]

        if len(input_mags) == 0:
            return input_mags, input_mags

        idxs = self.kdtree.query(input_mags)[1][:, 0]

        output_mags = input_mags + self.asdf_noise[idxs]

        nans = np.isnan(output_mags).any(axis = 1)

        output_mags = output_mags[~nans]


        return input_mags, output_mags 
    
    def make_cmd(self, mags):
        cmd = mags
        for ii in range(mags.shape[1] - 1):
            cmd[:, ii + 1] -= cmd[:, 0]

        return cmd

    def best_gamma(self, cmd, q = 0.68, fac = 1, NN = 5): # pass a scaled CMD
        nbr = NearestNeighbors(n_neighbors = NN, algorithm = 'kd_tree', metric = 'minkowski', p = 2)
        nbr.fit(cmd)
        dst, idx = nbr.kneighbors(cmd, return_distance = True)
        dst = dst[:, -1] # pick NNth distance

        best_dist = np.quantile(dst, q)
        gamma = 1 / (2 * (fac * best_dist)**2)

        return gamma


    def set_sfh_dist(self, pdict, sfh_type):

        if sfh_type == 'gaussian':
            cov = pdict['age_feh_corr'] * pdict['sig_age'] * pdict['sig_feh']
            covmat = np.array([[pdict['sig_age'], cov], [cov, pdict['sig_feh']]])


            if not isPD(covmat):
                covmat = nearestPD(covmat)
                print('found nearest SFH covmat...')

            means = np.array([pdict['age'], pdict['feh']])

            return SW_SFH(stats.multivariate_normal(mean = means, cov = covmat, allow_singular = True))

    def make_prior(self, parameters):
    
        priors = [];
        self.fixed_params = {};
        self.param_mapper = {};
        idx = 0

        for ii,(name, param) in enumerate(parameters.dict.items()):
            
            if param.fixed:
                self.fixed_params[name] = param.value
                continue
            
            
            lower = param.bounds[0]
            upper = param.bounds[1]
            
            if param.distribution == 'uniform':
                distribution = torch.distributions.Uniform(lower*torch.ones(1), upper*torch.ones(1))
                priors.append(distribution)
                
            elif param.distribution == 'norm':
                try:
                    mean = param.dist_kwargs['mean']
                    sigma = param.dist_kwargs['sigma']
                except:
                    raise ValueError('please pass valid distribution arguments!')
                distribution = torch.distributions.Normal(torch.tensor(mean), torch.tensor(sigma))
                priors.append(distribution)
                
            else:
                raise ValueError('invalid distribution name')

            self.param_mapper[name] = idx
            idx += 1 # IDX maps the vector of sampled parameters, leaving apart the fixed ones. 
    
        return priors

    def sample_cmd(self, params, model):

        is_pdict = False

        if isinstance(params, torch.FloatTensor):
            params = params.detach().cpu().numpy()
        elif isinstance(params, (list, np.ndarray)):
            pass
        elif isinstance(params, dict):
            pdict = params
            is_pdict = True


        self.make_prior(self.params) # Re-initialize priors, check fixed parameters

        pdict = {};
        for name in self.params.keys():
            if name in self.fixed_params.keys():
                pdict[name] = self.fixed_params[name]
            else:
                if is_pdict:
                    pdict[name] = params[name] # if params are dictionary  
                else:
                    pdict[name] = params[self.param_mapper[name]] # if params are array or tensor

        # if self.debug:
        #     print('param dictionary in sample_cmd:' + str(pdict))

        if model == 'spl':
            gr_dict = {'logM':set_GR_spl(pdict['slope'])}
        elif model == 'bpl':
            gr_dict = {'logM':set_GR_bpl(pdict['alow'], pdict['ahigh'], pdict['bm'])}
        elif model == 'ln':
            gr_dict = {'logM':set_GR_ln10full(pdict['mean'], pdict['sigma'], pdict['bm'], pdict['slope'])}
        else:
            print('Unrecognized model!')

        gr_dict['BinQ'] = set_GR_unif(pdict['bf'])
        gr_dict['SFH'] = self.set_sfh_dist(pdict, self.sfh_type)
        gr_dict['DM'] = SWDist(stats.norm(loc = pdict['dm'], scale = pdict['sig_dm']))

        intensity = 10**pdict['log_int']
        nstars = int(stats.poisson.rvs(intensity))
        
        mags_in, mags_out = self.get_cmd(nstars, gr_dict)
        cmd_in = self.make_cmd(mags_in)
        cmd_out = self.make_cmd(mags_out)
        return cmd_in, cmd_out

    def sample_norm_cmd(self, params, model):
        in_cmd, out_cmd = self.sample_cmd(params, model)
        if len(in_cmd) == 0 or len(out_cmd) == 0:
            print('empty cmd!')
            return self.dummy_cmd, self.dummy_cmd
        return self.cmd_scaler.transform(in_cmd), self.cmd_scaler.transform(out_cmd)

    def kernel_representation(self, P, mapping):
        Phi_P = mapping(P).sum(axis=0)
        return Phi_P

    def cmd_sim(self, params, imf_type):
        in_cmd, out_cmd = self.sample_norm_cmd(params, model = imf_type)
        if self.return_inputmags:
            return self.kernel_representation(in_cmd, self.mapping)
        else:
            return self.kernel_representation(out_cmd, self.mapping)

    def fit_cmd(self, observed_cmd, n_rounds = 5, n_sims = 100, savename = 'starwave', min_acceptance_rate = 0.0001, gamma = None, 
                    cores = 1, alpha = 0.5,
                    statistic = 'output',
                    gamma_kw = {}):


        if cores == 1:
            pass # IMPLEMENT SBI MULTICORE

        scaled_observed_cmd = self.init_scaler(observed_cmd, gamma = gamma)
        obs = torch.tensor(self.kernel_representation(scaled_observed_cmd, self.mapping))
        self.obs = obs

        if gamma is None:
            print('finding optimal kernel width...')
            gamma = self.best_gamma(scaled_observed_cmd, **gamma_kw)
            print('setting gamma = %i' % gamma)

        self.dummy_cmd = np.zeros(observed_cmd.shape)
        
        def simcmd(imf_type):
            return lambda params: self.cmd_sim(params, imf_type = imf_type)

        Nobs = len(scaled_observed_cmd)

        #self.params['log_int'].set(value = np.log10(Nobs), bounds = [np.log10(Nobs/2) , np.log10(Nobs*10)])

        print_prior_summary(self.params)
        
        prior = user_input.user_input_checks.MultipleIndependent(self.make_prior(self.params))
        simulator = simcmd(self.imf_type)

        simulator,prior = prepare_for_sbi(simulator,prior)

        inference = SNPE(prior = prior)

        self.posteriors = [];
        proposal = prior


        for _ in range(n_rounds):
            print('Starting round %i of neural inference...' % (_+1))
            theta, x = simulate_for_sbi(simulator, proposal, num_simulations=n_sims, num_workers = cores)
            density_estimator = inference.append_simulations(theta, x, proposal=proposal).train()
            posterior = inference.build_posterior(density_estimator)
            self.posteriors.append(posterior)
            proposal = posterior.set_default_x(obs)

        return self.posteriors[-1]

    # def gof_lf(self, df, w, observed_cmd, imf_type, n_samples = 25, kde = False, n_bins = 35, color = True):

    #     if imf_type == 'spl':
    #         simulator = self.cmd_sim_spl
    #     elif imf_type == 'bpl':
    #         simulator = self.cmd_sim_bpl
    #     elif imf_type == 'ln':
    #         simulator = self.cmd_sim_ln

    #     idxs = np.arange(len(df))
    #     post_samples = df.iloc[np.random.choice(idxs, size = n_samples, p = w)]

    #     self.cmd_scaler = MinMaxScaler()
    #     self.cmd_scaler.fit(observed_cmd)

    #     cmds = [self.cmd_scaler.inverse_transform(simulator(sample)['data']) for _,sample in post_samples.iterrows()]
        
    #     if kde:
    #         return plot_lfs_kde(cmds)
    #     else:
    #         if color:
    #             return plot_lfs(cmds, n_bins = n_bins, axis = 1), plot_lfs(cmds, n_bins = n_bins, axis = 0)
    #         return plot_lfs(cmds, n_bins = n_bins)

    #def gof_lf(self, df, w, observed_cmd, imf_type, n_samples = 25, n_bins = 35):


    def load_history(self, dbpath, id):
        def fakesim(p):
            return dict(null = p)

        dummy_abc = pyabc.ABCSMC(fakesim, None, None, sampler = pyabc.sampler.SingleCoreSampler())

        return dummy_abc.load("sqlite:///" + dbpath, id)
    
if __name__ == '__main__':
    sw = StarWave()
    sw.params.pretty_print()

# class StarWave_pyabc:
#     """
#     StarWave: fitting the stellar birth function of resolved stellar populations 
#     with approximate Bayesian computation. 
    
#     """

#     def __init__(self, bands = ['ACS_HRC_F606W', 'ACS_HRC_F814W'], imf_type = 'spl'):
                    
#         if isinstance(imf_type, (list, np.ndarray, tuple)):
#             pass
#         else:
#             imf_type = [imf_type]

#         self.imf_type = imf_type
#         self.params = [make_params(imf) for imf in imf_type]
#         self.bands = bands
#         self.iso_int = minimint.Interpolator(bands)
        
#         print('initalized starwave with %s IMF and default priors' % imf_type)

#     def init_scaler(self, observed_cmd, gamma = 0.5):
#         self.cmd_scaler = MinMaxScaler()
#         self.cmd_scaler.fit(observed_cmd);
#         scaled_observed_cmd = self.cmd_scaler.transform(observed_cmd)
#         Phi_approx = Nystroem(kernel = 'rbf', n_components=50, gamma = gamma) 
#         Phi_approx.fit(scaled_observed_cmd)
#         self.mapping = Phi_approx.transform
#         print('scaler initialized and mapping defined!')
#         return scaled_observed_cmd

#     def get_cmd(self, nstars, gr_dict):

#         age = 8 ## logAge, Incorporate this into GR_dict too along with Av, etc. Maybe replace with scipy objects? 
#         feh = -1 ## same as above

#         cmd = np.empty((nstars, len(self.bands)))
#         cmd[:] = np.nan

#         for ii in range(nstars):
#             mass = gr_dict['logM'].sample(1)[0]
#             age = age
#             feh = feh
#             binq = gr_dict['BinQ'].sample(1)[0]

#             input_mag = get_absolute_mags(mass, age, feh, binq, self.iso_int, self.bands)

#             cmd[ii, :] = input_mag

#         nans = np.isnan(cmd).any(axis = 1)
#         cmd = cmd[~nans]

#         return cmd, cmd # make input and output

        
#         # weights = self.base_weights

#         # keys = ['logM', 'BinQ']

#         # for key in keys:
#         #     new_prob = gr_dict[key].getpdf(self.simdf[key])
#         #     weights = new_prob * weights
        
#         # weights = weights / np.nansum(weights)
        
#         # idx = np.random.choice(len(self.simdf), size = nstars, replace=True, p=weights)
        
#         # sel_df = self.simdf.iloc[idx]
        
#         # in_mags = sel_df[['input_mag1', 'input_mag2']].dropna().to_numpy()
#         # out_mags = sel_df[['output_mag1', 'output_mag2']].dropna().to_numpy()

#         #return in_mags, out_mags
    
#     def make_cmd(self, mags):
#         return np.asarray( [mags[:,0] - mags[:,1], mags[:,1]] ).T

#     def sample_cmd(self, params, model = 'spl'):

#         for key in params.keys():
#             if isinstance(params[key], torch.FloatTensor):
#                 params[key] = params[key].detach().cpu().numpy()
            
#         if model == 'spl':
#             gr_dict = {'logM':set_GR_spl(params['slope']), 'BinQ': set_GR_unif(params['bf'])}
#         elif model == 'bpl':
#             gr_dict = {'logM':set_GR_bpl(params['alow'], params['ahigh'], params['bm']),\
#                        'BinQ': set_GR_unif(params['bf'])}
#         elif model == 'ln':
#             gr_dict = {'logM':set_GR_ln10full(params['mean'], params['sigma'], params['bm'], \
#                     params['slope']), 'BinQ': set_GR_unif(params['bf'])}
#         else:
#             print('Unrecognized model!')
        
#         intensity = 10**params['log_int']
#         nstars = int(stats.poisson.rvs(intensity))
        
#         j_in, j_out = self.get_cmd(nstars, gr_dict)
#         cmd_in = np.asarray([j_in[:,0] - j_in[:,1], j_in[:,1]]).T
#         cmd_out = np.asarray([j_out[:,0] - j_out[:,1], j_out[:,1]]).T
#         return cmd_in, cmd_out

#     def sample_norm_cmd(self, params, model = 'spl'):
#         in_cmd, out_cmd = self.sample_cmd(params, model)
#         if len(in_cmd) == 0 or len(out_cmd) == 0:
#             return np.zeros((1000,2)), np.zeros((1000, 2))
#         return self.cmd_scaler.transform(in_cmd), self.cmd_scaler.transform(out_cmd)

#     def kernel_representation(self, P, mapping):
#         Phi_P = mapping(P).sum(axis=0)
#         return Phi_P

#     def approx_kernel_distance(self, P, Q, mapping):
#         Phi_P = self.kernel_representation(P, mapping)
#         Phi_Q = self.kernel_representation(Q, mapping)
#         return np.sqrt(np.sum((Phi_P - Phi_Q)**2))

#     def exact_kernel_distance(self, P, Q, gamma):
#         P = P[np.max(np.isfinite(P),1)]
#         Q = Q[np.max(np.isfinite(Q),1)]
#         PP = np.exp(- gamma*np.sum((P[:, None, :] - P[None, :, :])**2, axis=-1)).sum()
#         QQ = np.exp(- gamma*np.sum((Q[:, None, :] - Q[None, :, :])**2, axis=-1)).sum()
#         PQ = np.exp(- gamma*np.sum((P[:, None, :] - Q[None, :, :])**2, axis=-1)).sum()
#         return np.sqrt(PP + QQ - 2 * PQ)

#     def cmd_sim(self, params, imf_type):
#         in_cmd, out_cmd = self.sample_norm_cmd(params, model = imf_type)
#         return {'output': self.kernel_representation(out_cmd, self.mapping)}
#         #'input': self.kernel_representation(in_cmd, self.mapping),
#     def fit_cmd(self, observed_cmd, pop_size = 1000, max_n_pop = np.Inf, savename = 'starwave', min_acceptance_rate = 0.0001, gamma = 0.5, 
#                     cores = 1, accept = 'uniform', alpha = 0.5, population_strategy = 'constant',
#                     statistic = 'output'):


#         if cores == 1:
#             pyabc_sampler = pyabc.sampler.SingleCoreSampler()
#         elif cores > 1:
#             pyabc_sampler = pyabc.sampler.MulticoreEvalParallelSampler(n_procs = cores)
#         else:
#             print('invalid number of cores. defaulting to 1 core.')
#             pyabc_sampler = pyabc.sampler.SingleCoreSampler()

#         if population_strategy == 'constant':
#             population_strategy = pyabc.populationstrategy.ConstantPopulationSize(pop_size)
#         elif population_strategy == 'adapt':
#             population_strategy = pyabc.populationstrategy.AdaptivePopulationSize(pop_size)


#         scaled_observed_cmd = self.init_scaler(observed_cmd, gamma = gamma)

#         obs = dict(output = self.kernel_representation(scaled_observed_cmd, self.mapping))

#         dummy_cmd = np.zeros(observed_cmd.shape)
        
#         def simcmd(imf_type):
#             return lambda params: self.cmd_sim(params, imf_type = imf_type)
        
#         simulator = [];
#         prior = [];
#         for idx,imf in enumerate(self.imf_type):
#             simulator.append(simcmd(imf))
#             prior.append(self.params[idx].to_pyabc())        


#         if accept == 'uniform':
#             acceptor = pyabc.acceptor.UniformAcceptor()
#             eps = pyabc.epsilon.QuantileEpsilon(alpha = alpha)
#             def distance(cmd1, cmd2):
#                 return np.sqrt(np.sum((cmd1[statistic] - cmd2[statistic])**2))

#         elif accept == 'stochastic':
#             acceptor = pyabc.StochasticAcceptor()
#             eps = pyabc.Temperature()
#             base_params = make_params(self.imf_type[0]).get_values()
#             sim_rep = np.asarray([self.cmd_sim(base_params, imf_type = self.imf_type[0])['output'] for ii in range(25)])
#             var = np.var(sim_rep, 0)
#             distance = pyabc.IndependentNormalKernel(var = var, keys = ['input'])

#         abc = pyabc.ABCSMC(simulator, 
#                             prior,
#                             distance, 
#                             sampler = pyabc_sampler,
#                             population_size = pop_size, 
#                             eps = eps,
#                             acceptor = acceptor)

#         db_path = ("sqlite:///" + savename + ".db")

#         abc.new(db_path, obs);

#         self.history = abc.run(min_acceptance_rate = min_acceptance_rate, max_nr_populations = max_n_pop)

#         return self.history

#     def gof_lf(self, df, w, observed_cmd, imf_type, n_samples = 25, kde = False, n_bins = 35, color = True):

#         if imf_type == 'spl':
#             simulator = self.cmd_sim_spl
#         elif imf_type == 'bpl':
#             simulator = self.cmd_sim_bpl
#         elif imf_type == 'ln':
#             simulator = self.cmd_sim_ln

#         idxs = np.arange(len(df))
#         post_samples = df.iloc[np.random.choice(idxs, size = n_samples, p = w)]

#         self.cmd_scaler = MinMaxScaler()
#         self.cmd_scaler.fit(observed_cmd)

#         cmds = [self.cmd_scaler.inverse_transform(simulator(sample)['data']) for _,sample in post_samples.iterrows()]
        
#         if kde:
#             return plot_lfs_kde(cmds)
#         else:
#             if color:
#                 return plot_lfs(cmds, n_bins = n_bins, axis = 1), plot_lfs(cmds, n_bins = n_bins, axis = 0)
#             return plot_lfs(cmds, n_bins = n_bins)

#     #def gof_lf(self, df, w, observed_cmd, imf_type, n_samples = 25, n_bins = 35):


#     def load_history(self, dbpath, id):
#         def fakesim(p):
#             return dict(null = p)

#         dummy_abc = pyabc.ABCSMC(fakesim, None, None, sampler = pyabc.sampler.SingleCoreSampler())

#         return dummy_abc.load("sqlite:///" + dbpath, id)
    
# if __name__ == '__main__':
#     sw = StarWave()
#     sw.params.pretty_print()