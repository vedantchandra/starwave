import matplotlib.pyplot as plt
import numpy as np
import pickle,bz2,sys 
import pyabc
import tempfile
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import MinMaxScaler
from GeneralRandom import GeneralRandom
import glob
from scipy import stats
import pandas as pd

def fit_cmd(catalog, sim_dict, imf_type, pop_size, max_n_pop, savename):

	with bz2.BZ2File(catalog,'rb') as f: 
		catalog = pickle.load(f) 

	if isinstance(sim_dict, str):
		with bz2.BZ2File(sim_dict, 'rb') as f: 
			simdict = pickle.load(f) 
	
	elif isinstance(sim_dict, list):
		with bz2.BZ2File(sim_dict,'rb') as f: 
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

		simdict['Output Mags'] = np.vstack((simdict['outmag1'],simdict['outmag2'])).T

		del simdict['outmag1']
		del simdict['outmag2']
		del simdict['Input Mags']

	simdict['output_mag1'] = simdict['Output Mags'][:, 0]

	simdict['output_mag2'] = simdict['Output Mags'][:, 1]

	del simdict['Output Mags']
	del simdict['Input Mags']

	simdf = pd.DataFrame.from_dict(simdict)

	l_logm = np.min(simdict['logM'])
	u_logm = np.max(simdict['logM'])


	   ############# EVERYTHING BELOW THIS SHOULD BE ABSTRACTED

	def make_cmd(mags):
		return np.asarray( [mags[:,0] - mags[:,1], mags[:,0]] ).T

	base_weights = 1 / simdf['MassProb'] / simdf['BinProb']

	def get_cmd(nstars, gr_dict, simdict):
		
		weights = base_weights

		keys = ['logM', 'BinQ']

		for key in keys:
			new_prob = gr_dict[key].getpdf(simdf[key])
			weights = new_prob * weights
		
		weights = weights / np.nansum(weights)
		
		idx = np.random.choice(len(simdf), size = nstars, replace=True, p=weights)
		
		sel_df = simdf.iloc[idx]
		
		out_mags = sel_df[['output_mag1', 'output_mag2']].dropna().to_numpy()

		return out_mags

	def sample_cmd(params, model = 'spl'):
		
		if model == 'spl':
			gr_dict = {'logM':set_GR_spl(params['slope']), 'BinQ': set_GR_unif(params['binfrac'])}
		elif model == 'bpl':
			gr_dict = {'logM':set_GR_bpl(params['alow'], params['ahigh'], params['split']),\
					   'BinQ': set_GR_unif(params['binfrac'])}
		elif model == 'ln':
			gr_dict = {'logM':set_GR_ln10full(params['mean'], params['width'], params['transition'], \
					params['slope']), 'BinQ': set_GR_unif(params['binfrac'])}
		else:
			print('Unrecognized model!')
		
		intensity = 10**params['log_intensity']
		nstars = int(stats.poisson.rvs(intensity))
		
		noisymags = get_cmd(nstars, gr_dict, simdict)
		j = noisymags 
		return np.asarray([j[:,0] - j[:,1], j[:,0]]).T

	def sample_norm_cmd(params, model = 'spl'):
		cmd = sample_cmd(params, model)
		if len(cmd) == 0:
			return np.zeros((1000,2))
		return cmd_scaler.transform(cmd)

	def approx_kernel_distance(P, Q, mapping):
		Phi_P = mapping(P).sum(axis=0)
		Phi_Q = mapping(Q).sum(axis=0)
		return np.sqrt(np.sum((Phi_P - Phi_Q)**2))

	def exact_kernel_distance(P, Q, gamma = 1):
		P = P[np.max(np.isfinite(P),1)]
		Q = Q[np.max(np.isfinite(Q),1)]
		PP = np.exp(- gamma*np.sum((P[:, None, :] - P[None, :, :])**2, axis=-1)).sum()
		QQ = np.exp(- gamma*np.sum((Q[:, None, :] - Q[None, :, :])**2, axis=-1)).sum()
		PQ = np.exp(- gamma*np.sum((P[:, None, :] - Q[None, :, :])**2, axis=-1)).sum()
		return np.sqrt(PP + QQ - 2 * PQ)


	   ############# EVERYTHING ABOVE THIS SHOULD BE ABSTRACTED

	mag1 = catalog['dat_mag1']
	mag2 = catalog['dat_mag2']
	flag = catalog['dat_det']
	mag1 = mag1[flag]
	mag2 = mag2[flag]

	observed_cmd = make_cmd(np.vstack((mag1,mag2)).T)

	cmd_scaler = MinMaxScaler()
	cmd_scaler.fit(observed_cmd);
	scaled_observed_cmd = cmd_scaler.transform(observed_cmd)

	obs = dict(data = scaled_observed_cmd)

	dummy_cmd = np.zeros(observed_cmd.shape)

	def cmd_sim_spl(params):
		simulated_cmd = sample_norm_cmd(params, model = 'spl')
		return {'data': simulated_cmd}

	def cmd_sim_bpl(params):
		if params['ahigh'] > params['alow']:
			return {'data': dummy_cmd}
		simulated_cmd = sample_norm_cmd(params, model = 'bpl')
		return {'data': simulated_cmd}

	def cmd_sim_ln(params):
		if params['transition'] < params['mean']:
			return {'data':  dummy_cmd}
		simulated_cmd = sample_norm_cmd(params, model = 'ln')
		return {'data': simulated_cmd}

	R = np.random.uniform(0, 1, (len(observed_cmd),2))
	Phi_approx = Nystroem(kernel = 'rbf', n_components=50, gamma = 200) 
	Phi_approx.fit(R)

	def distance(cmd1, cmd2):
		if cmd2 is np.nan or cmd1 is np.nan:
			print('nan!')
		else:
			return approx_kernel_distance(cmd1['data'], cmd2['data'], Phi_approx.transform)

	prior_spl = pyabc.Distribution(slope = pyabc.RV("uniform", -4, 4),
							binfrac = pyabc.RV("uniform", 0, 1),
							  log_intensity = pyabc.RV("uniform", 2, 3))

	prior_bpl = pyabc.Distribution(alow = pyabc.RV("uniform", -2, 2), ahigh = pyabc.RV("uniform", -3, 2),
							split = pyabc.RV("uniform", 0.2, 0.8), binfrac = pyabc.RV("uniform", 0, 1),
							  log_intensity = pyabc.RV("uniform", 2, 3))

	prior_ln = pyabc.Distribution(mean = pyabc.RV("uniform", 0.1, 0.7,), \
			width = pyabc.RV("uniform", 0, 1),  slope = pyabc.RV("uniform", -3, 2),
					transition = pyabc.RV("uniform", 0.8, 0.4), binfrac = pyabc.RV("uniform", 0, 1),
					log_intensity = pyabc.RV("uniform", 2, 3))

	if imf_type is 'spl':
		simulator = cmd_sim_spl
		prior = prior_spl
	elif imf_type is 'bpl':
		simulator = cmd_sim_bpl
		prior = prior_bpl
	elif imf_type is 'ln':
		simulator = cmd_sim_ln
		prior = prior_ln

	abc = pyabc.ABCSMC(simulator, prior,\
					   distance, sampler = pyabc.sampler.SingleCoreSampler(),\
					   population_size = pop_size, \
					   eps = pyabc.epsilon.QuantileEpsilon(alpha = 0.5))

	db_path = ("sqlite:///" + savename + ".db")

	abc.new(db_path, {'data': obs['data']});

	history = abc.run(min_acceptance_rate = 0.001, max_nr_populations = max_n_pop)

	return history
