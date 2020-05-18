from getdist import plots, MCSamples
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

def cornerplot(sampledf, weights = None, labels = None, markers = None, ranges = None):
	names = sampledf.columns
	samples = MCSamples(samples = sampledf.to_numpy(), names = names, weights = weights, labels = labels, ranges = ranges)
	#samples.updateSettings({'contours': [0.68, 0.95, 0.99]})
	g = plots.get_subplot_plotter()
	return g.triangle_plot(samples, filled = True, markers = markers)

def plot_lf(cmd):
	cmd_samples = MCSamples(samples = cmd, names = ['color', 'mag'])
	g = plots.get_single_plotter(width_inch = 2*3.464, ratio = 0.75)
	g.plot_1d(cmd_samples, 'mag')

def plot_lfs_kde(cmd_list):
	g = plots.get_single_plotter(width_inch = 2*3.464, ratio = 0.75)
	colors = ['k']
	colors.extend(['r']*(len(cmd_list) - 1))
	ct = 0
	for cmd in cmd_list:
		cmd_samples = MCSamples(samples = cmd, names = ['color', 'mag'])
		if ct > 0:
			g.settings.linewidth = 0.5
		g.plot_1d(cmd_samples, 'mag', colors = [colors[ct]])
		ct+=1

def plot_lfs(cmd_list, n_bins = 35, axis = 1):
	f = plt.figure(figsize = (8,5))
	cmd = cmd_list[0]
	mag = cmd[:, axis]
	bins = stats.binned_statistic(mag, mag, statistic = 'count', bins = n_bins)
	lf = bins.statistic
	binedges = bins.bin_edges
	mags = bins.bin_edges[:-1] + ((bins.bin_edges[1] - bins.bin_edges[0])/2)

	plt.plot(mags, lf, 'k')
	model_lfs = [];
	for cmd in cmd_list[1:]:
		mag = cmd[:, axis]
		bins = stats.binned_statistic(mag, mag, statistic = 'count', bins = binedges)
		lf = bins.statistic
		model_lfs.append(lf)

	model_lfs = np.asarray(model_lfs)
	std_lf = np.std(model_lfs, 0)
	mean_lf = np.mean(model_lfs, 0)

	plt.plot(mags, mean_lf, 'orange', linewidth = 2)
	plt.fill_between(mags, mean_lf - std_lf/2, mean_lf + std_lf/2, color = 'orange', alpha = 0.5)