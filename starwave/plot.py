from getdist import plots, MCSamples

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

def plot_lfs(cmd_list):
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