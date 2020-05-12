from getdist import plots, MCSamples

def cornerplot(sampledf, weights = None, labels = None, markers = None, ranges = None):
	names = sampledf.columns
	samples = MCSamples(samples = sampledf.to_numpy(), names = names, weights = weights, labels = labels, ranges = ranges)
	#samples.updateSettings({'contours': [0.68, 0.95, 0.99]})
	g = plots.get_subplot_plotter()
	return g.triangle_plot(samples, filled = True, markers = markers)