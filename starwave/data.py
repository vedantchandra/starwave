import pandas

class MagData:
	def __init__(self, magnitudes, names):

		self.mags = magnitudes
		self.names = names
		self.data = pandas.DataFrame(self.mags, columns = self.names);

	def to_cmd(band1, band2):
		mag = self.data[band1]
		color = self.data[band1] - self.data[band2]
		return np.asarray([color,mag]).T