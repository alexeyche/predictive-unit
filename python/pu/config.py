
from pu.io import *


class NetworkConfig(object):
	def __init__(self, **kwargs):
		buff = kwargs.get("buff")
		if buff:
			self.deserialize(buff)
		else:
			self.__dict__.update(kwargs)

	def serialize(self, buff):
		writeVarUInt(len(self.layerConfigs), buff)
		for lc in self.layerConfigs:
			lc.serialize(buff)

	def deserialize(self, buff):
		size = readVarUInt(buff)
		for i in xrange(size):
			self.layerConfigs.append(
				LayerConfig(buff=buff)
			)
			
	def __str__(self):
		return """NetworkConfig(
			{}
		)""".format("\n".join([str(lc) for lc in self.layerConfigs]))

class LayerConfig(object):
	def __init__(self, **kwargs):
		buff = kwargs.get("buff")
		if buff:
			self.deserialize(buff)
		else:
			self.__dict__.update(kwargs)

	def __str__(self):
		return """LayerConfig(
			layerType = {layerType}
			networkSize = {networkSize}
			inputSize = {inputSize}
			batchSize = {batchSize}
			filterSize = {filterSize}
			bufferSize = {bufferSize}
			tau = {tau}
			tauMean = {tauMean}
			adaptGain = {adaptGain}
			learningRate = {learningRate}
		)""".format(**self.__dict__)

	def deserialize(self, buff):
		self.layerType = readVarUInt(buff)
		self.networkSize = readVarUInt(buff)
		self.inputSize = readVarUInt(buff)
		self.batchSize = readVarUInt(buff)
		self.filterSize = readVarUInt(buff)
		self.bufferSize = readVarUInt(buff)

		self.tau = readDouble(buff)
		self.tauMean = readDouble(buff)
		self.adaptGain = readDouble(buff)
		self.learningRate = readDouble(buff)

	def serialize(self, buff):
		writeVarUInt(self.layerType, buff)
		writeVarUInt(self.networkSize, buff)
		writeVarUInt(self.inputSize, buff)
		writeVarUInt(self.batchSize, buff)
		writeVarUInt(self.filterSize, buff)
		writeVarUInt(self.bufferSize, buff)

		writeDouble(self.tau, buff)
		writeDouble(self.tauMean, buff)
		writeDouble(self.adaptGain, buff)
		writeDouble(self.learningRate, buff)



