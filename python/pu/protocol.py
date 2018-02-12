
from collections import OrderedDict

import io
import numpy as np

from varint import decode_stream as readVarUInt
from varint import encode as encode_varint
import struct

def readDouble(buf):
	return struct.unpack("d", buf.read(8))[0]

def writeDouble(v, buf):
	buf.write(struct.pack("d", v))


def writeVarUInt(v, buf):
	b = encode_varint(v)
	buf.write(b)


class Reader(object):
	def __init__(self, v):
		self.buff = io.BytesIO(v)

	def read(self, n):
		return self.buff.read(n)

	def get(self):
		return self.buff.getvalue()

class Writer(object):
	def __init__(self):
		self.buff = io.BytesIO()

	def write(self, v):
		return self.buff.write(v)

	def get(self):
		return self.buff.getvalue()

class ProtocolObject(object):
	def __init__(self, **params):
		self._params = self.SCHEMA.copy() 
		for k, v in params.iteritems():
			if k != "buff":
				if not k in self._params: 		
					raise Exception("Unknown parameter: {}".format(k))
				self._params[k] = v

		if params.get("buff"):
			self.serial(params["buff"])

	def copy(self):
		v = self.__new__(self.__class__)
		v.__init__(**self._params)
		return v


	def serial(self, buff):
		def write_instance(v):
			if type(v) is float or type(v) is np.float64:
				writeDouble(v, buff)				
			elif type(v) is int:
				writeVarUInt(v, buff)
			elif isinstance(v, ProtocolObject):
				v.serial(buff)
			elif type(v) is tuple or type(v) is list:
				writeVarUInt(len(v), buff)
				for vv in v:
					write_instance(vv)
			elif type(v) is np.ndarray:
				assert len(v.shape) == 2
				writeVarUInt(v.shape[0], buff)
				writeVarUInt(v.shape[1], buff)					
				for row in v:
					for vv in row:
						writeDouble(vv, buff)
			else:
				raise Exception("Unknown type {} with value {}".format(type(v), v))

		def read_instance(v):
			if type(v) is float:
				return readDouble(buff)
			elif type(v) is int:
				return readVarUInt(buff)
			elif isinstance(v, ProtocolObject):
				return v.serial(buff)
			elif type(v) is tuple or type(v) is list:
				size = readVarUInt(buff)
				vc = v[0].copy()
				return tuple([
					read_instance(vvc)
					for vvc in [vc for _ in xrange(size)]
				])
			
			elif type(v) is np.ndarray:
				rows = readVarUInt(buff)
				cols = readVarUInt(buff)
				print rows, cols
				data = np.zeros((rows, cols), dtype=np.float64)
				for i in xrange(cols):
					for j in xrange(rows):
						data[i, j] = readDouble(buff)
				return data
			else:
				raise Exception("Unknown type {} with value {}".format(type(v), v))

		if isinstance(buff, Reader):
			new_vals = []
			for k, v in self._params.iteritems():
				new_v = read_instance(v)
				if not new_v is None:
					new_vals.append((k, new_v))
			for k, new_v in new_vals:
				self._params[k] = new_v

		elif isinstance(buff, Writer):
			for k, v in self._params.iteritems():
				write_instance(v)
		else:
			raise Exception("Unknown type of buffer: {}".format(buff))

	def __str__(self):
		return "{}\n{}\n".format(
			type(self).__name__, 
			"\n".join(["\t{} - {}".format(k, v) for k, v in self._params.iteritems()]),
		)



class LayerConfig(ProtocolObject):
	SCHEMA = OrderedDict([
		("LayerType", 0) ,
		("LayerSize", 100),
		("InputSize", 4),
		("BatchSize", 4),
		("FilterSize", 1),
		("BufferSize", 100),
		("Tau", 5.0),
		("TauMean", 100.0),
		("AdaptGain", 1.0),
		("LearningRate", 0.1),
	])



class NetworkConfig(ProtocolObject):
	SCHEMA = OrderedDict([
		("LayerConfigs", [LayerConfig()]),
	])

class MessageType(ProtocolObject):
	SCHEMA = OrderedDict([
		("MessageType", 0)
	])

class StartSim(ProtocolObject):
	SCHEMA = OrderedDict([
		("NetworkConfig", NetworkConfig()),
		("Data", np.matrix(((0,0)), np.float64))
	])


