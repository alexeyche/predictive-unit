
import struct
import socket

from pu.config import *
from io import BytesIO

lc0 = LayerConfig(
	layerType = 0 ,
	networkSize = 100,
	inputSize = 4,
	batchSize = 4,
	filterSize = 1,
	bufferSize = 100,
	tau = 5.0,
	tauMean = 100.0 ,
	adaptGain = 1.0,
	learningRate = 0.1,
)

lc1 = LayerConfig(
	layerType = 0,
	networkSize = 2,
	inputSize = 100,
	batchSize = 4,
	filterSize = 1,
	bufferSize = 100,
	tau = 5.0,
	tauMean = 100.0 ,
	adaptGain = 1.0,
	learningRate = 0.99,
)

nc = NetworkConfig(layerConfigs = (lc0, lc1))



port = 8080
site = "localhost"

out = BytesIO()
nc.serialize(out)


sck = socket.socket()
sck.connect((site, port))


bytes_sent = sck.send(out.getvalue())
if bytes_sent == 0:
    raise Exception("Failed to sent data to {}:{}".format(site, port))
print bytes_sent