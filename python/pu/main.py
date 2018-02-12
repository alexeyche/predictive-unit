
import struct
import socket
import numpy as np

from pu.protocol import *
from collections import OrderedDict 
from StringIO import StringIO


ss = StartSim(
	NetworkConfig = NetworkConfig(
		LayerConfigs = (
			LayerConfig(
				LayerType = 0,
				LayerSize = 100,
				InputSize = 4,
				BatchSize = 4
			),
			LayerConfig(
				LayerType = 1,
				LayerSize = 2,
				InputSize = 100,
				BatchSize = 4
			)
		)
	),
	Data = np.random.random((10, 10))
)



bytes_str = serial_message(ss)


port = 8080
site = "localhost"

sck = socket.socket()
sck.connect((site, port))


bytes_sent = sck.send(bytes_str)
if bytes_sent == 0:
    raise Exception("Failed to sent data to {}:{}".format(site, port))
print bytes_sent