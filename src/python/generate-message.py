#!/usr/bin/env python

import sys
import os
import struct

# script_dir = os.path.dirname(os.path.realpath(__file__))
# sys.path.insert(0, "{}/../predictive-unit/protos".format(script_dir))

sys.path.insert(0, "predictive-unit/protos")

import messages_pb2 as messages



start_sim = messages.TStartSim()
start_sim.SimulationTime = 1001

m = (
	struct.pack('<L', 0) + 
	struct.pack('<L', start_sim.ByteSize()) + 
	start_sim.SerializeToString()
)

print m,