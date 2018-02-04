#!/usr/bin/env python

import sys
sys.path.insert(0, "/var/tmp/predictive-unit-protos")
from pu.pu_common import *
import numpy as np

from util import *


stat_req = messages.TStatRequest()

resp = send_message(stat_req, "localhost", 8080)

print resp.Message
layer_size = 10

def l_reshape(a):
	return a.reshape((a.shape[0], a.shape[1]/layer_size, layer_size))


mem = l_reshape(proto_to_np(resp.Stats.Membrane))


act = l_reshape(proto_to_np(resp.Stats.Activation))


F = proto_to_np(resp.Stats.F)
Fc = proto_to_np(resp.Stats.Fc)



