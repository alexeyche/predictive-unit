#!/usr/bin/env python

import sys
import os
import struct
import socket

import argparse
import numpy as np
# script_dir = os.path.dirname(os.path.realpath(__file__))
# sys.path.insert(0, "{}/../predictive-unit/protos".format(script_dir))

sys.path.insert(0, "predictive-unit/protos")

import messages_pb2 as messages
import matrix_pb2 as matrix


MESSAGE_NAME_TO_NUM = dict(zip(
   messages._TMESSAGETYPE_EMESSAGETYPE.values_by_name, 
   messages._TMESSAGETYPE_EMESSAGETYPE.values_by_number
))

MESSAGE_NUM_TO_NAME = dict(zip(
   messages._TMESSAGETYPE_EMESSAGETYPE.values_by_number, 
   messages._TMESSAGETYPE_EMESSAGETYPE.values_by_name
))


RESPONSE_NUM_TO_NAME = dict(zip(
   messages._TSERVERRESPONSE_ERESPONSETYPE.values_by_number, 
   messages._TSERVERRESPONSE_ERESPONSETYPE.values_by_name
))

def np_to_proto(m, m_p):
    start_i, end_i = 0, m.shape[0]
    
    for i in xrange(start_i, end_i):
        r = m_p.Matrix.add()
        r.Data.extend(m[i, :])
    return m_p
        

def read_message(sck):
   data = sck.recv(8)
   mtype, = struct.unpack_from("<L", data)
   msize, = struct.unpack_from("<L", data, offset=4)

   mname = MESSAGE_NUM_TO_NAME.get(mtype)
   if mname == "SERVER_RESPONSE":
      mdata = sck.recv(msize)
      
      resp = messages.TServerResponse()
      resp.ParseFromString(mdata)
   else:
      raise NotImplementedError("Failed to handle respoonse: {}".format(mname))

   sck.close()
   return resp


def send_message(m, site, port):
   sck = socket.socket()
   sck.connect((site, port))

   if isinstance(m, messages.TStartSim):
      mtype = MESSAGE_NAME_TO_NUM.get("START_SIM")
   elif isinstance(m, messages.TInputData):
      mtype = MESSAGE_NAME_TO_NUM.get("INPUT_DATA")
   else:
      raise Exception("Failed to recognize message type")

   m = (
      struct.pack("<L", mtype) + 
      struct.pack("<L", m.ByteSize()) + 
      m.SerializeToString()
   )
   
   sck.send(m)
   return read_message(sck)


def start_sim(args):
   start_sim = messages.TStartSim()
   start_sim.SimulationTime = args.duration

   resp = send_message(start_sim, args.site, args.port)
   
   print RESPONSE_NUM_TO_NAME.get(resp.ResponseType)
   print resp

   
def send_data(args):
    inp_data = messages.TInputData()

    x_v = np.asarray([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ])
    y_v = np.asarray([
        [0.0],
        [1.0],
        [1.0],
        [0.0]
    ])
    
    np_to_proto(x_v, inp_data.Data)
    resp = send_message(inp_data, args.site, args.port)
    
    print RESPONSE_NUM_TO_NAME.get(resp.ResponseType)
    print resp.Message


def main(argv):
    parser = argparse.ArgumentParser(description="Client for predictive unit")
    parser.add_argument("-p", "--port", help="Server port, default: %(default)s", type=int, default=8080)
    parser.add_argument("-s", "--site", help="Server site, default: %(default)s", default="localhost")
    
    subparsers = parser.add_subparsers()

    start_sim_p = subparsers.add_parser("start_sim", help="command_a help")
    start_sim_p.add_argument("-d", "--duration", type=int, default=100, help="Duration of simulation")
    start_sim_p.set_defaults(func=start_sim)

    send_data_p = subparsers.add_parser("send_data", help="command_a help")
    send_data_p.set_defaults(func=send_data)

    args = parser.parse_args(argv)

    args.func(args)

if __name__ == "__main__":
    main(sys.argv[1:])
