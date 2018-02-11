

import sys
from dataset import XorDataset, to_sparse_ts
import re
import numpy as np

import struct
import socket


import sys
sys.path.insert(0, "/var/tmp/predictive-unit-protos")


def to_snake_case(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

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
        r = m_p.Row.add()
        r.Data.extend(m[i, :])
    return m_p
        
def proto_to_np(m):
    rows = len(m.Row)
    if rows == 0:
        return np.zeros((0,0))
    
    cols = len(m.Row[0].Data)
    res = np.zeros((rows, cols))
    for r_id, r in enumerate(m.Row):
        res[r_id, :] = r.Data[:]
    return res

def read_message(sck):
    data = sck.recv(8)
    if len(data) == 0:
        raise Exception("Failed to read data from socket")
      
    mtype, = struct.unpack_from("<L", data)
    msize, = struct.unpack_from("<L", data, offset=4)
 
    mname = MESSAGE_NUM_TO_NAME.get(mtype)

    if mname == "SERVER_RESPONSE":
        mdata = sck.recv(msize)

        resp = messages.TServerResponse()
        resp.ParseFromString(mdata)
    else:
        raise NotImplementedError("Failed to handle response: {}".format(mname))
 
    sck.close()
    return resp

def pack_message(mtype, m):
    return (
        struct.pack("<L", mtype) + 
        struct.pack("<L", m.ByteSize()) + 
        m.SerializeToString()
    ) 

def send_message(m, site, port):
    sck = socket.socket()
    sck.connect((site, port))

    mname = "_".join(to_snake_case(m.DESCRIPTOR.name).split("_")[1:]).upper()
   
    mtype = MESSAGE_NAME_TO_NUM.get(mname)
    if mtype is None:
        raise Exception("Failed to recognize message type: {}".format(m.DESCRIPTOR.name))

    mpck = pack_message(mtype, m)
   
    bytes_sent = sck.send(mpck)
    if bytes_sent == 0:
        raise Exception("Failed to sent data to {}:{}".format(site, port))
    return read_message(sck)


# do = np.asarray([[
#     range(0, 10),
#     range(10, 20)
# ]])

# from util import *

# d = do.transpose((0, 2, 1))

# dd = d.reshape((1, 2*10))
# for t in xrange(0, 20, 2):
#     print dd[0:1, t:(t + 2)]

# ds = XorDataset()
# x_v, y_v = ds.next_train_batch()
# num_iters = 100
# x_v_ts = to_sparse_ts(x_v, num_iters, at=10)

# data = (
#     np.transpose(x_v_ts, (1, 0, 2))
#         .reshape((
#             x_v_ts.shape[1], 
#             x_v_ts.shape[0]*x_v_ts.shape[2]
#         ))
# )

# dddd = []

# for t in xrange(0, 200, 2):
#     print data[:, t:(t + 2)]
    # dddd.append(data[:, t:(t + 2)])

def form_input_data():
    inp_data = messages.TInputData()
    
    ds = XorDataset()
    x_v, y_v = ds.next_train_batch()
    num_iters = 100
    x_v_ts = to_sparse_ts(x_v, num_iters, at=10)
    
    data = (
        np.transpose(x_v_ts, (1, 0, 2))
            .reshape((
                x_v_ts.shape[1], 
                x_v_ts.shape[0]*x_v_ts.shape[2]
            ))
    )
    
    np_to_proto(data, inp_data.Data)
    return inp_data
    
