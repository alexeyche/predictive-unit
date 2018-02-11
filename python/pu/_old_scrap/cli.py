#!/usr/bin/env python

import sys
import os

import argparse
import numpy as np

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.realpath("{}/../".format(script_dir)))

# script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, "predictive-unit/protos")

from pu.pu_common import *

def start_sim(args):
    start_sim = messages.TStartSim()
    start_sim.SimulationTime = args.duration

    resp = send_message(start_sim, args.site, args.port)
    
    print RESPONSE_NUM_TO_NAME.get(resp.ResponseType)
    print resp

   
def send_data(args):
    resp = send_message(form_input_data(), args.site, args.port)
    
    print RESPONSE_NUM_TO_NAME.get(resp.ResponseType)
    print resp.Message


def dump_data(args):
    d = form_input_data()
    mtype = MESSAGE_NAME_TO_NUM["INPUT_DATA"]
    mpck = pack_message(mtype, d)    
    print "Dumping message type {} and size {}b".format(mtype, d.ByteSize()) 
    with open(args.dst, "wb") as fptr:
        fptr.write(mpck)

def get_stats(args):
    stat_req = messages.TStatRequest()

    resp = send_message(stat_req, args.site, args.port)

    print RESPONSE_NUM_TO_NAME.get(resp.ResponseType)
    print resp.Message
    print resp.Stats    

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
    
    dump_data_p = subparsers.add_parser("dump_data", help="command_a help")
    dump_data_p.set_defaults(func=dump_data)
    dump_data_p.add_argument("-d", "--dst", help="Destinatin file", default="data.bin")
    
    get_stats_p = subparsers.add_parser("get_stats", help="command_a help")
    get_stats_p.set_defaults(func=get_stats)

    args = parser.parse_args(argv)

    args.func(args)

if __name__ == "__main__":
    main(sys.argv[1:])
