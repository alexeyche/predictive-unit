#pragma once

#include "simulator.h"

#include <sys/socket.h>

#include <iostream>
#include <type_traits>


#include <predictive-unit/defaults.h>
#include <predictive-unit/protocol.h>
#include <predictive-unit/base.h>
#include <predictive-unit/log.h>
#include <predictive-unit/protos/messages.pb.h>
#include <predictive-unit/util/string.h>

#include <Poco/Net/TCPServer.h>
#include <Poco/Net/TCPServerConnection.h>
#include <Poco/Net/TCPServerConnectionFactory.h>
#include <Poco/Net/TCPServerParams.h>
#include <Poco/Net/StreamSocket.h>
#include <Poco/Net/ServerSocket.h>
#include <Poco/Exception.h>



namespace NPredUnit {

	using namespace NStr;

	class TMessageServerConnection: public  Poco::Net::TCPServerConnection {
	public:
		TMessageServerConnection(const Poco::Net::StreamSocket& s, TSimulator& sim)
			: Poco::Net::TCPServerConnection(s)
			, Sim(sim)
		{
		}
		
		void run() {
			Poco::Net::StreamSocket& sck = socket();
			NPredUnitPb::TServerResponse response;
			try {
				L_INFO << "Got connection from " << sck.peerAddress().toString();		

				THeader header = ReadMessageHeaderFromSocket(sck);
				L_INFO << "Got message type " << header.MessageType << " of size " << header.MessageSize << "b"; 

				switch (header.MessageType) {
					case NPredUnitPb::TMessageType::START_SIM:
						{
							TStartSim startSim = ReadProtobufMessageFromSocket<TStartSim>(sck, header.MessageSize);
			  			
							if (Sim.StartSimulationAsync(startSim.SimConfig)) {
								response.set_responsetype(NPredUnitPb::TServerResponse::OK);
							} else {
								response.set_responsetype(NPredUnitPb::TServerResponse::BUSY);
								response.set_message("Simulation is busy");
							}
						}
						break;
					case NPredUnitPb::TMessageType::INPUT_DATA:
						{
							TInputData inp = ReadProtobufMessageFromSocket<TInputData>(sck, header.MessageSize);
							if (Sim.IsSumulationRunning(inp.SimId)) {
								L_INFO << "Got input data for sim #" << inp.SimId << ", " << inp.Data.rows() << "x" << inp.Data.cols() << ", ingesting ...";
								Sim.IngestDataAsync(inp);
								response.set_responsetype(NPredUnitPb::TServerResponse::OK);
							} else {
								L_INFO << "Sim #" << inp.SimId << " is not running; no data ingest";
								response.set_responsetype(NPredUnitPb::TServerResponse::SIM_NOT_RUN);
								response.set_message(TStringBuilder() << "Simulation " << inp.SimId << " is not running");	
							}	
						}
						break;
					case NPredUnitPb::TMessageType::SERVER_RESPONSE:
						{
							response.set_responsetype(NPredUnitPb::TServerResponse::ERROR);
							response.set_message("Server to server comms");
						}
						break;
					case NPredUnitPb::TMessageType::STAT_REQUEST:
						{
							TStatRequest sr = ReadProtobufMessageFromSocket<TStatRequest>(sck, header.MessageSize);
							auto stats = Sim.GetStats(sr);
							if (stats) {

								response.mutable_stats()->CopyFrom(stats->ToProto());
								response.set_responsetype(NPredUnitPb::TServerResponse::OK);
								L_INFO << response.ByteSize();
							} else {
								response.set_message("Simluation is busy (probably started to colect statistics)");
								response.set_responsetype(NPredUnitPb::TServerResponse::BUSY);
							}
						}
						break;
				} 

			} catch (Poco::Exception& err) {
				L_ERROR << "Got POCO error: " << err.message();
				response.set_responsetype(NPredUnitPb::TServerResponse::ERROR);
				response.set_message(TStringBuilder() << "POCO error: " << err.message());
			} catch (TErrException& err) {
				L_ERROR << "Got error: " << err.what();
				response.set_responsetype(NPredUnitPb::TServerResponse::ERROR);
				response.set_message(TStringBuilder() << "Error: " << err.what());
			}
			
			WriteHeaderAndProtobufMessageToSocket(
				response, 
				NPredUnitPb::TMessageType::SERVER_RESPONSE, 
				&sck
			);
		}

	
	private:
	
		TSimulator& Sim;
	};


	class TMessageServerConnectionFactory: public Poco::Net::TCPServerConnectionFactory
	{
	public:
		TMessageServerConnectionFactory(TSimulator& sim)
			: Sim(sim) 
		{
		}
		
		Poco::Net::TCPServerConnection* createConnection(const Poco::Net::StreamSocket& socket)
		{
			return new TMessageServerConnection(socket, Sim);
		}
	
	private:
		TSimulator& Sim;
	};

	class TMessageServer {
	public:
		TMessageServer(TSimulator& sim, ui32 port)
			: Port(port)
			, Socket(Port)
			, Server(new TMessageServerConnectionFactory(sim), Socket)

		{
		}

		void RunAsync() {
			L_INFO << "Starting server in async mode";
			Server.start();
		}

		void Stop() {
			L_INFO << "Stopping server";
			Server.stop();
		}
		
		void Run() {
			RunAsync();
			Wait();
		}

		void Wait() {
			L_INFO << "Listening port " << Port;
			Terminate.wait();
		
			L_INFO << "Got terminate signal, going down";
			Server.stop();	
		}

	private:
		ui32 Port;
		Poco::Net::ServerSocket Socket;
		Poco::Net::TCPServer Server;
		Poco::Event Terminate;
	};



}