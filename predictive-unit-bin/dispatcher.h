#pragma once

#include "simulator.h"
#include "defaults.h"
#include "protocol.h"

#include <sys/socket.h>

#include <iostream>
#include <type_traits>

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

	class TDispatcherConnection: public  Poco::Net::TCPServerConnection {
	public:
		TDispatcherConnection(const Poco::Net::StreamSocket& s, TSimulator& sim)
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
			  			
							if (Sim.StartSimulationAsync(startSim)) {
								response.set_responsetype(NPredUnitPb::TServerResponse::OK);
							} else {
								response.set_responsetype(NPredUnitPb::TServerResponse::BUSY);
								response.set_message("Simulation is busy");
							}
						}
						break;
					case NPredUnitPb::TMessageType::INPUT_DATA:
						{
							if (Sim.IsSumulationRunning()) {
								TInputData inp = ReadProtobufMessageFromSocket<TInputData>(sck, header.MessageSize);
								L_INFO << "Got input data " << inp.Data.rows() << "x" << inp.Data.cols() << ", ingesting ...";
								Sim.IngestData(inp.Data);
								response.set_responsetype(NPredUnitPb::TServerResponse::OK);
							} else {
								response.set_responsetype(NPredUnitPb::TServerResponse::SIM_NOT_RUN);
								response.set_message("Simulation is not running");	
							}	
						}
						break;
					case NPredUnitPb::TMessageType::SERVER_RESPONSE:
						{
							response.set_responsetype(NPredUnitPb::TServerResponse::ERROR);
							response.set_message("Server to server comms");
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


	class TDispatcherConnectionFactory: public Poco::Net::TCPServerConnectionFactory
	{
	public:
		TDispatcherConnectionFactory(TSimulator& sim)
			: Sim(sim) 
		{
		}
		
		Poco::Net::TCPServerConnection* createConnection(const Poco::Net::StreamSocket& socket)
		{
			return new TDispatcherConnection(socket, Sim);
		}
	
	private:
		TSimulator& Sim;
	};

	class TDispatcher {
	public:
		TDispatcher(TSimulator& sim, ui32 port)
			: Port(port)
			, Socket(Port)
			, Server(new TDispatcherConnectionFactory(sim), Socket)

		{
		}
		
		void Run() {
			Server.start();
	
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