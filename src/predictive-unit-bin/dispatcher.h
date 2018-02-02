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

#include <Poco/Net/TCPServer.h>
#include <Poco/Net/TCPServerConnection.h>
#include <Poco/Net/TCPServerConnectionFactory.h>
#include <Poco/Net/TCPServerParams.h>
#include <Poco/Net/StreamSocket.h>
#include <Poco/Net/ServerSocket.h>
#include <Poco/Exception.h>



namespace NPredUnit {

	class TDispatcherConnection: public  Poco::Net::TCPServerConnection {
	public:
		TDispatcherConnection(const Poco::Net::StreamSocket& s, TSimulator& sim)
			: Poco::Net::TCPServerConnection(s)
			, Sim(sim)
		{
		}
		
		void run() {
			Poco::Net::StreamSocket& sck = socket();
			try {
				L_INFO << "Got connection from " << sck.peerAddress().toString();		

				THeader header = ReadMessageHeaderFromSocket(sck);
				L_INFO << "Got message type " << header.MessageType << " of size " << header.MessageSize << "b"; 

				switch (header.MessageType) {
					case NPredUnitPb::TMessageType::START_SIM:
						
						NPredUnitPb::TStartSim startSimMessage;
						ReadProtobufMessageFromSocket(sck, header.MessageSize, &startSimMessage);

		  				L_INFO << "Got message" << startSimMessage.simulationtime() << " - " << startSimMessage.DebugString();
		  			
						// start sim
						Sim.StartSimulation(startSimMessage);
						break;
				} 

			} catch (Poco::Exception& err) {
				L_ERROR << "Got POCO error: " << err.message();
			} catch (TErrException& err) {
				L_ERROR << "Got error: " << err.what();
			}

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
		TDispatcher(ui32 port)
			: Port(port)
			, Socket(Port)
			, Server(new TDispatcherConnectionFactory(Sim), Socket)

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
		TSimulator Sim;
	
		ui32 Port;
		Poco::Net::ServerSocket Socket;
		Poco::Net::TCPServer Server;
		Poco::Event Terminate;
	};



}