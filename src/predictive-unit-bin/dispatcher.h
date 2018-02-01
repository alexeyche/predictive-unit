#pragma once

#include "simulator.h"
#include "defaults.h"

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

#include <google/protobuf/message.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>


namespace NPb = google::protobuf;
namespace NPbIO = google::protobuf::io;


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

			L_INFO << "Got connection from " << sck.peerAddress().toString();
			
			ui32 headerSize = sizeof(NPb::uint32) + sizeof(NPb::uint32);
			char buf[headerSize];
			
			ui32 nRec = 0;
			while (nRec < headerSize) {
				nRec += sck.receiveBytes(buf + nRec, headerSize, MSG_PEEK);
			}
			
			L_INFO << "Recieved " << nRec << " bytes";

			NPb::uint32 messageType;
			NPb::uint32 messageSize;
			{
	  			NPb::io::ArrayInputStream ais(buf, sizeof(NPb::uint32) + sizeof(NPb::uint32));
	  			NPbIO::CodedInputStream codedInput(&ais);

	  			bool succ0 = codedInput.ReadVarint32(&messageType);
	  			ENSURE(succ0, "Failed to read message type");
	  			bool succ1 = codedInput.ReadVarint32(&messageSize);
	  			ENSURE(succ1, "Failed to read message size");
			}
			
			L_INFO << "Got message type " << messageType << " of size " << messageSize << "b"; 

			if (messageType == NPredUnitPb::TMessageType::START_SIM) {
				char* messageBytes = new char[headerSize + messageSize];
				
				ui32 totalReceived = 0;
				while (totalReceived < (messageSize + headerSize)) {
					totalReceived += sck.receiveBytes(messageBytes + totalReceived, TDefaults::ServerSocketBuffSize);
					L_INFO << "R:" << totalReceived;
				}

	  			for (ui32 bi=0; bi < messageSize; ++bi) {
					L_INFO << (int)messageBytes[bi];	
				}

				NPredUnitPb::TStartSim startSimMessage;
				// {
				// 	bool succ = startSimMessage.ParseFromArray(messageBytes, messageSize);
		  // 			if (!succ) {
		  // 				L_ERROR << "Failed to read protobuf 0";	
		  // 			}
				// }
				{
					NPb::io::ArrayInputStream aisM(messageBytes, headerSize + messageSize);
		  			NPbIO::CodedInputStream codedInputM(&aisM);
	
		  			bool succ0 = codedInputM.ReadVarint32(&messageType);
		  			ENSURE(succ0, "Failed to read message type");
		  			bool succ1 = codedInputM.ReadVarint32(&messageSize);
		  			ENSURE(succ1, "Failed to read message size");

		  			bool succ = startSimMessage.ParseFromCodedStream(&codedInputM);
		  			if (!succ) {
		  				L_ERROR << "Failed to read protobuf 1";	
		  			}
				}
	  			
	  			L_INFO << "Got message" << startSimMessage.simulationtime() << " - " << startSimMessage.DebugString();
	  			
	  			delete[] messageBytes;

	  			Sim.StartSimulation(startSimMessage);
			}

			// try {
			// 	TString ss("asd");
			// 	socket().sendBytes(ss.c_str(), ss.size());
			
			// } catch (Poco::Exception& exc) {
			// 	L_ERROR << "Exception " << exc.message();
			// }
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