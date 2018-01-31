#pragma once

#include "simulator.h"

#include <predictive-unit/base.h>
#include <predictive-unit/log.h>

#include <Poco/Net/TCPServer.h>
#include <Poco/Net/TCPServerConnection.h>
#include <Poco/Net/TCPServerConnectionFactory.h>
#include <Poco/Net/TCPServerParams.h>
#include <Poco/Net/StreamSocket.h>
#include <Poco/Net/ServerSocket.h>
#include <Poco/Exception.h>

#include <iostream>
#include <type_traits>

#include <predictive-unit/protos/messages.pb.h>

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
			L_INFO << "Got connection from " << socket().peerAddress().toString();
			
			char buf[sizeof(NPb::uint32)];
			socket().receiveBytes(&buf, sizeof(NPb::uint32));

			NPb::uint32 messageType;
  			NPb::io::ArrayInputStream ais(buf, sizeof(NPb::uint32));
  			NPbIO::CodedInputStream codedInput(&ais);

  			codedInput.ReadVarint32(&messageType);
			
			L_INFO << "Got message type " << messageType; 

			if (messageType == NPredUnitPb::TMessageType::START_SIM) {
				L_INFO << "asd ";
			}

			// MessageType
			// NPredUnitPb::TMessageType::EMessageType messageType;

			// socket().receiveBytes(&messageType, sizeof(messageType));
        	
			try {
				TString ss("asd");
				socket().sendBytes(ss.c_str(), ss.size());
			
			} catch (Poco::Exception& exc) {
				L_ERROR << "Exception " << exc.message();
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