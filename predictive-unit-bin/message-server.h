#pragma once

#include "network.h"

#include <sys/socket.h>

#include <iostream>
#include <type_traits>


#include <predictive-unit/base.h>
#include <predictive-unit/log.h>
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
		TMessageServerConnection(const Poco::Net::StreamSocket& s)
			: Poco::Net::TCPServerConnection(s)
		{
		}
		
		void run() {
			Poco::Net::StreamSocket& sck = socket();

			try {
				L_INFO << "Got connection from " << sck.peerAddress().toString();		
				
				char rawBuffer[100];
				sck.receiveBytes(&rawBuffer[0], 100);
				
				TMemBuf buffer(&rawBuffer[0], 100);
				TInputStream io(&buffer);
				
				TNetworkConfig config;
				
				config.Serial(IOStream(io));

				config.Serial(NamedLogStream("NetworkConfig"));

		
			} catch (Poco::Exception& err) {
				L_ERROR << "Got POCO error: " << err.message();
			} catch (TErrException& err) {
				L_ERROR << "Got error: " << err.what();
			}
		}	
	};


	class TMessageServerConnectionFactory: public Poco::Net::TCPServerConnectionFactory
	{
	public:
		TMessageServerConnectionFactory()
		{
		}
		
		Poco::Net::TCPServerConnection* createConnection(const Poco::Net::StreamSocket& socket)
		{
			return new TMessageServerConnection(socket);
		}
	
	};

	class TMessageServer {
	public:
		TMessageServer(ui32 port)
			: Port(port)
			, Socket(Port)
			, Server(new TMessageServerConnectionFactory(), Socket)

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