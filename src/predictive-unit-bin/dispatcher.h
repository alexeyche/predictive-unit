#pragma once

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

namespace NPredUnit {


	class TDispatcherConnection: public  Poco::Net::TCPServerConnection {
	public:
		TDispatcherConnection(const Poco::Net::StreamSocket& s):
			Poco::Net::TCPServerConnection(s)
		{
		}
		
		void run() {
			L_INFO << "Got connection from " << socket().peerAddress().toString();
			
			try {
				TString ss("asd");
				socket().sendBytes(ss.c_str(), ss.size());
			
			} catch (Poco::Exception& exc) {
				L_ERROR << "Exception " << exc.message();
			}
		}
	};


	class TDispatcherConnectionFactory: public Poco::Net::TCPServerConnectionFactory
	{
	public:
		TDispatcherConnectionFactory() {
		}
		
		Poco::Net::TCPServerConnection* createConnection(const Poco::Net::StreamSocket& socket)
		{
			return new TDispatcherConnection(socket);
		}
	};

	class TDispatcher {
	public:
		TDispatcher(ui32 port)
			: Port(port)
			, Socket(Port)
			, Server(new TDispatcherConnectionFactory(), Socket)

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