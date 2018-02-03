#pragma once

#include <predictive-unit/base.h>
#include <predictive-unit/log.h>
#include <predictive-unit/protos/messages.pb.h>

#include <Poco/Net/StreamSocket.h>
#include <Poco/Net/SocketAddress.h>

#include <google/protobuf/message.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>


namespace NPb = google::protobuf;
namespace NPbIO = google::protobuf::io;


namespace NPredUnit {

	class TClient {
	public:
		TClient(TString serverHost, ui32 serverPort)
			: ServerHost(serverHost) 
			, ServerPort(serverPort)
			, SocketAddr(ServerHost, ServerPort)
		{
		}

		void SendData() {
			L_DEBUG << "Sending data to " << ServerHost << ":" << ServerPort;

			Socket.connect(SocketAddr);
			{
				NPredUnitPb::TStartSim startSimMessage;
				startSimMessage.set_simulationtime(10000000);

				NPb::uint32 messageType = NPredUnitPb::TMessageType::START_SIM;
				NPb::uint32 messageSize = startSimMessage.ByteSize();

				ui32 bytesToSend = sizeof(NPb::uint32) + sizeof(NPb::uint32) + messageSize;
				char* buf = new char[bytesToSend];

				{
					NPb::io::ArrayOutputStream aos(buf, bytesToSend);
		  			NPbIO::CodedOutputStream codedOutput(&aos);

		  			codedOutput.WriteLittleEndian32(messageType);
		  			codedOutput.WriteLittleEndian32(messageSize);
		  			startSimMessage.SerializeToCodedStream(&codedOutput);
				}
				
	  			ui32 totatSent = 0;
	  			while (totatSent < bytesToSend) {
	  				int nSent = Socket.sendBytes(buf, bytesToSend);
	  				totatSent += nSent;
	  			}

	  			L_INFO << "Sent " << totatSent << " out of " << bytesToSend;
				for (ui32 bi=bytesToSend-startSimMessage.ByteSize(); bi < bytesToSend; ++bi) {
					L_INFO << (int)buf[bi];
				}

				L_INFO << startSimMessage.DebugString();
				
	  			delete[] buf;
			}

			Socket.shutdown();
		}

	private:
		TString ServerHost;
		ui32 ServerPort;

		Poco::Net::SocketAddress SocketAddr;
		Poco::Net::StreamSocket Socket;

	};




}

