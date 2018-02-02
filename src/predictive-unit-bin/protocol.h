#pragma once

#include "defaults.h"

#include <predictive-unit/base.h>

#include <Poco/Net/StreamSocket.h>

#include <google/protobuf/message.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>

namespace NPb = google::protobuf;
namespace NPbIO = google::protobuf::io;


namespace NPredUnit {

	// using TMap<NPredUnitPb::MessageType, std::function<void(const NPb::message&)>>;

	struct THeader {
		NPb::uint32 MessageType;
		NPb::uint32 MessageSize;		
	};


	THeader ReadMessageHeaderFromSocket(Poco::Net::StreamSocket& sck);

	THeader ReadMessageHeaderFromStream(std::istream& in);

	template <typename T>
	void ReadProtobufMessageFromSocket(Poco::Net::StreamSocket& sck, ui32 messageSize, T* dstMessage) {
		char* messageBytes = new char[messageSize];
		
		ui32 totalReceived = 0;
		while (totalReceived < messageSize) {
			totalReceived += sck.receiveBytes(messageBytes + totalReceived, TDefaults::ServerSocketBuffSize);
		}

		NPb::io::ArrayInputStream ais(messageBytes, messageSize);
		NPbIO::CodedInputStream codedInput(&ais);

		bool readSuccess = dstMessage->ParseFromCodedStream(&codedInput);
		delete[] messageBytes;
		ENSURE(readSuccess, "Failed to read protobuf");
	}


	void ReadProtobufMessage(std::istream& in);

}