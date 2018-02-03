#pragma once

#include "defaults.h"

#include <predictive-unit/base.h>
#include <predictive-unit/protos/messages.pb.h>

#include <Poco/Net/StreamSocket.h>

#include <google/protobuf/message.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>

namespace NPb = google::protobuf;
namespace NPbIO = google::protobuf::io;


namespace NPredUnit {

	// using TMap<NPredUnitPb::MessageType, std::function<void(const NPb::Message&)>>;

	struct THeader {
		NPb::uint32 MessageType;
		NPb::uint32 MessageSize;		
	};



	THeader ReadMessageHeaderFromStream(std::istream& in);

	THeader ReadMessageHeaderFromSocket(Poco::Net::StreamSocket& sck);

	void ReadProtobufMessageFromSocket(Poco::Net::StreamSocket& sck, ui32 messageSize, NPb::Message* dstMessage);

	void WriteHeaderAndProtobufMessageToSocket(const NPb::Message& src, NPredUnitPb::TMessageType::EMessageType messageType, Poco::Net::StreamSocket* sck);

	void ReadProtobufMessage(std::istream& in);

}