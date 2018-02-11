#include "protocol.h"

#include <predictive-unit/log.h>
#include <predictive-unit/protos/messages.pb.h>

namespace NPredUnit {

	namespace {
		
		void ReadHeaderFromCoded(NPbIO::CodedInputStream& codedInput, THeader* header) {
			bool readSuccess = codedInput.ReadLittleEndian32(&header->MessageType);
			readSuccess &= codedInput.ReadLittleEndian32(&header->MessageSize);
		
			ENSURE(readSuccess, "Failed to read message header");
		}

		void WriteHeaderToCoded(const THeader& header, NPbIO::CodedOutputStream* codedOutput) {
			codedOutput->WriteLittleEndian32(header.MessageType);
			codedOutput->WriteLittleEndian32(header.MessageSize);
		}
	}

	THeader ReadMessageHeaderFromSocket(Poco::Net::StreamSocket& sck) {
		char buf[sizeof(THeader)];
		
		ui32 nRec = 0;
		while (nRec < sizeof(THeader)) {
			nRec += sck.receiveBytes(buf + nRec, sizeof(THeader)); //, MSG_PEEK);
		}
				
		NPb::io::ArrayInputStream ais(buf, sizeof(THeader));
		NPbIO::CodedInputStream codedInput(&ais);

		THeader header;
		ReadHeaderFromCoded(codedInput, &header);
		return header;
	}

	void ReadProtobufMessageFromSocket(Poco::Net::StreamSocket& sck, ui32 messageSize, NPb::Message* dstMessage) {
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

	void WriteHeaderAndProtobufMessageToSocket(const NPb::Message& src, NPredUnitPb::TMessageType::EMessageType messageType, Poco::Net::StreamSocket* sck) {
		THeader header;
		header.MessageSize = src.ByteSize();
		header.MessageType = messageType;
		
		L_INFO << header.MessageSize << " " << header.MessageType << " " << sizeof(THeader);

		ui32 bytesToSend = sizeof(THeader) + header.MessageSize;
		char* buf = new char[bytesToSend];
		
		{
			NPb::io::ArrayOutputStream aos(buf, bytesToSend);
			NPbIO::CodedOutputStream codedOutput(&aos);

			WriteHeaderToCoded(header, &codedOutput);
			src.SerializeToCodedStream(&codedOutput);
		}

		ui32 totatSent = 0;
		while (totatSent < bytesToSend) {
			int nSent = sck->sendBytes(buf, bytesToSend);
			totatSent += nSent;
		}
	}

	void ReadProtobufMessage(std::istream& in, NPb::Message* dstMessage) {
        NPbIO::IstreamInputStream zeroIn(&in);
        NPbIO::CodedInputStream codedInput(&zeroIn);

        THeader header;
		ReadHeaderFromCoded(codedInput, &header);

		L_INFO << "Got message type " << header.MessageType << ", size " << header.MessageSize << "b";
		dstMessage->ParseFromCodedStream(&codedInput);
	}	
}