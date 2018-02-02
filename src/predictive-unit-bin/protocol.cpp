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

	void ReadProtobufMessage(std::istream& in) {
        NPbIO::IstreamInputStream zeroIn(&in);
        NPbIO::CodedInputStream codedInput(&zeroIn);

        THeader header;
		ReadHeaderFromCoded(codedInput, &header);

		L_INFO << "Got message type " << header.MessageType << ", size " << header.MessageSize << "b";

		switch (header.MessageType) {
			case NPredUnitPb::TMessageType::START_SIM:
						
				NPredUnitPb::TStartSim startSimMessage;
				startSimMessage.ParseFromCodedStream(&codedInput);
		
				L_INFO << startSimMessage.DebugString();
				break;
		}
	}	
}