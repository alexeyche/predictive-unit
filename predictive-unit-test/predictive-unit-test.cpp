
#include <predictive-unit/protos/messages.pb.h>
#include <predictive-unit/base.h>
#include <predictive-unit/log.h>


#include <google/protobuf/message.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>

namespace NPb = google::protobuf;
namespace NPbIO = google::protobuf::io;

using namespace NPredUnit;

void TestProtobuf() {
	NPredUnitPb::TStartSim startSimMessage;
	startSimMessage.set_simulationtime(10000000);

	NPb::uint32 messageType = NPredUnitPb::TMessageType::START_SIM;
	NPb::uint32 messageSize = startSimMessage.ByteSize();

	ui32 bytesToSend = 4 + 4 + messageSize;
	char* buf = new char[bytesToSend];

	L_INFO << "Message size: " << messageSize << " " << 8;


	{
		NPb::io::ArrayOutputStream aos(buf, bytesToSend);
		NPbIO::CodedOutputStream codedOutput(&aos);	
	
		codedOutput.WriteVarint32(messageType);
		codedOutput.WriteVarint32(messageSize);
		bool succWrite = startSimMessage.SerializeToCodedStream(&codedOutput);
		ENSURE(succWrite, "Failed to write");
	}

	
	L_INFO << startSimMessage.DebugString();

	ui32 headerSize = sizeof(NPb::uint32) + sizeof(NPb::uint32);

	NPredUnitPb::TStartSim startSimMessage2;
	{
		NPb::io::ArrayInputStream aisM(buf, headerSize + messageSize);
		NPbIO::CodedInputStream codedInputM(&aisM);

		bool succ0 = codedInputM.ReadVarint32(&messageType);
		ENSURE(succ0, "Failed to read message type");
		bool succ1 = codedInputM.ReadVarint32(&messageSize);
		ENSURE(succ1, "Failed to read message size");

		bool succ = startSimMessage2.ParseFromCodedStream(&codedInputM);
		if (!succ) {
			L_ERROR << "Failed to read protobuf";	
		} else {
			L_INFO << startSimMessage2.DebugString(); 
		}
	}
	ENSURE(startSimMessage2.simulationtime() == startSimMessage.simulationtime(), "FAIL");
	delete[] buf;
}


int main(int argc, char** argv) {
	TestProtobuf();

	return 0;
}
