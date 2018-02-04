#pragma once

#include "defaults.h"

#include <predictive-unit/base.h>
#include <predictive-unit/log.h>
#include <predictive-unit/protos/messages.pb.h>
#include <predictive-unit/util/proto-struct.h>
#include <predictive-unit/nn/layer-config.h>

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

	template <typename TProtoStruct>
	TProtoStruct ReadProtobufMessageFromSocket(Poco::Net::StreamSocket& sck, ui32 messageSize) {
		typename TProtoStruct::TTemplateProto dstProto;
		ReadProtobufMessageFromSocket(sck, messageSize, &dstProto);
		return TProtoStruct(dstProto);
	}


	void WriteHeaderAndProtobufMessageToSocket(const NPb::Message& src, NPredUnitPb::TMessageType::EMessageType messageType, Poco::Net::StreamSocket* sck);

	void ReadProtobufMessage(std::istream& in, NPb::Message* dst);


	struct TInputData: TProtoStructure<NPredUnitPb::TInputData> {
		TInputData(const NPredUnitPb::TInputData& m) {
			FillFromProto(m, NPredUnitPb::TInputData::kSimIdFieldNumber, &SimId);
			FillFromProto(m, NPredUnitPb::TInputData::kDataFieldNumber, &Data);
		}

		ui32 SimId = 0;
		TMatrixD Data;
	};


	struct TStartSim: public TProtoStructure<NPredUnitPb::TStartSim> {
		static TStartSim DefaultConfig() {
			return TStartSim(NPredUnitPb::TStartSim());	
		}
		
		TStartSim(const NPredUnitPb::TStartSim& m)
			: LayerConfig(m.layerconfig()) 
			, InputData(m.inputdata())
		{
			FillFromProto(m, NPredUnitPb::TStartSim::kSimIdFieldNumber, &SimId);
			FillFromProto(m, NPredUnitPb::TStartSim::kSimulationTimeFieldNumber, &SimulationTime);
			FillFromProto(m, NPredUnitPb::TStartSim::kCollectStatsFieldNumber, &CollectStats);
		}

		TLayerConfig LayerConfig;
		ui32 SimId = 0;
		i32 SimulationTime = -1;
		bool CollectStats = true;
		TInputData InputData;
	};


	struct TStatRequest: TProtoStructure<NPredUnitPb::TStatRequest> {
		TStatRequest(const NPredUnitPb::TStatRequest& m) {
			FillFromProto(m, NPredUnitPb::TStatRequest::kSimIdFieldNumber, &SimId);
		}
		
		ui32 SimId = 0;
	};

}