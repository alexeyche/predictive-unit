#pragma once

#include <predictive-unit/protos/layer-config.pb.h>
#include <predictive-unit/util/proto-struct.h>

namespace NPredUnit {


	struct TLayerConfig: public TProtoStructure<NPredUnitPb::TLayerConfig> {
		TLayerConfig(const NPredUnitPb::TLayerConfig& m) {
			FillFromProto(m, 1, &Tau);			
			FillFromProto(m, 2, &TauMean);			
			FillFromProto(m, 3, &AdaptGain);			
			FillFromProto(m, 4, &LearningRate);			
		}

		double Tau;
		double TauMean;
		double AdaptGain;
		double LearningRate;
	};

}