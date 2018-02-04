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

		double Tau = 5.0;
		double TauMean = 100.0;
		double AdaptGain = 1.0;
		double LearningRate = 0.1;
	};

}