#pragma once

#include <predictive-unit/protos/layer-config.pb.h>
#include <predictive-unit/util/proto-struct.h>

namespace NPredUnit {


	struct TLayerConstants: public TProtoStructure<NPredUnitPb::TLayerConstants> {
		TLayerConstants(const NPredUnitPb::TLayerConstants& m) {
			FillFromProto(m, NPredUnitPb::TLayerConstants::kTauFieldNumber, &Tau);			
			FillFromProto(m, NPredUnitPb::TLayerConstants::kTauMeanFieldNumber, &TauMean);			
			FillFromProto(m, NPredUnitPb::TLayerConstants::kAdaptGainFieldNumber, &AdaptGain);			
			FillFromProto(m, NPredUnitPb::TLayerConstants::kLearningRateFieldNumber, &LearningRate);			
		}

		double Tau = 5.0;
		double TauMean = 100.0;
		double AdaptGain = 1.0;
		double LearningRate = 0.1;
	};

	struct TLayerConfig: public TProtoStructure<NPredUnitPb::TLayerConfig> {
		TLayerConfig(const NPredUnitPb::TLayerConfig& m)
			: LayerConstants(m.layerconstants()) 
		{
			FillFromProto(m, NPredUnitPb::TLayerConfig::kLayerSizeFieldNumber, &LayerSize);
			FillFromProto(m, NPredUnitPb::TLayerConfig::kInputSizeFieldNumber, &InputSize);
			FillFromProto(m, NPredUnitPb::TLayerConfig::kBatchSizeFieldNumber, &BatchSize);
			FillFromProto(m, NPredUnitPb::TLayerConfig::kFilterSizeFieldNumber, &FilterSize);
		}

		ui32 LayerSize = 100;
		ui32 InputSize = 2;
		ui32 BatchSize = 4;
		ui32 FilterSize = 1;
		TLayerConstants LayerConstants;
	};

}