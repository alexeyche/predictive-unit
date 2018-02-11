#pragma once

#include <predictive-unit/base.h>
#include <predictive-unit/log.h>
#include <predictive-unit/nn/layer-config.h>

namespace NPredUnit {

	inline TMatrixD unitScalingInit(ui32 Rows, ui32 Cols) {
		auto m = TMatrixD::Random(Rows, Cols);
		double initSpan = std::sqrt(3.0)/std::sqrt(std::max(Rows, Cols));
		return 2.0 * initSpan * (m.array() + 1.0) / 2.0 - initSpan;
	}

	inline TMatrixD relu(TMatrixD x) {
		return x.cwiseMax(0.0);
	}


	class TLayer {
	public:

		TLayer(TLayerConfig config)
			: C(config.LayerConstants)
			, BatchSize(config.BatchSize)
			, LayerSize(config.LayerSize)
			, InputSize(config.InputSize)
			, FilterSize(config.FilterSize)
 		{
 			Activation = TMatrixD::Zero(BatchSize, LayerSize);
 			Membrane = TMatrixD::Zero(BatchSize, LayerSize);

			F = unitScalingInit(FilterSize * InputSize, LayerSize);
			F = (F.array().rowwise())/(F.colwise().norm().array());
			Fc = F.transpose() * F - TMatrixD::Identity(LayerSize, LayerSize);
		}

		void Tick(const TMatrixD input) {
			TMatrixD input_hat = Activation * F.transpose();
			
			TMatrixD e = input - input_hat;

			Membrane += (e * F - Membrane) / C.Tau;

			Activation = relu(Membrane);
		}

	private:
		TLayerConstants C;
	
	private:
		ui32 BatchSize;
		ui32 LayerSize;
		ui32 InputSize;
		ui32 FilterSize;


	public:
		TMatrixD Membrane;
		TMatrixD Activation;
		
		TMatrixD F;
		TMatrixD Fc;
	};




} // namespace NPredUnit