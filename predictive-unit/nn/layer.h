#pragma once

#include <predictive-unit/base.h>
#include <predictive-unit/log.h>
#include <predictive-unit/nn/layer-config.h>

namespace NPredUnit {

	template <ui32 Rows, ui32 Cols>
	TMatrix<Rows, Cols> unitScalingInit() {
		auto m = TMatrix<Rows, Cols>::Random();
		double initSpan = std::sqrt(3.0)/std::sqrt(std::max(Rows, Cols));
		return 2.0 * initSpan * (m.array() + 1.0) / 2.0 - initSpan;
	}

	template <int Rows, int Cols>
	TMatrix<Rows, Cols> relu(TMatrix<Rows, Cols> x) {
		return x.cwiseMax(0.0);
	}


	template <ui32 BatchSize, ui32 LayerSize, ui32 InputSize, ui32 FilterSize>
	class TLayer {
	public:

		TLayer(TLayerConfig config)
			: C(config)
 		{
 			Activation = TMatrix<BatchSize, LayerSize>::Zero();
 			Membrane = TMatrix<BatchSize, LayerSize>::Zero();

			F = unitScalingInit<FilterSize * InputSize, LayerSize>();
			F = (F.array().rowwise())/(F.colwise().norm().array());
			Fc = F.transpose() * F - TMatrix<LayerSize, LayerSize>::Identity();
		}

		void Tick(const TMatrix<BatchSize, InputSize*FilterSize> input) {
			TMatrix<BatchSize, InputSize*FilterSize> input_hat = Activation * F.transpose();
			TMatrix<BatchSize, InputSize*FilterSize> e = input - input_hat;

			Membrane += (e * F - Membrane) / C.Tau;

			Activation = relu(Membrane);
		}

	private:
		TLayerConfig C;


	public:
		TMatrix<BatchSize, LayerSize> Membrane;
		TMatrix<BatchSize, LayerSize> Activation;
		
		TMatrix<InputSize*FilterSize, LayerSize> F;
		TMatrix<LayerSize, LayerSize> Fc;
	};




} // namespace NPredUnit