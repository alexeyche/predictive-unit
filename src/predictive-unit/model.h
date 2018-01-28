#pragma once

#include <predictive-unit/base.h>

namespace NPredUnit {

	template <ui32 Rows, ui32 Cols>
	TMatrix<Rows, Cols> unitScalingInit() {
		auto m = TMatrix<Rows, Cols>::Random();
		double initSpan = std::sqrt(3.0)/std::sqrt(std::max(Rows, Cols));
		return 2.0 * initSpan * (m.array() + 1.0) / 2.0 - initSpan;
	}

	template <ui32 Rows, ui32 Cols>
	TMatrix<Rows, Cols> relu(TMatrix<Rows, Cols> x) {
		return x.cwiseMax(0.0);
	}


	template <ui32 BatchSize, ui32 LayerSize, ui32 InputSize, ui32 FilterSize>
	class TModel {
	public:
		TModel()
			: Membrane(TMatrix<BatchSize, LayerSize>::Zero())
			, Activation(TMatrix<BatchSize, LayerSize>::Zero())
 		{			
			F = unitScalingInit<FilterSize * InputSize, LayerSize>();
			F = (F.array().rowwise())/(F.colwise().norm().array());
			Fc = F.transpose() * F - TMatrix<LayerSize, LayerSize>::Identity();
		}

		void Tick(TMatrix<BatchSize, InputSize*FilterSize> input) {
			TMatrix<BatchSize, LayerSize> feedback = Activation * Fc;

		}

	private:
		double Tau;
		double Lambda;

	public:
		TMatrix<BatchSize, LayerSize> Membrane;
		TMatrix<BatchSize, LayerSize> Activation;
		
		TMatrix<InputSize*FilterSize, LayerSize> F;
		TMatrix<LayerSize, LayerSize> Fc;
	};




} // namespace NPredUnit