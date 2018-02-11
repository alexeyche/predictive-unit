
#include "io.h"

#include <predictive-unit/base.h>

namespace NPredUnit {

	class ILayer {
	public:
		virtual ~ILayer() {}
		virtual void Tick(TMatrixD input) = 0;
	};


	enum ELayerType: ui8 {
		ELT_PREDICTIVE = 0,
		ELT_OUTPUT = 1
	};

	struct TLayerConfig {
		ui8 LayerType = ELT_PREDICTIVE;
		ui32 NetworkSize = 100;
		ui32 InputSize = 2;
		ui32 BatchSize = 4;
		ui32 FilterSize = 1;
		ui32 BufferSize = 1000;
		double Tau = 5.0;
		double TauMean = 100.0;
		double AdaptGain = 1.0;
		double LearningRate = 0.1;


		template <typename T>
		void Serial(TIOStream<T>&& io) {
			io.VarUInt(LayerType);
			io.VarUInt(NetworkSize);
			io.VarUInt(InputSize);
			io.VarUInt(BatchSize);
			io.VarUInt(FilterSize);
			io.VarUInt(BufferSize);
			io.Double(Tau);
			io.Double(TauMean);
			io.Double(AdaptGain);
			io.Double(LearningRate);
		}
	};

	struct TNetworkConfig {
		TVector<TLayerConfig> LayerConfigs;

		template <typename T>
		void Serial(TIOStream<T>&& io) {
			io.Array(LayerConfigs);
		}
	};

	
	class TPredictiveLayer: public ILayer {
	public:
		TPredictiveLayer(TLayerConfig config) {}
		
		void Tick(TMatrixD input) override {
			L_INFO << "Tick";
		}
	};

	TUniquePtr<ILayer> ConstructLayer(TLayerConfig config) {
		switch (config.LayerType) {
			case ELT_PREDICTIVE:
				return std::make_unique<TPredictiveLayer>(config);
			case ELT_OUTPUT:
				ENSURE(0, "Not implemented");
		}
		ENSURE(0, "Not implemented");
	}


}