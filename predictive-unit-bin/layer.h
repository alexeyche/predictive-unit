
#include "protocol.h"

#include <predictive-unit/base.h>

namespace NPredUnit {

	class ILayer {
	public:
		virtual ~ILayer() {}
		virtual void Tick(TMatrixD input) = 0;
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