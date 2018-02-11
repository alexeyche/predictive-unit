#pragma once

#include "layer.h"

#include <predictive-unit/base.h>
#include <predictive-unit/log.h>

namespace NPredUnit {


	class TNetwork {
	public:
		TNetwork(const TNetworkConfig& config)
			: Config(config)
		{
			for (const auto& lc: Config.LayerConfigs) {
				Layers.push_back(ConstructLayer(lc));
			}
		}

		TNetworkConfig Config;
		TVector<TUniquePtr<ILayer>> Layers;
	};

}