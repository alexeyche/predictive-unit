#pragma once

#include <predictive-unit/simulator/dispatcher.h>
#include <predictive-unit/protocol.h>
#include <predictive-unit/protos/messages.pb.h>
#include <predictive-unit/nn/layer.h>
#include <predictive-unit/util/proto-struct.h>
#include <predictive-unit/util/optional.h>
#include <predictive-unit/base.h>
#include <predictive-unit/log.h>

namespace NPredUnit {


	struct TSimConfig {
		ui32 Id = 0;
		TNetworkConfig NetworkConfig;
	};


	class TSimulator {
	public:
		struct TSimulatorCtx {
			TSimulatorCtx(const TNetworkConfig& layerConfig)
				: NetworkConfig(layerConfig) 
			{
				InputBuff = TMatrixD::Zero(NetworkConfig.BatchSize, NetworkConfig.InputSize*NetworkConfig.BufferSize);
				OutputBuff = TMatrixD::Zero(NetworkConfig.BatchSize, NetworkConfig.NetworkSize*NetworkConfig.BufferSize);
			}

			TMutex SimMutex;
			ui32 Id = 0;
			
			TMatrixD InputBuff;
			TMatrixD OutputBuff;

			TNetworkConfig NetworkConfig;
		};
		
		TSimulator(ui32 jobsNum)
			: JobsNum(jobsNum)
		{
		}

		~TSimulator() {
			if (SimThread.size() > 0) {
					
				for (auto& t: SimThread) {
					L_INFO << "Joining SimThread #" << t.first;
					t.second.join();	
				}
			}
		}
		
		bool StartSimulationAsync(const TSimConfig& simConfig) {
			if (!IsSumulationRunning(simConfig.Id)) {
				TGuard lock(SimThreadAccMutex);
				
				auto simCtxOld = SimCtx.find(simConfig.Id);
				if (simCtxOld != SimCtx.end()) {
					SimCtx.erase(simCtxOld);
				}

				auto simThreadOld = SimThread.find(simConfig.Id);
				if (simThreadOld != SimThread.end()) {
					simThreadOld->second.join();
					SimThread.erase(simThreadOld);
				}
				
				auto res = SimCtx.emplace(
					simConfig.Id, 
					std::make_unique<TSimulatorCtx>(simConfig.NetworkConfig)
				);
				
				ENSURE(res.second, "Failed to insers thread ctx");
				ENSURE(res.first->second->SimMutex.try_lock(), "Failed to acquire sim lock");
				TSimulatorCtx& ctx = *(res.first->second);

				SimThread.emplace(
					simConfig.Id,
					std::thread(
						StartSimulationImpl, 
						std::ref(ctx), 
						simConfig
					)
				);
				return true;
			} else {
				L_INFO << "Sim # " << simConfig.Id << " is running, unable to start new one ...";
				return false;
			}
		}

		bool IsSumulationRunning(ui32 simId) {
			TGuard lock(SimThreadAccMutex);
			auto simCtxPtr = SimCtx.find(simId);
			if (simCtxPtr == SimCtx.end()) {
				return false;
			}

			if (simCtxPtr->second->SimMutex.try_lock()) {
				simCtxPtr->second->SimMutex.unlock();
				return false;
			} else {
				return true;
			}
		}

		TSimulatorCtx& GetContext(ui32 simId) {
			TGuard lock(SimThreadAccMutex);
			auto simCtxPtr = SimCtx.find(simId);
			ENSURE(simCtxPtr != SimCtx.end(), "Simulation #" << simId << " is not found");
			return *(simCtxPtr->second);
		}

		static void StartSimulationImpl(TSimulatorCtx& sim, TSimConfig simConfig) {
			TGuard lock(sim.SimMutex, std::adopt_lock);

			L_INFO << "Starting simulation #" << sim.Config.Id << " till " << simConfig.SimulationTime;
			
			TNetwork layer(simConfig.NetworkConfig);

			ui64 iter = sim.Iterator;
			ui64 nCycle = 0;
			
			i32 nSimTime = 0;

			
			const bool endLess = simConfig.SimulationTime < 0;
			while (endLess || (nSimTime < simConfig.SimulationTime)) {
				ui32 inputIter = iter * sim.Config.NetworkConfig.InputSize;
				ui32 layerIter = iter * sim.Config.NetworkConfig.NetworkSize;

				auto input = sim.InputBuff.block(
					0, 
					inputIter,
					sim.Config.NetworkConfig.BatchSize, 
					sim.Config.NetworkConfig.InputSize*sim.Config.NetworkConfig.FilterSize	
				);
				
				layer.Tick(input);
				
				sim.OutputBuff.block(
					0, 
					layerIter,
					sim.Config.NetworkConfig.BatchSize, 
					sim.Config.NetworkConfig.NetworkSize
				) = layer.Activation;


				++iter;
				++nSimTime;
			}
			
  			L_INFO << "Sim id #" << sim.Config.Id << ": simulation is done";
		}

		ui32 JobsNum;
		TMutex SimThreadAccMutex;
		TMap<ui32, TThread> SimThread;
		TMap<ui32, TUniquePtr<TSimulatorCtx>> SimCtx;
	};
}