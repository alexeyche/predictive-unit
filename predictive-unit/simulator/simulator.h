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

	class TSimulator {
	public:

		static constexpr ui32 BufferSize = 2000;

		struct TStats: TProtoStructure<NPredUnitPb::TStats> {
			TStats(TLayerConfig config) {
				Membrane = TMatrixD::Zero(config.BatchSize, BufferSize*config.LayerSize);
				Activation = TMatrixD::Zero(config.BatchSize, BufferSize*config.LayerSize);
				F = TMatrixD::Zero(config.InputSize*config.FilterSize, config.LayerSize);
				Fc = TMatrixD::Zero(config.LayerSize, config.LayerSize);
			}

			NPredUnitPb::TStats ToProto() const {
				NPredUnitPb::TStats stats;
				SerializeMatrix(Membrane, stats.mutable_membrane());
				SerializeMatrix(Activation, stats.mutable_activation());
				SerializeMatrix(F, stats.mutable_f());
				SerializeMatrix(Fc, stats.mutable_fc());
				return stats;
			}

			TMatrixD Membrane;
			TMatrixD Activation;
			
			TMatrixD F;
			TMatrixD Fc;
		};

		struct TSimulatorCtx {
			TSimulatorCtx(TSimConfig config, TDispatcher& dispatcher)
				: Config(config)
				, OutputQueue(1)
				, Dispatcher(dispatcher)
			{
				InputBuffLock.clear();
				StatsBuffLock.clear();
				InputBuff = TMatrixD::Zero(Config.LayerConfig.BatchSize, Config.LayerConfig.InputSize*BufferSize);
				OutputBuff = TMatrixD::Zero(Config.LayerConfig.BatchSize, Config.LayerConfig.LayerSize*BufferSize);
				Dispatcher.AddQueueToDispatch(Config.Id, &OutputQueue);
			}
			
			TSimConfig Config;

			TMutex SimMutex;
			TMatrixD InputBuff;
			TMatrixD OutputBuff;

			TOptional<TStats> Stats;
			TOptional<TStats> CollectedStats;

			TAtomicFlag InputBuffLock;
			TAtomicFlag StatsBuffLock;

			TMatrixRWQ OutputQueue;

			TDispatcher& Dispatcher;
		};
		

		TSimulator(ui32 jobsNum, THostMap hostMap)
			: JobsNum(jobsNum)
			, HostMap(hostMap)
			, Dispatcher(HostMap)
		{
		}

		~TSimulator() {
			if (SimThread.size() > 0) {
					
				for (auto& t: SimThread) {
					L_INFO << "Joining SimThread #" << t.first;
					t.second.join();	
				}
			}
			Dispatcher.Stop();
		}
		
		void StartSimulationsAsync() {
			for (const auto& hr: HostMap.HostRecord) {
				for (const auto& sim: hr.SimConfig) {
					StartSimulationAsync(sim);
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
					std::make_unique<TSimulatorCtx>(simConfig, Dispatcher)
				);
				
				ENSURE(res.second, "Failed to insers thread ctx");
				ENSURE(res.first->second->SimMutex.try_lock(), "Failed to acquire sim lock");
				TSimulatorCtx& ctx = *(res.first->second);
				
				// if (startSim.InputData.Data.size() > 0) {
				// 	IngestData(startSim.InputData, &ctx);
				// }

				if (simConfig.CollectStats) {
					ctx.Stats.emplace(simConfig.LayerConfig);	
				}

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

		void IngestData(const TInputData& inp, TSimulatorCtx* simCtx) {
			for (ui32 t=0; t<ToUi32(inp.Data.cols()); t+=simCtx->Config.LayerConfig.InputSize) {
				simCtx->InputBuff.block(
					0, 
					t,
					simCtx->Config.LayerConfig.BatchSize, 
					simCtx->Config.LayerConfig.InputSize*simCtx->Config.LayerConfig.FilterSize
				) = inp.Data.block(
					0, 
					t,
					simCtx->Config.LayerConfig.BatchSize, 
					simCtx->Config.LayerConfig.InputSize*simCtx->Config.LayerConfig.FilterSize
				);
			}
		}

		void IngestDataAsync(const TInputData& inp) {
			auto& simCtx = GetContext(inp.SimId);
			
			ENSURE(simCtx.Config.LayerConfig.BatchSize == ToUi32(inp.Data.rows()), "Wrong batch size"); 
			while (simCtx.InputBuffLock.test_and_set(std::memory_order_acquire)) {}
			IngestData(inp, &simCtx);
			simCtx.InputBuffLock.clear(std::memory_order_release);
		}

		TOptional<TStats> GetStats(const TStatRequest& sr) {
			auto& simCtx = GetContext(sr.SimId);	
			
			TOptional<TStats> stat;
			RunLock(simCtx.StatsBuffLock, [&]() {
				if (simCtx.CollectedStats) {
					L_INFO << "Found collected stats to give away";
					simCtx.CollectedStats.swap(stat);
				} else {
					L_INFO << "Collected stats are not found, starting to collect";
					simCtx.Stats.emplace(simCtx.Config.LayerConfig);	
				}
			}); 
			return stat;
		}

		static void StartSimulationImpl(TSimulatorCtx& sim, TSimConfig simConfig) {
			TGuard lock(sim.SimMutex, std::adopt_lock);

			L_INFO << "Starting simulation #" << sim.Config.Id << " till " << simConfig.SimulationTime;
			
			TLayer layer(simConfig.LayerConfig);

			ui64 iter = 0;
			ui64 nCycle = 0;
			
			i32 nSimTime = 0;

			bool collectStats = false;
			
			RunLock(sim.StatsBuffLock, [&]() {
				if (sim.Stats) {
					L_INFO << "Cycle #" << nCycle << ": Starting to collect stats";
					collectStats = true;
				}
			});

			const bool endLess = simConfig.SimulationTime < 0;
			while (endLess || (nSimTime < simConfig.SimulationTime)) {
				ui32 inputIter = iter * sim.Config.LayerConfig.InputSize;
				ui32 layerIter = iter * sim.Config.LayerConfig.LayerSize;
				
				if (inputIter >= sim.InputBuff.cols() - sim.Config.LayerConfig.FilterSize) {
					RunLock(sim.StatsBuffLock, [&]() {
						if (collectStats) {
							collectStats = false;
							sim.CollectedStats = TOptional<TStats>(sim.Config.LayerConfig);
							sim.CollectedStats.swap(sim.Stats);
							sim.CollectedStats->F = layer.F;
							sim.CollectedStats->Fc = layer.Fc;
							L_INFO << "Sim id #" << sim.Config.Id << ", cycle #" << nCycle << ": Stat collect is done";
						} else
						if (sim.Stats) {
							L_INFO << "Sim id #" << sim.Config.Id << ", cycle #" << nCycle << ": Starting to collect stats";
							collectStats = true;
						}
					});

					L_INFO << "Sim id #" << sim.Config.Id << " is adding data to output buffer, approx size: " << sim.OutputQueue.size_approx();
					while (!sim.OutputQueue.enqueue(sim.OutputBuff)) {
						L_INFO << "Sim id #" << sim.Config.Id << " waiting output queue, approx size: " << sim.OutputQueue.size_approx();
					}
					
					iter = 0;
					inputIter = iter * sim.Config.LayerConfig.InputSize;
					layerIter = iter * sim.Config.LayerConfig.LayerSize;
					++nCycle;
				}
				
				while (sim.InputBuffLock.test_and_set(std::memory_order_acquire)) {}
				auto input = sim.InputBuff.block(
					0, 
					inputIter,
					sim.Config.LayerConfig.BatchSize, 
					sim.Config.LayerConfig.InputSize*sim.Config.LayerConfig.FilterSize	
				);
				sim.InputBuffLock.clear(std::memory_order_release);

				layer.Tick(input);
				
				if (collectStats) {
					RunLock(sim.StatsBuffLock, [&]() {
						sim.Stats->Membrane.block(
							0, 
							layerIter,
							sim.Config.LayerConfig.BatchSize, 
							sim.Config.LayerConfig.LayerSize
						) = layer.Membrane;
						sim.Stats->Activation.block(
							0, 
							layerIter,
							sim.Config.LayerConfig.BatchSize, 
							sim.Config.LayerConfig.LayerSize
						) = layer.Activation;	
					});
				}
		
				sim.OutputBuff.block(
					0, 
					layerIter,
					sim.Config.LayerConfig.BatchSize, 
					sim.Config.LayerConfig.LayerSize
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
		THostMap HostMap;
		TDispatcher Dispatcher;
	};
}