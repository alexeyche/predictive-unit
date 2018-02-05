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
		static constexpr ui32 BatchSize = 4;
		static constexpr ui32 InputSize = 2;
		static constexpr ui32 FilterSize = 1;
		static constexpr ui32 LayerSize = 10;
		static constexpr ui32 BufferSize = 100;

		struct TStats: TProtoStructure<NPredUnitPb::TStats> {
			NPredUnitPb::TStats ToProto() const {
				NPredUnitPb::TStats stats;
				SerializeMatrix(Membrane, stats.mutable_membrane());
				SerializeMatrix(Activation, stats.mutable_activation());
				SerializeMatrix(F, stats.mutable_f());
				SerializeMatrix(Fc, stats.mutable_fc());
				return stats;
			}

			TMatrix<BatchSize, BufferSize*LayerSize> Membrane;
			TMatrix<BatchSize, BufferSize*LayerSize> Activation;
			
			TMatrix<InputSize*FilterSize, LayerSize> F;
			TMatrix<LayerSize, LayerSize> Fc;
		};

		using TSimLayer = TLayer<BatchSize, LayerSize, InputSize, FilterSize>;

		struct TSimulatorCtx {
			TSimulatorCtx(ui32 simId, TDispatcher& dispatcher)
				: SimId(simId)
				, OutputQueue(1)
				, Dispatcher(dispatcher)
			{
				InputBuffLock.clear();
				StatsBuffLock.clear();
				InputBuff = TMatrix<BatchSize, InputSize*BufferSize>::Zero();
				Dispatcher.AddQueueToDispatch(SimId, &OutputQueue);
			}

			TMutex SimMutex;
			TMatrix<BatchSize, InputSize*BufferSize> InputBuff;
			TMatrix<BatchSize, LayerSize*BufferSize> OutputBuff;

			TOptional<TStats> Stats;
			TOptional<TStats> CollectedStats;

			TAtomicFlag InputBuffLock;
			TAtomicFlag StatsBuffLock;

			ui32 SimId;
			TMatrixRWQ OutputQueue;

			TDispatcher& Dispatcher;
		};
		

		TSimulator(ui32 jobsNum, TDispatcher& dispatcher)
			: JobsNum(jobsNum)
			, Dispatcher(dispatcher)
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

		bool StartSimulationAsync(const TStartSim& startSim) {
			if (!IsSumulationRunning(startSim.SimId)) {
				TGuard lock(SimThreadAccMutex);
				
				auto simCtxOld = SimCtx.find(startSim.SimId);
				if (simCtxOld != SimCtx.end()) {
					SimCtx.erase(simCtxOld);
				}

				auto simThreadOld = SimThread.find(startSim.SimId);
				if (simThreadOld != SimThread.end()) {
					simThreadOld->second.join();
					SimThread.erase(simThreadOld);
				}
				
				auto res = SimCtx.emplace(
					startSim.SimId, 
					std::make_unique<TSimulatorCtx>(startSim.SimId, Dispatcher)
				);
				
				ENSURE(res.second, "Failed to insers thread ctx");
				ENSURE(res.first->second->SimMutex.try_lock(), "Failed to acquire sim lock");
				TSimulatorCtx& ctx = *(res.first->second);
				
				if (startSim.InputData.Data.size() > 0) {
					IngestData(startSim.InputData, &ctx);
				}

				if (startSim.CollectStats) {
					ctx.Stats.emplace();	
				}

				SimThread.emplace(
					startSim.SimId,
					std::thread(
						StartSimulationImpl, 
						std::ref(ctx), 
						startSim
					)
				);
				return true;
			} else {
				L_INFO << "Sim # " << startSim.SimId << " is running, unable to start new one ...";
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
			for (ui32 t=0; t<ToUi32(inp.Data.cols()); t+=InputSize) {
				simCtx->InputBuff.block<BatchSize, InputSize*FilterSize>(0, t) = inp.Data.block<BatchSize, InputSize*FilterSize>(0, t);
			}
		}

		void IngestDataAsync(const TInputData& inp) {
			auto& simCtx = GetContext(inp.SimId);
			
			ENSURE(BatchSize == inp.Data.rows(), "Wrong batch size"); 
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
					simCtx.Stats.emplace();	
				}
			}); 
			return stat;
		}

		static void StartSimulationImpl(TSimulatorCtx& sim, TStartSim startSim) {
			TGuard lock(sim.SimMutex, std::adopt_lock);

			L_INFO << "Starting simulation #" << sim.SimId << " till " << startSim.SimulationTime;
			
			TSimLayer layer(startSim.LayerConfig);

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

			const bool endLess = startSim.SimulationTime < 0;
			while (endLess || (nSimTime < startSim.SimulationTime)) {
				ui32 inputIter = iter * InputSize;
				ui32 layerIter = iter * LayerSize;
				
				if (inputIter >= sim.InputBuff.cols() - FilterSize) {
					RunLock(sim.StatsBuffLock, [&]() {
						if (collectStats) {
							collectStats = false;
							sim.CollectedStats = TOptional<TStats>();
							sim.CollectedStats.swap(sim.Stats);
							sim.CollectedStats->F = layer.F;
							sim.CollectedStats->Fc = layer.Fc;
							L_INFO << "Sim id #" << sim.SimId << ", cycle #" << nCycle << ": Stat collect is done";
						} else
						if (sim.Stats) {
							L_INFO << "Sim id #" << sim.SimId << ", cycle #" << nCycle << ": Starting to collect stats";
							collectStats = true;
						}
					});

					L_INFO << "Sim id #" << sim.SimId << " is adding data to output buffer, approx size: " << sim.OutputQueue.size_approx();
					while (!sim.OutputQueue.enqueue(sim.OutputBuff)) {
						L_INFO << "Sim id #" << sim.SimId << " waiting output queue, approx size: " << sim.OutputQueue.size_approx();
					}
					
					iter = 0;
					inputIter = iter * InputSize;
					layerIter = iter * LayerSize;
					++nCycle;
				}
				
				while (sim.InputBuffLock.test_and_set(std::memory_order_acquire)) {}
				auto input = sim.InputBuff.block<BatchSize, InputSize*FilterSize>(0, inputIter);
				sim.InputBuffLock.clear(std::memory_order_release);

				layer.Tick(input);
				
				if (collectStats) {
					RunLock(sim.StatsBuffLock, [&]() {
						sim.Stats->Membrane.block<BatchSize, LayerSize>(0, layerIter) = layer.Membrane;
						sim.Stats->Activation.block<BatchSize, LayerSize>(0, layerIter) = layer.Activation;	
					});
				}
		
				sim.OutputBuff.block<BatchSize, LayerSize>(0, layerIter) = layer.Activation;

				++iter;
				++nSimTime;
			}
			
  			L_INFO << "Sim id #" << sim.SimId << ": simulation is done";
		}

		ui32 JobsNum;
		TMutex SimThreadAccMutex;
		TMap<ui32, TThread> SimThread;
		TMap<ui32, TUniquePtr<TSimulatorCtx>> SimCtx;
		TDispatcher& Dispatcher;
	};
}