#pragma once

#include "protocol.h"

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
		static constexpr ui32 BufferSize = 20;

		TSimulator() {
			InputBuffLock.clear();
		}

		~TSimulator() {
			if (SimThread) {
				L_INFO << "Joining SimThread";
				SimThread->join();
			}
		}

		bool StartSimulationAsync(const TStartSim& startSim) {
			if (SimMutex.try_lock()) {
				if (SimThread) {
					SimThread->join();
				}

				InputBuff = TMatrix<BatchSize, InputSize*BufferSize>::Zero();
				SimThread.emplace(StartSimulationImpl, std::ref(*this), startSim);
				return true;
			} else {
				L_INFO << "Sim is running, unable to start new one ...";
				return false;
			}
		}

		bool IsSumulationRunning() {
			if (SimMutex.try_lock()) {
				SimMutex.unlock();
				return false;
			} else {
				return true;
			}
		}

		void IngestData(const TMatrixD& data) {
			ENSURE(BatchSize == data.rows(), "Wrong batch size"); 
			
			while (InputBuffLock.test_and_set(std::memory_order_acquire)) {}
			for (ui32 t=0; t<ToUi32(data.cols()); ++t) {
				InputBuff.block<BatchSize, InputSize*FilterSize>(0, t) = data;	
			}
			
			L_INFO << InputBuff.rows() << "x" << InputBuff.cols();
			L_INFO << InputBuff;
			InputBuffLock.clear(std::memory_order_release);
		}

		static void StartSimulationImpl(TSimulator& sim, TStartSim startSim) {
			TGuard lock(sim.SimMutex, std::adopt_lock);

			L_INFO << "Starting simulation till " << startSim.SimulationTime;
			
			TLayer<BatchSize, LayerSize, InputSize, FilterSize> layer(startSim.LayerConfig);

			ui32 inputIter = 0;
			ui32 nCycle = 0;
			
			i32 nSimTime = 0;
			bool endLess = startSim.SimulationTime < 0;
			while (endLess || (nSimTime < startSim.SimulationTime)) {
				if (inputIter >= sim.InputBuff.cols() - FilterSize) {
					inputIter = 0;
					++nCycle;
				}
				
				while (sim.InputBuffLock.test_and_set(std::memory_order_acquire)) {}
				auto input = sim.InputBuff.block<BatchSize, InputSize*FilterSize>(0, inputIter);
				sim.InputBuffLock.clear(std::memory_order_release);

				layer.Tick(input);

				inputIter += InputSize;

				++nSimTime;
			}
			
  			L_INFO << "Simulation is done";
		}

		TOptional<TThread> SimThread;
		TMutex SimMutex;
		TMatrix<BatchSize, InputSize*BufferSize> InputBuff;
		std::atomic_flag InputBuffLock;
	};
}