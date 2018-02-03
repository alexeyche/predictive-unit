#pragma once

#include <predictive-unit/protos/messages.pb.h>
#include <predictive-unit/nn/layer.h>
#include <predictive-unit/util/proto-struct.h>
#include <predictive-unit/util/optional.h>
#include <predictive-unit/base.h>
#include <predictive-unit/log.h>

namespace NPredUnit {

	struct TStartSim: public TProtoStructure<NPredUnitPb::TStartSim> {
		TStartSim(const NPredUnitPb::TStartSim& m) {
			FillFromProto(m, 1, &SimulationTime);			
		}

		ui32 SimulationTime;
	};

	class TSimulator {
	public:
		static constexpr ui32 BatchSize = 4;
		static constexpr ui32 InputSize = 2;
		static constexpr ui32 FilterSize = 2;
		static constexpr ui32 LayerSize = 10;
		static constexpr ui32 BufferSize = 20;

		~TSimulator() {
			if (SimThread) {
				L_INFO << "Joining SimThread";
				SimThread->join();
			}
		}

		bool StartSimulationAsync(const NPredUnitPb::TStartSim& startSim) {
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

		static void StartSimulationImpl(TSimulator& sim, NPredUnitPb::TStartSim startSimProto) {
			TGuard lock(sim.SimMutex, std::adopt_lock);

			TStartSim startSim(startSimProto);

			L_INFO << "Got " << startSim.SimulationTime;
			
			TLayer<BatchSize, LayerSize, InputSize, FilterSize> layer;

			ui32 inputIter = 0;
			ui32 nCycle = 0;
			while (true) {
				if (inputIter >= sim.InputBuff.cols() - FilterSize) {
					L_INFO << "Cycle " << nCycle;
					inputIter = 0;
					++nCycle;
					break;
				}

				auto input = sim.InputBuff.block<BatchSize, InputSize*FilterSize>(0, inputIter);

				layer.Tick(input);

				L_INFO << input;

				inputIter += InputSize;
			}
			std::this_thread::sleep_for(std::chrono::seconds(10));  			

  			L_INFO << "Simulation is done";
		}

		TOptional<TThread> SimThread;
		TMutex SimMutex;
		TMatrix<BatchSize, InputSize*BufferSize> InputBuff;
	};
}