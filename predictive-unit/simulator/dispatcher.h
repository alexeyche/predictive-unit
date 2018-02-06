#pragma once

#include <predictive-unit/util/proto-struct.h>
#include <predictive-unit/protos/hostmap.pb.h>
#include <predictive-unit/protos/messages.pb.h>
#include <predictive-unit/protocol.h>
#include <predictive-unit/log.h>
#include <predictive-unit/defaults.h>
#include <predictive-unit/util/string.h>

#include <Poco/Net/DNS.h>
#include <Poco/Net/StreamSocket.h>
#include <Poco/Net/SocketAddress.h>
#include <Poco/ThreadPool.h>
#include <Poco/Mutex.h>

namespace NPredUnit {

	struct THostRecord: public TProtoStructure<NPredUnitPb::THostRecord> {
		THostRecord(const NPredUnitPb::THostRecord& hr) {
			FillFromProto(hr, NPredUnitPb::THostRecord::kIdFieldNumber, &Id);
			FillFromProto(hr, NPredUnitPb::THostRecord::kHostFieldNumber, &Host);
			for (const auto& sr: hr.simconfig()) {
				SimConfig.push_back(TSimConfig(sr));
			}
		}

		ui32 Id = 0;
		TString Host;
		TVector<TSimConfig> SimConfig;
	};

	struct TConnectionInstance: public TProtoStructure<NPredUnitPb::TConnectionInstance> {
		TConnectionInstance(const NPredUnitPb::TConnectionInstance& cn) {
			FillFromProto(cn, NPredUnitPb::TConnectionInstance::kHostFieldNumber, &Host);
			FillFromProto(cn, NPredUnitPb::TConnectionInstance::kSimFieldNumber, &Sim);
		}

		ui32 Host = 0;
		ui32 Sim = 0;
	};

	struct TConnection: public TProtoStructure<NPredUnitPb::TConnection> {
		TConnection(const NPredUnitPb::TConnection& cn)
			: From(cn.from())
			, To(cn.to()) 
		{
		}

		TConnectionInstance From;
		TConnectionInstance To;
	};

	struct THostMap: public TProtoStructure<NPredUnitPb::THostMap> {
		THostMap(const NPredUnitPb::THostMap& hostMap) {
			for (const auto& hr: hostMap.hostrecord()) {
				HostRecord.push_back(THostRecord(hr));
			}
			
			for (const auto& cn: hostMap.connection()) {
				Connection.push_back(TConnection(cn));
			}
		}

		TVector<THostRecord> HostRecord;
		TVector<TConnection> Connection;
	};

	
	class TDispatcher {
	public:
		
		static TPair<THostRecord, TSimConfig> GetHostAndSimConfig(ui32 hostId, ui32 simId, const THostMap& hostMap) {
			ENSURE(hostId < hostMap.HostRecord.size(), "Failed to find HostRecord with id #" << hostId);
			const THostRecord& hostRecord = hostMap.HostRecord.at(hostId);
			ENSURE(simId < hostRecord.SimConfig.size(), 
				"Failed to find SimConfig with id # " << simId << " on host " << hostRecord.Host);
			const TSimConfig& simRecord = hostRecord.SimConfig.at(simId);

			return MakePair(hostRecord, simRecord);
		}
		
		TDispatcher(const THostMap& hostMap)
			: HostMap(hostMap) 
		{
			L_INFO << "Setting up dispatch for host " << Poco::Net::DNS::hostName();

			for (const auto& conn: HostMap.Connection) {
				auto fromHostSimPair = GetHostAndSimConfig(conn.From.Host, conn.From.Sim, HostMap);
				const THostRecord& fromHostRecord = fromHostSimPair.first;
				const TSimConfig& fromSimConfig = fromHostSimPair.second;

				auto toHostSimPair = GetHostAndSimConfig(conn.To.Host, conn.To.Sim, HostMap);
				const THostRecord& toHostRecord = toHostSimPair.first;
				const TSimConfig& toSimConfig = toHostSimPair.second;

				if ((fromHostRecord.Host == "localhost") || (fromHostRecord.Host == "127.0.0.1") || (fromHostRecord.Host == Poco::Net::DNS::hostName())) {
					TVector<TString> hostAndPort = NStr::Split(toHostRecord.Host, ':', 1);
					
					ENSURE((hostAndPort.size() <= 2) && (hostAndPort.size() > 0), "Failed to parse host " << toHostRecord.Host);

					TString host = hostAndPort.at(0);
					ui32 port = TDefaults::ServerPort;

					L_INFO << "Connecting " << fromHostRecord.Host << ", sim #" << fromSimConfig.Id  << " to " << host << " sim # " << toSimConfig.Id; 
					
					if ((hostAndPort.size() == 2) && !hostAndPort[1].empty()) {
						port = std::stoi(hostAndPort.at(1));
					}

					ReverseMap.emplace(
						fromSimConfig.Id,
						MakePair(
							Poco::Net::SocketAddress(host, port),
							toSimConfig.Id
						)
					);
				}
			
			}

			Mutex.clear();
			
			MainThread = TThread(
				Run,
				std::ref(*this)
			);
		}

		~TDispatcher() {
			Stop();
		}

		void AddQueueToDispatch(ui32 simId, TMatrixRWQ* queue) {
			RunLock(Mutex, [&]() {
				SimQueues.push_back(MakePair(simId, queue));
			});
			// Poco::FastMutex::ScopedLock lock(Mutex);
			// SimQueues.push_back(MakePair(simId, queue));
		}

		static void SendData(const TMatrixD& data, const Poco::Net::SocketAddress& address, ui32 simId) {
			NPredUnitPb::TInputData dataMessage;
			dataMessage.set_simid(simId);
			SerializeMatrix(data, dataMessage.mutable_data());

			Poco::Net::StreamSocket socket;
			socket.connect(address);
			WriteHeaderAndProtobufMessageToSocket(dataMessage, NPredUnitPb::TMessageType::INPUT_DATA, &socket);
			socket.shutdown();
		}

		static void Run(TDispatcher& dispatcher) {
			while (true) {
				while (dispatcher.Mutex.test_and_set(std::memory_order_acquire)) {}
				if (dispatcher.NeedToStop) break;
				dispatcher.Mutex.clear(std::memory_order_release);

				// L_INFO << "Dispatcher: Running ..";
				// {
				// 	L_INFO << "Dispatcher: Lock";

				// 	Poco::FastMutex::ScopedLock lock(dispatcher.Mutex);
				// 	if (dispatcher.NeedToStop) {
				// 		break;
				// 	}
				// }

				// L_INFO << "Dispatcher: Check queue";
				for (const auto& sq: dispatcher.SimQueues) {
					while (sq.second->size_approx()>0) {
						TMatrixD data;
						if (sq.second->try_dequeue(data)) {
							L_INFO << "Dispatcher: Dequeuing from #sim id " << sq.first << " buffer, approx size: " << sq.second->size_approx();
							auto range = dispatcher.ReverseMap.equal_range(sq.first);
							for (auto it = range.first; it != range.second; ++it) {
								L_INFO << "Dispatcher: Sending data for " << it->second.first.toString() << ", #sim id " << it->second.second;
								SendData(data, it->second.first, it->second.second);
							}
							
						}	
					}
					
				}
			}
		}

		void Stop() {
			{
				if (!NeedToStop) {
					RunLock(Mutex, [&](){
						NeedToStop = true;
					});
				}
				// Poco::FastMutex::ScopedLock lock(Mutex);
				// NeedToStop = true;
			}
			if (MainThread.joinable()) {
				MainThread.join();
			}
		}

	private:

		TThread MainThread;		
		THostMap HostMap;

		// mutable Poco::FastMutex Mutex;
		TAtomicFlag Mutex;

		bool NeedToStop = false;
		TMultiMap<ui32, TPair<Poco::Net::SocketAddress, ui32>> ReverseMap;
		TVector<TPair<ui32, TMatrixRWQ*>> SimQueues;
	};


}