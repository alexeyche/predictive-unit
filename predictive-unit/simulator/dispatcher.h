#pragma once

#include <predictive-unit/util/proto-struct.h>
#include <predictive-unit/protos/hostmap.pb.h>
#include <predictive-unit/log.h>
#include <predictive-unit/defaults.h>
#include <predictive-unit/util/string.h>

#include <Poco/Net/DNS.h>
#include <Poco/Net/StreamSocket.h>
#include <Poco/Net/SocketAddress.h>
#include <Poco/ThreadPool.h>

namespace NPredUnit {

	struct TSimRecord: public TProtoStructure<NPredUnitPb::TSimRecord> {
		TSimRecord(const NPredUnitPb::TSimRecord& sr) {
			FillFromProto(sr, NPredUnitPb::TSimRecord::kIdFieldNumber, &Id);
		}

		ui32 Id = 0;
	};

	struct THostRecord: public TProtoStructure<NPredUnitPb::THostRecord> {
		THostRecord(const NPredUnitPb::THostRecord& hr) {
			FillFromProto(hr, NPredUnitPb::THostRecord::kIdFieldNumber, &Id);
			FillFromProto(hr, NPredUnitPb::THostRecord::kHostFieldNumber, &Host);
			for (const auto& sr: hr.simrecord()) {
				SimRecord.push_back(TSimRecord(sr));
			}
		}

		ui32 Id = 0;
		TString Host;
		TVector<TSimRecord> SimRecord;
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

		static TPair<THostRecord, TSimRecord> GetHostAndSimRecord(ui32 hostId, ui32 simId, const THostMap& hostMap) {
			ENSURE(hostId < hostMap.HostRecord.size(), "Failed to find HostRecord with id #" << hostId);
			const THostRecord& hostRecord = hostMap.HostRecord.at(hostId);
			ENSURE(simId < hostRecord.SimRecord.size(), 
				"Failed to find SimRecord with id # " << simId << " on host " << hostRecord.Host);
			const TSimRecord& simRecord = hostRecord.SimRecord.at(simId);

			return MakePair(hostRecord, simRecord);
		}
		TDispatcher(const THostMap& hostMap)
			: HostMap(hostMap) 
		{
			TMultiMap<ui32, TPair<Poco::Net::SocketAddress, ui32>> reverseMap;
			L_INFO << "Setting up dispatch for host " << Poco::Net::DNS::hostName();

			for (const auto& conn: HostMap.Connection) {
				auto fromHostSimPair = GetHostAndSimRecord(conn.From.Host, conn.From.Sim, HostMap);
				const THostRecord& fromHostRecord = fromHostSimPair.first;
				const TSimRecord& fromSimRecord = fromHostSimPair.second;

				auto toHostSimPair = GetHostAndSimRecord(conn.To.Host, conn.To.Sim, HostMap);
				const THostRecord& toHostRecord = toHostSimPair.first;
				const TSimRecord& toSimRecord = toHostSimPair.second;

				if ((fromHostRecord.Host == "localhost") || (fromHostRecord.Host == "127.0.0.1") || (fromHostRecord.Host == Poco::Net::DNS::hostName())) {
					TVector<TString> hostAndPort = NStr::Split(toHostRecord.Host, ':', 1);
					
					ENSURE((hostAndPort.size() <= 2) && (hostAndPort.size() > 0), "Failed to parse host " << toHostRecord.Host);

					TString host = hostAndPort.at(0);
					ui32 port = TDefaults::ServerPort;

					L_INFO << "Connecting " << fromHostRecord.Host << ", sim #" << fromSimRecord.Id  << " to " << host << " sim # " << toSimRecord.Id; 
					
					if ((hostAndPort.size() == 2) && !hostAndPort[1].empty()) {
						port = std::stoi(hostAndPort.at(1));
					}

					reverseMap.emplace(
						fromSimRecord.Id,
						MakePair(
							Poco::Net::SocketAddress(host, port),
							toSimRecord.Id
						)
					);
				}
			
			}
		}

		template <typename Rows, typename Cols>
		class TDispatcherWorker: public Poco::Runnable {
		public:
			TDispatcherWorker(const Poco::Net::SocketAddress& dst, ui32 dstSimId, const TMatrix<Rows, Cols>& data)
				: Dst(dst)
				, DstSimId(dstSimId)
				, Data(data) 
			{
			}

			void run() override final {

			}

			Poco::Net::SocketAddress Dst;
			ui32 DstSimId;
			TMatrix<Rows, Cols> Data;
		};

		template <int Rows, int Cols>
		void ProcessActivationAsync(ui32 simId, const TMatrix<Rows, Cols>& activation) {
			ThreadPool
		}

	private:
		
		
		Poco::ThreadPool ThreadPool;
		THostMap HostMap;
	};


}