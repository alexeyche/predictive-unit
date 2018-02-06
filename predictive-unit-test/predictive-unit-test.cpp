
#include <predictive-unit/base.h>
#include <predictive-unit/simulator/dispatcher.h>
#include <predictive-unit/simulator/message-server.h>
#include <predictive-unit/simulator/simulator.h>
#include <predictive-unit/log.h>


using namespace NPredUnit;


void TestBaseTalk() {
	ui32 jobsNum = 2;

	NPredUnitPb::THostMap hostMapPb;

	{
		auto* hr = hostMapPb.add_hostrecord();
		hr->set_id(0);
		hr->set_host("127.0.0.1");
		hr->add_simrecord()->set_id(0);
		hr->add_simrecord()->set_id(1);
		
		auto* conn = hostMapPb.add_connection();
		conn->mutable_from()->set_host(0);
		conn->mutable_from()->set_sim(0);
		conn->mutable_to()->set_host(0);
		conn->mutable_to()->set_sim(1);
	}

	THostMap hostMap(hostMapPb);


	TDispatcher dispatcher(hostMap);

	NPredUnitPb::TStartSim defaultSimConfigPb;
	TStartSim defaultSimConfig(defaultSimConfigPb);
	defaultSimConfig.SimulationTime = 10001;

	TSimulator sim(jobsNum, dispatcher);

	TMessageServer server(sim, 8080);
	server.RunAsync();

	std::this_thread::sleep_for(std::chrono::seconds(1));

	L_INFO << "Starting 1 sim";
	sim.StartSimulationAsync(defaultSimConfig);
	L_INFO << "Starting 2 sim";
	defaultSimConfig.SimId = 1;
	sim.StartSimulationAsync(defaultSimConfig);

	std::this_thread::sleep_for(std::chrono::seconds(2));

	dispatcher.Stop();
	server.Stop();
}


int main(int argc, char** argv) {
	TestBaseTalk();

	return 0;
}
