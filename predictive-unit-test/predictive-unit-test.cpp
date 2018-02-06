
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

		auto* sc0 = hr->add_simconfig();
		sc0->set_id(0);
		sc0->set_simulationtime(10000);
		sc0->mutable_layerconfig()->set_layersize(100);
		sc0->mutable_layerconfig()->set_inputsize(10);
		sc0->mutable_layerconfig()->set_batchsize(5);
		sc0->mutable_layerconfig()->set_filtersize(1);


		auto* sc1 = hr->add_simconfig();
		sc1->set_id(1);
		sc1->set_simulationtime(10000);
		sc1->mutable_layerconfig()->set_layersize(100);
		sc1->mutable_layerconfig()->set_inputsize(100);
		sc1->mutable_layerconfig()->set_batchsize(5);
		sc1->mutable_layerconfig()->set_filtersize(1);

		auto* conn = hostMapPb.add_connection();
		conn->mutable_from()->set_host(0);
		conn->mutable_from()->set_sim(0);
		conn->mutable_to()->set_host(0);
		conn->mutable_to()->set_sim(1);
	}

	THostMap hostMap(hostMapPb);

	NPredUnitPb::TStartSim defaultSimConfigPb;
	TStartSim defaultSimConfig(defaultSimConfigPb);

	TSimulator sim(jobsNum, hostMap);

	TMessageServer server(sim, 8080);
	server.RunAsync();

	sim.StartSimulationsAsync();

	std::this_thread::sleep_for(std::chrono::seconds(2));

	server.Stop();
}


int main(int argc, char** argv) {
	TestBaseTalk();

	return 0;
}
