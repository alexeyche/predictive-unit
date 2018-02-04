
#include <fstream>

#include <predictive-unit/defaults.h>
#include <predictive-unit/simulator/message-server.h>
#include <predictive-unit/simulator/dispatcher.h>
#include <predictive-unit/simulator/simulator.h>
#include <predictive-unit/util/protobuf.h>
#include <predictive-unit/protos/hostmap.pb.h>

#include "client.h"


#include <predictive-unit/log.h>
#include <predictive-unit/util/maybe.h>
#include <predictive-unit/util/argument.h>
#include <predictive-unit/util/string.h>

using namespace NPredUnit;
using namespace NPredUnit::NStr;


int server(const TVector<TString>& argsVec) {
	ui32 port = TDefaults::ServerPort;
	bool help = false;
	TOptional<int> startSim;
	ui32 jobsNum = 1;
	TString startData;
	TString hostMapFile;

	auto args = ArgumentSet(
		Argument("--port", "-p", port, "the port for TCPServer, default 8080"),
		Argument("--jobs", "-j", jobsNum, "Maximum jobs number"),
		Argument("--host-map", "-m", hostMapFile, "Host map schema"),
		Argument("--help", "-h", help, "This option will print this menu", /*required*/ false, /*stopProcessingAfterMatch*/ true),
		Argument("--start-sim", "-s", startSim, "This option will force server start simulation with pointed duration"),
		Argument("--start-data", "-d", startData, "Points to file with TInputData")
	);

	if (!args.TryParse(argsVec)) {
		return 1;
	}

	if (help) {
		std::cout << "server\n";
		args.GenerateHelp(std::cout);
		return 1;
	}

	// HostMap
	ENSURE(!hostMapFile.empty(), "Host map must be pointed out as argument");
	NPredUnitPb::THostMap hostMapPb;
	ReadProtoTextFromFile(hostMapFile, &hostMapPb);

	L_INFO << hostMapPb.DebugString();
	
	// Dispatcher
	THostMap hostMap(hostMapPb);
	TDispatcher dispatcher(hostMap);

	// Simulator
	NPredUnitPb::TStartSim defaultSimConfig;
	TSimulator sim(jobsNum, dispatcher);
	
	{
		if (!startData.empty()) {
			std::ifstream ifile(startData);
			ReadProtobufMessage(ifile, defaultSimConfig.mutable_inputdata());
		}

		if (startSim) {
			TStartSim startSimConfig(defaultSimConfig);
			startSimConfig.SimulationTime = *startSim;
			sim.StartSimulationAsync(startSimConfig);
		}
	}

	// Message server
	TMessageServer server(sim, port);

	server.Run();

	return 0;
}

int client(const TVector<TString>& argsVec) {
	bool help = false;
	TString server = TDefaults::ServerHost;
	ui32 port = TDefaults::ServerPort;
	
	TString protobufSourceToRead;

	auto args = ArgumentSet(
		Argument("--help", "-h", help, "This option will print this menu", /*required*/ false, /*stopProcessingAfterMatch*/ true),
		Argument("--server", "-s", server, "Host of the server"),
		Argument("--port", "-p", port, "the port for TCPServer, default 8080"),
		Argument("--file", "-f", protobufSourceToRead, TStringBuilder() << "Read protobuf file, default: /dev/stdin")
	);
	
	if (!args.TryParse(argsVec)) {
		return 1;
	}

	if (help) {
		std::cout << "client\n";
		args.GenerateHelp(std::cout);
		return 1;
	}
	

	// if (protobufSourceToRead.empty()) {
	// 	ReadProtobufMessage(std::cin);
	// } else {
	// 	std::ifstream ifile(protobufSourceToRead);
	// 	ENSURE(ifile, "Failed to open file: " << protobufSourceToRead);
	// 	ReadProtobufMessage(ifile);
	// }

	// TClient cli(server, port);
	// cli.SendData();

	return 0;
}


typedef int(*TMainEntryFun)(const TVector<TString>& args);


void GenerateHelpWithSubPrograms(const TMap<TString, TMainEntryFun>& subPrograms) {
	for (const auto& subp: subPrograms) {
		subp.second({"--help"});
		std::cout << "\n";
	}
}

int main(int argc, const char** argv) {
	TMap<TString, TMainEntryFun> subProgramEntries = {
		{"server", &server},
		{"client", &client}
	};

	TVector<TString> argsVec;
	TVector<TString> subProgramArgsVec;

	TMap<TString, TMainEntryFun>::iterator subProgramToCallIt = subProgramEntries.end();
	int i = 1;
	while (i < argc) {
		auto subProgramIt = subProgramEntries.find(argv[i]);
		if (subProgramIt != subProgramEntries.end()) {
			subProgramToCallIt = subProgramIt;
		} else
		if (subProgramToCallIt == subProgramEntries.end()) {
			argsVec.push_back(argv[i]);
		} else {
			subProgramArgsVec.push_back(argv[i]);
		}
				
		++i;
	}
	
	bool help = false;
	auto args = ArgumentSet(
		Argument("--help", "-h", help, "This option will print this menu", /*required*/ false, /*stopProcessingAfterMatch*/ true)
	);


	if (!args.TryParse(argsVec)) {
		return 1;
	}

	if ((argsVec.empty() && (subProgramToCallIt == subProgramEntries.end())) || help) {
		std::cout << "\nCommon options to the program:\n";
		args.GenerateHelp(std::cout);
		std::cout << "\n";
		GenerateHelpWithSubPrograms(subProgramEntries);
		return 1;
	}

	if (subProgramToCallIt == subProgramEntries.end()) {
		L_ERROR << "Failed to find subprogram to call";
		return 1;
	}

	return subProgramToCallIt->second(subProgramArgsVec);
}