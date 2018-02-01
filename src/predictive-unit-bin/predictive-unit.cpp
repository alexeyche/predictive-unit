
#include "dispatcher.h"
#include "client.h"
#include "defaults.h"


#include <predictive-unit/log.h>
#include <predictive-unit/protos/layer-config.pb.h>
#include <predictive-unit/layer.h>
#include <predictive-unit/util/maybe.h>
#include <predictive-unit/util/argument.h>

using namespace NPredUnit;


int server(const TVector<TString>& argsVec) {
	ui32 port = TDefaults::ServerPort;
	bool help = false;

	auto args = ArgumentSet(
		Argument("--port", "-p", port, "the port for TCPServer, default 8080"),
		Argument("--help", "-h", help, "This option will print this menu", /*required*/ false, /*stopProcessingAfterMatch*/ true)
	);

	if (!args.TryParse(argsVec)) {
		return 1;
	}

	if (help) {
		std::cout << "server\n";
		args.GenerateHelp(std::cout);
		return 1;
	}

	TDispatcher dispatcher(port);

	dispatcher.Run();

	return 0;
}

int client(const TVector<TString>& argsVec) {
	bool help = false;
	TString server = TDefaults::ServerHost;
	ui32 port = TDefaults::ServerPort;

	auto args = ArgumentSet(
		Argument("--help", "-h", help, "This option will print this menu", /*required*/ false, /*stopProcessingAfterMatch*/ true),
		Argument("--server", "-s", server, "Host of the server"),
		Argument("--port", "-p", port, "the port for TCPServer, default 8080")
	);
	
	if (!args.TryParse(argsVec)) {
		return 1;
	}

	if (help) {
		std::cout << "client\n";
		args.GenerateHelp(std::cout);
		return 1;
	}

	TClient cli(server, port);
	cli.SendData();

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