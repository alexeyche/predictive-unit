
#include "message-server.h"

#include <fstream>

#include <predictive-unit/base.h>
#include <predictive-unit/log.h>
#include <predictive-unit/util/string.h>
#include <predictive-unit/defaults.h>
#include <predictive-unit/util/argument.h>

using namespace NPredUnit;
using namespace NPredUnit::NStr;



int main(int argc, const char** argv) {
	ui32 port = TDefaults::ServerPort;
	bool help = false;

	auto args = ArgumentSet(
		Argument("--port", "-p", port, "the port for TCPServer, default 8080"),
		Argument("--help", "-h", help, "This option will print this menu", /*required*/ false, /*stopProcessingAfterMatch*/ true)
	);

	if (!args.TryParse(argc, argv)) {
		return 1;
	}

	if (help) {
		args.GenerateHelp(std::cout);
		return 1;
	}

	TMessageServer server(port);

	server.Run();
	
	// TNetworkConfig config;
	// config.LayerConfigs.push_back(TLayerConfig());
	
	// config.Serial(NamedLogStream("LayerConfig:"));

	return 0;
}