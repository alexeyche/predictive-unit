
#include "dispatcher.h"
#include "arg.h"

#include <predictive-unit/log.h>
#include <predictive-unit/protos/layer-config.pb.h>
#include <predictive-unit/layer.h>
#include <predictive-unit/maybe.h>

#include <Poco/Util/Option.h>
#include <Poco/Util/OptionSet.h>
#include <Poco/Util/OptionProcessor.h>
#include <Poco/Util/HelpFormatter.h>

using namespace NPredUnit;


int main(int argc, const char** argv) {
	Poco::Util::OptionSet mainOptions;
	
	
	ui32 port = 8080;
	bool help = false;
	
	auto args = ArgumentSet(
		Argument("--port", "-p", port),
		Argument("--help", "-h", help)
	);

	args.Parse(argc, argv);
	
	// TArgSet(
	// 	TArg("--port", "-p", port)
	// );

	// for (auto it=args.begin(); it != args.end(); ++it) {
	// 	if (*it == "--port" || *it == "-p") {
	// 		it = args.erase(it);
	// 		ENSURE(it != args.end(), "Need option for port");
	// 		port = std::stoi(*it);
	// 	}
	// }
	


	TDispatcher dispatcher(port);

	dispatcher.Run();

	return 0;
}