
#include "dispatcher.h"

#include <predictive-unit/log.h>
#include <predictive-unit/protos/layer-config.pb.h>
#include <predictive-unit/layer.h>


using namespace NPredUnit;


int main(int argc, char** argv) {

	TDispatcher dispatcher(8080);

	dispatcher.Run();

	return 0;
}