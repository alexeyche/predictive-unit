

#include <predictive-unit/log.h>
#include <predictive-unit/protos/model-config.pb.h>
#include <predictive-unit/model.h>

using namespace NPredUnit;


int main(int argc, char** argv) {
	NPredUnitPb::TModelConfig mess;
	mess.set_batchsize(1);

	TModel<100, 100, 10, 1> model;

	L_INFO << "Test!: " << mess.DebugString();
}
