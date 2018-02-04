
#include <predictive-unit/base.h>
#include <predictive-unit/util/ring-matrix-buffer.h>
#include <predictive-unit/log.h>


using namespace NPredUnit;

void TestRingMatrixBuffer() {
	constexpr int rows = 10;
	constexpr int cols = 10;
	constexpr ui32 bufferSize = 10;
	
	TRingMatrixBuffer<rows, cols, bufferSize> RMBuff;

	TVector<TMatrix<rows, cols>> data;
	for (ui32 id=0; id < 20; ++id) {
		data.push_back(TMatrix<rows, cols>::Random());
	}

	for (ui32 id=0; id < 15; ++id) {
		RMBuff.Push(data[id]);
	}

	// for (ui32 id=0; id < 10; ++id) {
	// 	double diff = (RMBuff.Pop() - data[id+5]).norm();
	// 	ENSURE(diff == 0.0, "Test failed for id " << id);
	// }

	
	
}


int main(int argc, char** argv) {
	TestRingMatrixBuffer();

	return 0;
}
