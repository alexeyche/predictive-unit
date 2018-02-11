#include "io.h"

namespace NPredUnit {


	template <>
	void TIOStream<TOutputStream>::Double(double& x) {
		writeBinary(x, Stream);
	}

	template <>
	void TIOStream<TInputStream>::Double(double& x) {
		readBinary(x, Stream);
	}

	template <>
	void TIOStream<TLog>::Double(double& x) {
		Stream.Info() << x << ";";
	}

	template <>
	template <>
	void TIOStream<TLog>::VarUInt(ui8& x) {
		Stream.Info() << static_cast<ui32>(x) << ";";
	}

}