#include "io.h"

#include <predictive-unit/util/string.h>

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
	

	template <>
	void TIOStream<TOutputStream>::Matrix(TMatrixD& x) {
		ui32 rows = x.rows();
		ui32 cols = x.cols();
		
		VarUInt(rows);
		VarUInt(cols);
		for (ui32 i=0; i<rows; ++i) {
			for (ui32 j=0; j<cols; ++j) {
				writeBinary(x(i, j), Stream);
			}
		}
	}


	template <>
	void TIOStream<TInputStream>::Matrix(TMatrixD& x) {
		ui32 rows;
		ui32 cols;
		
		VarUInt(rows);
		VarUInt(cols);
		x.resize(rows, cols);
		for (ui32 i=0; i<rows; ++i) {
			for (ui32 j=0; j<cols; ++j) {
				readBinary(x(i, j), Stream);
			}
		}
	}

	template <>
	void TIOStream<TLog>::Matrix(TMatrixD& x) {
		Stream.Info() << "Matrix, " << x.rows() << "x" << x.cols();		
		for (ui32 i=0; i<x.rows(); ++i) {
			NStr::TStringBuilder str;
			for (ui32 j=0; j<x.cols(); ++j) {
				str << x(i, j) << ",";
			}
			Stream.Info() << str.Str() << "\n";
		}
	}


}