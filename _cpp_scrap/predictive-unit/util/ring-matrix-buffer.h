#pragma once

#include <predictive-unit/base.h>
#include <Eigen/Dense>

namespace NPredUnit {


	//  |     |
	// [1, 2, 3]
	// 

	template <int Rows, int Cols, ui32 BufferSize>
	class TRingMatrixBuffer {
	public:
		TRingMatrixBuffer() {
			Data.resize(BufferSize);
		}

		void Push(const TMatrix<Rows, Cols>& m) {
			Data[Last++] = m;
			if (Last == BufferSize) {
				Last = 0;
			}
		}

		const TMatrix<Rows, Cols>&& Pop(ui32 idx) {
			return std::move(Data[idx]);
		}

		ui32 First = 0;
		ui32 Last = 0;

		TVector<TMatrix<Rows, Cols>> Data;

		// TMatrix<Rows, Cols*BufferSize> Data;
	};
}