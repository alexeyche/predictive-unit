#pragma once

#include <Eigen/Dense>

#include <predictive-unit/base.h>
#include <predictive-unit/protos/matrix.pb.h>

namespace NPredUnit {

	void DeserializeMatrix(const NPredUnitPb::TMatrix& m, TMatrixD* dst);

	inline void SerializeMatrix(const TMatrixD& m, NPredUnitPb::TMatrix* dst) {
		if (m.rows() == 0) {
			return;
		}

		for (ui32 i=0; i<ToUi32(m.rows()); ++i) {
			auto* row = dst->add_row();
			for (ui32 j=0; j<ToUi32(m.cols()); ++j) {
				row->add_data(m(i,j));
			}
		}
	}
}
