#pragma once

#include <Eigen/Dense>

#include <predictive-unit/base.h>
#include <predictive-unit/protos/matrix.pb.h>

namespace NPredUnit {

	void DeserializeMatrix(const NPredUnitPb::TMatrix& m, TMatrixD* dst);

}
