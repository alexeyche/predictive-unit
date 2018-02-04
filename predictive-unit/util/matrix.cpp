#include "matrix.h"

#include <predictive-unit/log.h>


namespace NPredUnit {

	void DeserializeMatrix(const NPredUnitPb::TMatrix& m, TMatrixD* dst) {
		if (m.row_size() == 0) {
			*dst = TMatrixD(0,0);
		}
		
		*dst = TMatrixD(m.row_size(), m.row(0).data_size());

		for (ui32 rowId=0; rowId<ToUi32(dst->rows()); ++rowId) {
			for (ui32 colId=0; colId<ToUi32(dst->cols()); ++colId) {
				(*dst)(rowId, colId) = m.row(rowId).data(colId);
			}
		}
	}

	
}