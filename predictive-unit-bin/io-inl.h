
#include <predictive-unit/log.h>


namespace NPredUnit {

	template <>
	template <typename T>
	void TIOStream<TOutputStream>::VarUInt(T& x) {
		writeVarUInt(x, Stream);
	}

	template <>
	template <typename T>
	void TIOStream<TInputStream>::VarUInt(T& x) {
		readVarUInt(x, Stream);
	}


	template <>
	template <typename T>
	void TIOStream<TLog>::VarUInt(T& x) {
		Stream.Info() << x << ";";
	}


	template <>
	template <typename T>
	void TIOStream<TOutputStream>::Array(TVector<T>& v) {
		ui32 size = v.size();
		VarUInt(size);
		for (auto& elem: v) {
			elem.Serial(std::forward<TIOStream<TOutputStream>>(*this));
		}
	}

	template <>
	template <typename T>
	void TIOStream<TInputStream>::Array(TVector<T>& v) {
		ui32 size;
		VarUInt(size);
		v.resize(size);
		for (auto& elem: v) {
			elem.Serial(std::forward<TIOStream<TInputStream>>(*this));
		}
	}

	template <>
	template <typename T>
	void TIOStream<TLog>::Array(TVector<T>& v) {
		for (auto& elem: v) {
			elem.Serial(std::forward<TIOStream<TLog>>(*this));
		}
	}

}