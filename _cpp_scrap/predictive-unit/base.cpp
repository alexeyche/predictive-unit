#include "base.h"

namespace NPredUnit {

	ui32 ToUi32(int v) {
		return static_cast<ui32>(v);
	}

	void RunLock(TAtomicFlag& atomicFlag, std::function<void()> cb) {
		while (atomicFlag.test_and_set(std::memory_order_acquire)) {}
		cb();
		atomicFlag.clear(std::memory_order_release);
	}
}
