#pragma once

#include <predictive-unit/base.h>
#include <predictive-unit/log.h>

#include <queue>

namespace NPredUnit {



	template <typename T>
	class TRWQueue {
	public:
		void Push(const T& elem) {
			RunLock(Mutex, [&](){
				Queue.push(elem);
			});
		}

		bool TryDequeue(T* dst) {
			bool success = false;
			RunLock(Mutex, [&](){
				if (!Queue.empty()) {
					*dst = Queue.front();
					success = true;
					Queue.pop();
				}
			});
			return success;
		}

		ui32 QueueSize() const {
			ui32 size;
			RunLock(Mutex, [&](){
				size = Queue.size();
			});
			return size;
		}

	private:
		std::queue<T> Queue;
		mutable TAtomicFlag Mutex;
	};

}