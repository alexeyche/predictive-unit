#pragma once

#include "error.h"

#include <memory>
#include <string>
#include <vector>
#include <deque>
#include <tuple>
#include <thread>
#include <mutex>
#include <complex>
#include <set>
#include <map>
#include <thread>
#include <cstdint>
#include <predictive-unit/contrib/rwq/readerwriterqueue.h>

#include <Eigen/Dense>

namespace NPredUnit {

	template <typename T>
	using TUniquePtr = std::unique_ptr<T>;

	template <typename T>
	using TSharedPtr = std::shared_ptr<T>;


	template <typename X, typename Y>
	using TPair = std::pair<X, Y>;

	using TString = std::string;

	#define JOIN(X, Y) X##Y

	#define GENERATE_UNIQUE_ID(N) JOIN(N, __LINE__)

	template <typename X, typename Y>
	TPair<X, Y> MakePair(X x, Y y) {
		return TPair<X, Y>(x, y);
	}

    template <typename T>
    using TVector = std::vector<T>;

    using ui32 = size_t;
	using i32 = int;
	using ui64 = unsigned long;
	using ui8 = uint8_t;
		
    //template< class... Types>
    //using Tie = std::tie<Types& ...>;


	template< typename T1, typename T2>
	class TTie {
	public:
	 	TTie(T1 &first,T2 &second)
	  		: First(first)
	  		, Second(second)
	  	{
	  	}

	 	TPair<T1, T2> const & operator = (TPair<T1, T2> const &rhs) {
		    First = rhs.first;
		    Second = rhs.second;
		    return rhs;
		}

	private:
	  	void operator=(TTie const &);
	  	T1 &First;
	  	T2 &Second;
	};

	template <typename T1, typename T2>
	inline TTie<T1,T2> Tie(T1 &first, T2 &second)
	{
	  return TTie<T1, T2>(first, second);
	}

	template <typename T>
	TSharedPtr<T> MakeShared(T *ptr) {
		return TSharedPtr<T>(ptr);
	}

	using TOutputStream = std::ostream;
	using TInputStream = std::istream;

	template <typename T>
	using TRefWrap = std::reference_wrapper<T>;

	using TMutex = std::mutex;

	using TGuard = std::lock_guard<TMutex>;

	using TUniqueLock = std::unique_lock<TMutex>;

	struct TEmpty {};

	struct TTime {
	    TTime(double dt)
	    	: T(0)
	    	, Dt(dt)
	    {
	    }

	    void operator ++() {
	        T += Dt;
	    }
	    bool operator<(const double &dur) const {
	        return T < dur;
	    }
	
	    double T;
	    double Dt;
	};

	using TComplex = std::complex<double>;

	template <typename T>
	using TDeque = std::deque<T>;

	template <typename T>
	using TFunction = std::function<T>;

	template <typename T>
	using TSet = std::set<T>;


	template <typename K, typename V>
	using TMap = std::map<K, V>;

	template <int rows, int cols>
	using TMatrix = Eigen::Matrix<float, rows, cols>;

	
	using TMatrixD = TMatrix<Eigen::Dynamic, Eigen::Dynamic>;
	
	using TThread = std::thread;
			
	ui32 ToUi32(int v);

	using TAtomicFlag = std::atomic_flag;

	void RunLock(TAtomicFlag& atomicFlag, std::function<void()> cb);

	template <typename K, typename V>
	using TMultiMap = std::multimap<K, V>;

	template <typename T, size_t MAX_BLOCK_SIZE = 512>
	using TReaderWriterQueue = moodycamel::ReaderWriterQueue<T, MAX_BLOCK_SIZE>;

	using TMatrixRWQ = TReaderWriterQueue<TMatrixD>;
	// using TMatrixRWQ = std::queue<TMa

} // namespace NPredUnit
