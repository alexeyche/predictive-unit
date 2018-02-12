#pragma once

#include <streambuf>

#include <predictive-unit/base.h>
#include <predictive-unit/log.h>

#include <Poco/Net/StreamSocket.h>

namespace NPredUnit {

	struct TMemBuf: std::streambuf {
	    TMemBuf(char* begin, ui32 size) {
	        this->setg(begin, begin, begin + size);
	    }
	};

	template <typename TStream>
	class TIOStream {
	public:
		TIOStream(TStream& stream)
			: Stream(stream)
		{}
	public:
		template <typename TVarUInt>
		void VarUInt(TVarUInt& x);

		void Double(double& x);
		void Matrix(TMatrixD& x);
		
		template <typename T>
		void Array(TVector<T>& v);
	private:
		TStream& Stream;
	};


	template <typename T>
	TIOStream<T> IOStream(T& stream) {
		return TIOStream<T>(stream);
	}


	inline TIOStream<TLog> NamedLogStream(TString name) {
		TLog::Instance().Info() << "Streaming `" << name << "`";
		return TIOStream<TLog>(TLog::Instance());
	}

	inline void writeVarUInt(ui64 x, TOutputStream& ostr) {
	    for (size_t i = 0; i < 9; ++i) {
	        uint8_t byte = x & 0x7F;
	        if (x > 0x7F)
	            byte |= 0x80;

	        ostr.put(byte);

	        x >>= 7;
	        if (!x) {
	        	return;
	        }
	    }
	}


	template <typename T>
	inline void writeBinary(const T & x, TOutputStream& buf) {
	    buf.write(reinterpret_cast<const char *>(&x), sizeof(x));
	}

	template <typename T>
	inline void readBinary(T& x, TInputStream& buf) {
	    buf.read(reinterpret_cast<char *>(&x), sizeof(x));
	}

	inline void readVarUInt(ui64& x, TInputStream& istr) {
	    x = 0;
	    for (size_t i = 0; i < 9; ++i) {
	        ui64 byte = istr.get();
	        x |= (byte & 0x7F) << (7 * i);

	        if (!(byte & 0x80)) {
	        	return;
	        }
	    }
	}


	template <typename T>
	inline void readVarUInt(T& x, TInputStream& istr) {
	    ui64 tmp;
	    readVarUInt(tmp, istr);
	    x = tmp;
	}

	template <typename T>
	void ReadFromSocket(Poco::Net::StreamSocket& sck, std::function<void(TInputStream&)> callback) {
		char rawBuffer[sizeof(T)];
		sck.receiveBytes(&rawBuffer[0], sizeof(T));
		
		TMemBuf buffer(&rawBuffer[0], sizeof(T));
		TInputStream io(&buffer);
		callback(io);
	}

}

#include "io-inl.h"