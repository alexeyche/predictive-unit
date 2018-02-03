#pragma once

#include <predictive-unit/base.h>


#include <google/protobuf/message.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>


namespace NPb = google::protobuf;
namespace NPbIO = google::protobuf::io;


namespace NPredUnit {

	template <typename TProto>
	class TProtoStructure {
	public:
		TProtoStructure() {
			const auto& message = TProto::default_instance();
	        Refl = message.GetReflection();
	        Descr = message.GetDescriptor();
		}

		template <typename T>
		void FillFromProto(const TProto& m, ui32 fieldNumber, T* dst) {
			const auto* fd = Descr->FindFieldByNumber(fieldNumber);
			FillField(m, fd, dst);
		}
		

		void FillField(const TProto& message, const NPb::FieldDescriptor* fd, ui32* dst) {
			*dst = Refl->GetUInt32(message, fd);
		}
	
		// template <>
		// void FillField(const NPb::FieldDescriptor* fd, const TProto& message, ui32* dst) {
			
		// }

		
		const NPb::Reflection* Refl;
       	const NPb::Descriptor* Descr;
	};
}

#include "proto-struct-inl.h"