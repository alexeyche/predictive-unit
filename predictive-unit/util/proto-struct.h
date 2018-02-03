#pragma once

#include <predictive-unit/base.h>
#include <predictive-unit/util/matrix.h>
#include <predictive-unit/protos/matrix.pb.h>


#include <google/protobuf/message.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>


namespace NPb = google::protobuf;
namespace NPbIO = google::protobuf::io;


namespace NPredUnit {

	template <typename TProto>
	class TProtoStructure {
	public:
		using TTemplateProto = TProto;

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
	
		void FillField(const TProto& message, const NPb::FieldDescriptor* fd, i32* dst) {
			*dst = Refl->GetInt32(message, fd);
		}

		void FillField(const TProto& message, const NPb::FieldDescriptor* fd, double* dst) {
			*dst = Refl->GetDouble(message, fd);
		}


		void FillField(const TProto& message, const NPb::FieldDescriptor* fd, TMatrixD* dst) {
			const NPredUnitPb::TMatrix& mat = dynamic_cast<const NPredUnitPb::TMatrix&>(Refl->GetMessage(message, fd));
			DeserializeMatrix(mat, dst);
		}

		const NPb::Reflection* Refl;
       	const NPb::Descriptor* Descr;
	};

}
