#pragma once

#include <google/protobuf/message.h>
#include <predictive-unit/base.h>

namespace NPb = google::protobuf;
namespace NPbIO = google::protobuf::io;



namespace NPredUnit {

	void ReadProtoTextFromFile(const TString file, NPb::Message* message);

    void WriteProtoTextToFile(const NPb::Message& message, const TString file);

	void ReadProtoText(const TString& messageStr, NPb::Message& message);

	TString ProtoTextToString(const NPb::Message& message);


}