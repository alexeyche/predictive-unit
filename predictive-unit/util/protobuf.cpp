#include "protobuf.h"

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <fcntl.h>

namespace NPredUnit {

	void ReadProtoTextFromFile(const TString file, NPb::Message* message) {
		int fd = open(file.c_str(), O_RDONLY);
		ENSURE(fd >= 0, "Failed to open file " << file);

		NPb::io::FileInputStream fstream(fd);
	    fstream.SetCloseOnDelete(true);
		ENSURE(NPb::TextFormat::Parse(&fstream, message), "Failed to parse protobuf message from file: " << file);
	}

	void WriteProtoTextToFile(const NPb::Message& message, const TString file) {
		int fd = open(file.c_str(), O_CREAT | O_WRONLY, S_IRUSR | S_IWUSR);
		ENSURE(fd >= 0, "Failed to open file for write " << file);

		NPb::io::FileOutputStream fstream(fd);
	    fstream.SetCloseOnDelete(true);
	    ENSURE(NPb::TextFormat::Print(message, &fstream), "Failed to print protobuf message into file: " << file);
	}

	void ReadProtoText(const TString& messageStr, NPb::Message& message) {
		ENSURE(NPb::TextFormat::ParseFromString(messageStr, &message), "Failed to parse protobuf from string: \n" << messageStr);
	}

	TString ProtoTextToString(const NPb::Message& message) {
		TString s;
		NPb::TextFormat::PrintToString(message, &s);
		return s;
	}

}