
from varint import decode_stream as readVarUInt
from varint import encode as encode_varint
import struct

def readDouble(buf):
	return struct.unpack("d", buf.read(8))[0]

def writeDouble(v, buf):
	buf.write(struct.pack("d", v))


def writeVarUInt(v, buf):
	b = encode_varint(v)
	buf.write(b)

