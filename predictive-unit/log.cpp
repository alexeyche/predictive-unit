#include "log.h"

namespace NPredUnit {


    TLog& TLog::Instance() {
        static TLog _inst;
        return _inst;
    }


}
