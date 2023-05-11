#pragma once
#include "core/common/logging/logging.h"
#include "core/session/environment.h"

namespace ortki {
    class Environment;

    const onnxruntime::Environment &GetEnvironment();

/**
Static logging manager with a CLog based sink so logging macros that use the default logger will work
*/
    ::onnxruntime::logging::LoggingManager &DefaultLoggingManager();
}