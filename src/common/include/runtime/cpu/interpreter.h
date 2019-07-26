#pragma once
#include <runtime/interpreter.h>

namespace nncase
{
namespace runtime
{
    namespace cpu
    {
        class interpreter : public runtime::interpreter_base
        {
        public:
            using interpreter_base::interpreter_base;
        };
    }
}
}
