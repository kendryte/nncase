#pragma once
#include "k210_sim_types.h"
#include <runtime/interpreter.h>

namespace nncase
{
namespace targets
{
    namespace k210
    {
        class interpreter : public runtime::interpreter_base
        {
        public:
            using interpreter_base::memory_at;

            interpreter();

        protected:
            xtl::span<uint8_t> memory_at(const memory_range &range) const noexcept override;

        private:
#if NNCASE_TARGET_K210_SIMULATOR
            std::unique_ptr<uint8_t[]> kpu_mem_;
#endif
        };
    }
}
}
