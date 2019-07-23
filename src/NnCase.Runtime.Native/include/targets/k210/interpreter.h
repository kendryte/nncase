#pragma once
#include "k210_sim_types.h"
#include <runtime/interpreter.h>

namespace nncase
{
namespace targets
{
    namespace k210
    {
        struct k210_interpreter_context
        {
            runtime::interpreter_base *interpreter;
            runtime::interpreter_step_t step;
        };

        class interpreter : public runtime::interpreter_base
        {
        public:
            using interpreter_base::memory_at;

            interpreter();

#if !NNCASE_TARGET_K210_SIMULATOR

            dmac_channel_number_t dma_ch() const noexcept { return dma_ch_; }
            void dma_ch(dmac_channel_number_t dma_ch) noexcept { dma_ch_ = dma_ch; }
            k210_interpreter_context &context() noexcept { return context_; }
#endif

        protected:
            xtl::span<uint8_t> memory_at(const memory_range &range) const noexcept override;

        private:
#if NNCASE_TARGET_K210_SIMULATOR
            std::unique_ptr<uint8_t[]> kpu_mem_;
#else
            dmac_channel_number_t dma_ch_;
            k210_interpreter_context context_;
#endif
        };
    }
}
}
