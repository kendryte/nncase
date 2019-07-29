#include <runtime/k210/interpreter.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::k210;

interpreter::interpreter()
#if NNCASE_TARGET_K210_SIMULATOR
    : kpu_mem_(std::make_unique<uint8_t[]>(2 * 1024 * 1024))
#endif
{
#if !NNCASE_TARGET_K210_SIMULATOR
    kpu->interrupt_clear.reg = 7;
    kpu->interrupt_mask.reg = 7;
    kpu->fifo_threshold.reg = 10 | (1 << 4);
    kpu->eight_bit_mode.reg = 1;

    plic_set_priority(IRQN_AI_INTERRUPT, 1);
#endif
}

xtl::span<uint8_t> interpreter::memory_at(const memory_range &range) const noexcept
{
    if (range.memory_type == mem_k210_kpu)
    {
        uintptr_t base =
#if NNCASE_TARGET_K210_SIMULATOR
            (uintptr_t)kpu_mem_.get();
#else
            (uintptr_t)AI_IO_BASE_ADDR;
#endif
        return { reinterpret_cast<uint8_t *>(base + range.start), range.size };
    }

    return interpreter_base::memory_at(range);
}
