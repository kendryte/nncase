#include <targets/k210/interpreter.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::targets::k210;

interpreter::interpreter()
    : kpu_mem_(std::make_unique<uint8_t[]>(2 * 1024 * 1024))
{
}

xtl::span<uint8_t> interpreter::memory_at(const memory_range &range) const noexcept
{
    if (range.memory_type == mem_k210_kpu)
    {
        uintptr_t base =
#if NNCASE_TARGET_K210_SIMULATOR
            (uintptr_t)kpu_mem_.get();
#endif
        return { reinterpret_cast<uint8_t *>(base + range.start), range.size };
    }

    return interpreter_base::memory_at(range);
}
