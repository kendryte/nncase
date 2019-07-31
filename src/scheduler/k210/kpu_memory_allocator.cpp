#include <ir/op_utils.h>
#include <runtime/k210/k210_runtime_op_utility.h>
#include <scheduler/k210/kpu_memory_allocator.h>

using namespace nncase;
using namespace nncase::scheduler;
using namespace nncase::scheduler::k210;

kpu_memory_allocator::kpu_memory_allocator()
    : memory_allocator(64, 2 * 1024 * 1024)
{
}

size_t kpu_memory_allocator::get_bytes(datatype_t type, const ir::shape_t &shape) const
{
    if (type != dt_uint8)
        throw std::invalid_argument("KPU only support uint8 data");
    return runtime::k210::get_kpu_bytes(ir::to(shape));
}
