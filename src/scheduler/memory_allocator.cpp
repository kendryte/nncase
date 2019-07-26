#include <ir/op_utils.h>
#include <scheduler/memory_allocator.h>
#include <stdexcept>

using namespace nncase;
using namespace nncase::scheduler;

namespace
{
constexpr size_t align(size_t size, size_t alignment = 8)
{
    size_t remainder = size % alignment;
    if (remainder != 0)
        return size - remainder + alignment;
    return size;
}
}

memory_node::memory_node(memory_allocator &allocator, size_t start, size_t size)
    : allocator_(allocator), start_(start), size_(size), use_count_(0)
{
}

size_t memory_node::safe_start() const
{
    if (used())
        return start_;
    else
        throw std::runtime_error("Memory node has been freed");
}

void memory_node::add_ref()
{
    ++use_count_;
}

void memory_node::release()
{
    if (--use_count_ == 0)
        allocator_.free(*this);

    if (use_count_ < 0)
        throw std::runtime_error("Memory node has been freed");
}

memory_allocator::memory_allocator(size_t alignment)
    : alignment_(alignment)
{
}

memory_node &memory_allocator::allocate(size_t size)
{
    auto aligned_size = align(size, alignment_);
    auto free_node = freelist_.allocate(aligned_size);
    auto &node = nodes_.emplace_back(*this, free_node.start, free_node.size);
    node.add_ref();
    return node;
}

void memory_allocator::free(memory_node &node)
{
    freelist_.free(free_memory_node { node.start(), node.size() });
}

size_t memory_allocator::get_bytes(datatype_t type, const ir::shape_t &shape) const
{
    return runtime::get_bytes(type) * xt::compute_size(shape);
}
