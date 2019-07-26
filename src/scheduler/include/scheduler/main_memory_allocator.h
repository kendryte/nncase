#pragma once
#include "memory_allocator.h"

namespace nncase
{
namespace scheduler
{
    class main_memory_allocator : public memory_allocator
    {
    public:
        using memory_allocator::memory_allocator;
    };
}
}
