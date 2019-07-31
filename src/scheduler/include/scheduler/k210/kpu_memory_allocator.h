#pragma once
#include "../memory_allocator.h"

namespace nncase
{
namespace scheduler
{
    namespace k210
    {
        class kpu_memory_allocator : public memory_allocator
        {
        public:
            kpu_memory_allocator();

            size_t get_bytes(datatype_t type, const ir::shape_t &shape) const override;
        };
    }
}
}
