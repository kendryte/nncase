#pragma once
#include <list>
#include <map>
#include <optional>
#include <stdint.h>

namespace nncase
{
namespace scheduler
{
    struct free_memory_node
    {
        size_t start;
        size_t size;

        size_t end() const noexcept { return start + size; }
    };

    class freelist
    {
        using free_nodes_t = std::map<size_t, free_memory_node>;

    public:
        freelist(std::optional<size_t> fixed_size);
        size_t max_usage() const noexcept { return heap_end_; }

        free_memory_node allocate(size_t size);
        void free(const free_memory_node &node);

    private:
        free_nodes_t::iterator reserve(size_t size);
        void merge(free_nodes_t::iterator offset);

    private:
        bool is_fixed_;
        bool is_ping_ = true;
        free_nodes_t free_nodes_;
        size_t heap_end_ = 0;
    };
}
}
