/* Copyright 2019-2021 Canaan Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include "buffers.h"
#include <list>
#include <map>
#include <nncase/runtime/datatypes.h>
#include <optional>
#include <stdint.h>

namespace nncase::schedule
{
class NNCASE_API freelist
{
    using free_nodes_t = std::map<size_t, memory_span>;

public:
    freelist(std::optional<size_t> fixed_size);
    size_t max_usage() const noexcept { return heap_end_; }

    bool can_allocate(size_t size);
    memory_span allocate(size_t size);
    void free(const memory_span &node);
    std::vector<memory_span> free_nodes() const;

private:
    free_nodes_t::iterator reserve(size_t size);
    void merge(free_nodes_t::iterator offset);

private:
    bool is_fixed_;
    free_nodes_t free_nodes_;
    size_t heap_end_ = 0;
};
}
