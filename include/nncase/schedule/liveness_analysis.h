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
#include "schedule_types.h"

namespace nncase::schedule
{
class lifetime_recorder
{
public:
    lifetime_recorder(std::list<logical_buffer> &buffers, std::unordered_map<const ir::output_connector *, logical_buffer *> &buffer_map);

    size_t current_age() const noexcept { return cnt_age_; }
    void current_age(size_t age);

    void allocate(ir::output_connector &conn, memory_location_t location);
    void release(ir::output_connector &conn);
    void grow_age();

private:
    size_t next_buffer_id_ = 0;
    size_t cnt_age_ = 0;
    std::list<logical_buffer> &buffers_;
    std::unordered_map<const ir::output_connector *, logical_buffer *> &buffer_map_;
};
}
