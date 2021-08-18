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
#include <nncase/schedule/liveness_analysis.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::schedule;

lifetime_recorder::lifetime_recorder(std::list<logical_buffer> &buffers, std::unordered_map<const ir::output_connector *, logical_buffer *> &buffer_map)
    : buffers_(buffers), buffer_map_(buffer_map)
{
}

void lifetime_recorder::allocate(ir::output_connector &conn, memory_location_t location)
{
    auto it = buffer_map_.find(&conn);
    if (it == buffer_map_.end())
    {
        logical_buffer buffer(next_buffer_id_++, conn, location);
        buffer.lifetime().birth = cnt_age_;
        buffer.lifetime().used_count = conn.connections().size();
        buffer.strides_shape() = buffer.shape();
        buffer_map_.emplace(&conn, &buffers_.emplace_back(buffer));
    }
}

void lifetime_recorder::release(ir::output_connector &conn)
{
    auto node = buffer_map_.find(&conn);
    if (node != buffer_map_.end())
    {
        auto &lifetime = node->second->lifetime();
        if (!lifetime.is_alive())
            throw std::runtime_error("Trying to free a released buffer");
        else
            lifetime.used_count--;
    }
}

void lifetime_recorder::grow_age()
{
    current_age(cnt_age_ + 1);
}

void lifetime_recorder::current_age(size_t age)
{
    if (age < cnt_age_)
        throw std::invalid_argument("Cannot set back age");

    auto increase = age - cnt_age_;
    cnt_age_ = age;
    for (auto &b : buffers_)
    {
        auto &lifetime = b.lifetime();
        if (lifetime.is_alive())
            lifetime.age += increase;
    }
}
