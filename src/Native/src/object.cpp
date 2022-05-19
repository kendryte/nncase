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
#include <cassert>
#include <nncase/object.h>

using namespace nncase;

object_node::object_node() noexcept : ref_count_(1) {}

bool object_node::is_a(
    [[maybe_unused]] const object_kind &kind) const noexcept {
    return false;
}

bool object_node::equals(const object_node &other) const noexcept {
    return this == &other;
}

uint32_t object_node::add_ref() const noexcept {
    return ref_count_.fetch_add(1, std::memory_order_relaxed);
}

uint32_t object_node::release() const noexcept {
    assert(ref_count_);
    auto count = ref_count_.fetch_sub(1, std::memory_order_acq_rel);
    if (count == 1) {
        delete this;
    }
    return count;
}
