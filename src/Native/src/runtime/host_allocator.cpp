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
#include <nncase/runtime/allocator.h>

using namespace nncase;
using namespace nncase::runtime;

namespace {
class host_buffer_allocator : public buffer_allocator {
  public:
    result<buffer_t> allocate([[maybe_unused]] size_t bytes,
                              [[maybe_unused]] const buffer_allocate_options &options) override {
        return err(std::errc::not_supported);
    }

    result<buffer_t> attach([[maybe_unused]] gsl::span<gsl::byte> data,
                            [[maybe_unused]] const buffer_attach_options &options) override {
        return err(std::errc::not_supported);
    }

    result<void> free([[maybe_unused]] buffer_node &buffer) override {
        return err(std::errc::not_supported);
    }
};

host_buffer_allocator host_allocator;
} // namespace

buffer_allocator &buffer_allocator::host() { return host_allocator; }
