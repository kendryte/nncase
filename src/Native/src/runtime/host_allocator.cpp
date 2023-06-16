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
#include <nncase/runtime/host_buffer.h>

using namespace nncase;
using namespace nncase::runtime;

namespace {
class host_buffer_impl : public host_buffer_node {
  public:
    host_buffer_impl(gsl::byte *data, size_t bytes,
                     std::function<void(gsl::byte *)> deleter,
                     uintptr_t physical_address, buffer_allocator &allocator,
                     host_sync_status_t host_sync_status)
        : host_buffer_node(bytes, allocator, host_sync_status),
          data_(std::move(data)),
          physical_address_(physical_address),
          deleter_(std::move(deleter)) {}

    ~host_buffer_impl() { deleter_(data_); }

    bool has_physical_address() const noexcept override {
        return physical_address_;
    }

    result<uintptr_t> physical_address() noexcept override {
        return has_physical_address() ? ok(physical_address_)
                                      : err(std::errc::not_supported);
    }

    result<gsl::span<gsl::byte>>
    map_core([[maybe_unused]] map_access_t access) override {
        return ok(gsl::span<gsl::byte>(data_, size_bytes()));
    }

    result<void> unmap_core([[maybe_unused]] map_access_t access) override {
        return ok();
    }

    result<void> sync_core([[maybe_unused]] sync_op_t op) override {
        return ok();
    }

  private:
    gsl::byte *data_;
    uintptr_t physical_address_;
    std::function<void(gsl::byte *)> deleter_;
};

class host_buffer_allocator : public buffer_allocator {
  public:
    result<buffer_t>
    allocate([[maybe_unused]] size_t bytes,
             [[maybe_unused]] const buffer_allocate_options &options) override {
        auto data = new (std::nothrow) gsl::byte[bytes];
        if (!data)
            return err(std::errc::not_enough_memory);
        return ok<buffer_t>(object_t<host_buffer_impl>(
            std::in_place, data, bytes, [](gsl::byte *p) { delete[] p; }, 0,
            *this, host_sync_status_t::valid));
    }

    result<buffer_t>
    attach([[maybe_unused]] gsl::span<gsl::byte> data,
           [[maybe_unused]] const buffer_attach_options &options) override {
        return ok<buffer_t>(object_t<host_buffer_impl>(
            std::in_place, data.data(), data.size_bytes(),
            []([[maybe_unused]] gsl::byte *p) {}, options.physical_address,
            *this, host_sync_status_t::valid));
    }
};

host_buffer_allocator host_allocator;
} // namespace

buffer_allocator &buffer_allocator::host() { return host_allocator; }
