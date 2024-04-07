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

#ifdef DUMP_MEM
#include <iomanip>
#include <iostream>
static uint64_t used_mem = 0;
static uint64_t max_mem = 0;
#endif
namespace {
class host_buffer_impl : public host_buffer_node {
  public:
    host_buffer_impl(std::byte *data, size_t bytes,
                     std::function<void(std::byte *)> deleter,
                     uintptr_t physical_address, buffer_allocator &allocator,
                     host_sync_status_t host_sync_status,
                     [[maybe_unused]] bool collect = false)
        : host_buffer_node(bytes, allocator, host_sync_status),
          data_(std::move(data)),
          physical_address_(physical_address),
          deleter_(std::move(deleter)) {
#ifdef DUMP_MEM
        bytes_size_ = bytes;
        collect_ = collect;
#endif
    }

    ~host_buffer_impl() {
        deleter_(data_);
#ifdef DUMP_MEM
        if (collect_) {
            if (max_mem < used_mem)
                max_mem = used_mem;
            std::cout << "[Used_mem]:" << std::setw(16) << std::setfill(' ')
                      << used_mem << "\t[deleter ]:" << std::setw(16)
                      << std::setfill(' ') << bytes_size_
                      << "\t[Max_mem]: " << max_mem << std::endl;
            used_mem -= bytes_size_;
        }
#endif
    }

    bool has_physical_address() const noexcept override {
        return physical_address_;
    }

    result<uintptr_t> physical_address() noexcept override {
        return has_physical_address() ? ok(physical_address_)
                                      : err(std::errc::not_supported);
    }

    result<std::span<std::byte>>
    map_core([[maybe_unused]] map_access_t access) override {
        return ok(std::span<std::byte>(data_, size_bytes()));
    }

    result<void> unmap_core([[maybe_unused]] map_access_t access) override {
        return ok();
    }

    result<void> sync_core([[maybe_unused]] sync_op_t op) override {
        return ok();
    }

  private:
    std::byte *data_;
    uintptr_t physical_address_;
    std::function<void(std::byte *)> deleter_;
#ifdef DUMP_MEM
    size_t bytes_size_;
    bool collect_;
#endif
};

class host_buffer_allocator : public buffer_allocator {
  public:
    result<buffer_t>
    allocate([[maybe_unused]] size_t bytes,
             [[maybe_unused]] const buffer_allocate_options &options) override {
#ifdef DUMP_MEM
        std::cout << "[Used_mem]:" << std::setw(16) << std::setfill(' ')
                  << used_mem << "\t[allocate]:" << std::setw(16)
                  << std::setfill(' ') << bytes << std::endl;
        used_mem += bytes;
#endif
        auto data = new (std::nothrow) std::byte[bytes];
        if (!data)
            return err(std::errc::not_enough_memory);
        auto paddr =
            options.flags & HOST_BUFFER_ALLOCATE_SHARED ? (uintptr_t)data : 0;
        return ok<buffer_t>(object_t<host_buffer_impl>(
            std::in_place, data, bytes, [](std::byte *p) { delete[] p; }, paddr,
            *this, host_sync_status_t::valid, true));
    }

    result<buffer_t>
    attach([[maybe_unused]] std::span<std::byte> data,
           [[maybe_unused]] const buffer_attach_options &options) override {
        auto paddr = options.flags & HOST_BUFFER_ATTACH_SHARED
                         ? (options.physical_address ? options.physical_address
                                                     : (uintptr_t)data.data())
                         : 0;
        return ok<buffer_t>(object_t<host_buffer_impl>(
            std::in_place, data.data(), data.size_bytes(),
            []([[maybe_unused]] std::byte *p) {}, paddr, *this,
            host_sync_status_t::valid));
    }

    void shrink_memory_pool() override {}
};

host_buffer_allocator host_allocator;
} // namespace

buffer_allocator &buffer_allocator::host() { return host_allocator; }
