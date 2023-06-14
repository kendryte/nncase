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
#include "result.h"
#include <memory>
#include <nncase/object.h>
#include <nncase/runtime/datatypes.h>

BEGIN_NS_NNCASE_RUNTIME

class buffer_node;
class buffer_allocator;
class host_buffer_slice;

using buffer_t = object_t<buffer_node>;

class NNCASE_API buffer_node : public object_node {
    DEFINE_OBJECT_KIND(object_node, object_buffer);

  public:
    buffer_node(size_t size_bytes, buffer_allocator &allocator);

    size_t size_bytes() const noexcept { return size_bytes_; }
    buffer_allocator &allocator() const noexcept { return allocator_; }

    virtual result<void>
    copy_to(buffer_t dest, size_t src_start, size_t dest_start,
            datatype_t datatype, gsl::span<const size_t> shape,
            gsl::span<const size_t> strides,
            gsl::span<const size_t> dest_strides) noexcept = 0;

  private:
    size_t size_bytes_;
    buffer_allocator &allocator_;
};

class NNCASE_API buffer_slice {
  public:
    buffer_slice() noexcept = default;
    buffer_slice(buffer_t buffer) noexcept
        : buffer_(std::move(buffer)),
          start_(0),
          length_(buffer_->size_bytes()) {}

    buffer_slice(buffer_t buffer, size_t start, size_t length) noexcept
        : buffer_(std::move(buffer)), start_(start), length_(length) {}

    const buffer_t &buffer() const noexcept { return buffer_; }

    size_t start() const noexcept { return start_; }
    size_t size_bytes() const noexcept { return length_; }

    buffer_allocator &allocator() const noexcept {
        return buffer_->allocator();
    }

    result<host_buffer_slice> as_host() const noexcept;
    result<void> copy_to(const buffer_slice &dest, datatype_t datatype,
                         gsl::span<const size_t> shape,
                         gsl::span<const size_t> src_strides,
                         gsl::span<const size_t> dest_strides) const noexcept;

  private:
    buffer_t buffer_;
    size_t start_;
    size_t length_;
};

END_NS_NNCASE_RUNTIME
