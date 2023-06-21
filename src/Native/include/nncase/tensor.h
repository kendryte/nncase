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
#include "object.h"
#include "shape.h"
#include "value.h"
#include <nncase/runtime/buffer.h>
#include <nncase/runtime/datatypes.h>

namespace nncase {
class tensor_node;
using tensor = object_t<tensor_node>;

class NNCASE_API tensor_node : public value_node {
    DEFINE_OBJECT_KIND(value_node, object_tensor);

  public:
    tensor_node(datatype_t dtype, dims_t shape, strides_t strides,
                runtime::buffer_slice buffer);

    /** @brief Gets element type. */
    const datatype_t &dtype() const noexcept { return dtype_; }

    /** @brief Gets shape. */
    gsl::span<const size_t> shape() const noexcept { return shape_; }

    /** @brief Gets strides. */
    gsl::span<const size_t> strides() const noexcept { return strides_; }

    /** @brief Gets length. */
    size_t length() const noexcept { return length_; }

    /** @brief Gets buffer. */
    const runtime::buffer_slice &buffer() const noexcept { return buffer_; }

    /** @brief Gets whether buffer is contiguous. */
    bool is_contiguous() const noexcept;

    result<void> copy_from(tensor src) noexcept;
    result<void> copy_to(tensor dest) const noexcept;
    result<tensor> to_host() noexcept;

    result<void> copy_to(value_t dest) const noexcept override;

  private:
    datatype_t dtype_;
    dims_t shape_;
    strides_t strides_;
    size_t length_;
    runtime::buffer_slice buffer_;
};
} // namespace nncase
