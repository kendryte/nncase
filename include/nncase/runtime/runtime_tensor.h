/* Copyright 2019-2020 Canaan Inc.
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
#include "model.h"
#include "result.h"
#include <memory>

BEGIN_NS_NNCASE_RUNTIME

class runtime_tensor;

class NNCASE_API runtime_tensor_type
{
public:
    virtual bool can_copy_from_different_type(const runtime_tensor &src, const runtime_tensor &dest) noexcept;
    virtual bool can_copy_to_different_type(const runtime_tensor &src, const runtime_tensor &dest) noexcept;

    virtual result<void> copy_to_same_type(const runtime_tensor &src, const runtime_tensor &dest) noexcept;
    virtual result<void> copy_from_different_type(const runtime_tensor &src, const runtime_tensor &dest) noexcept;
    virtual result<void> copy_to_different_type(const runtime_tensor &src, const runtime_tensor &dest) noexcept;
    virtual result<void> copy_from_host(const runtime_tensor &src, const runtime_tensor &dest) noexcept;
    virtual result<void> copy_to_host(const runtime_tensor &src, const runtime_tensor &dest) noexcept;
};

inline bool operator==(runtime_tensor_type &lhs, runtime_tensor_type &rhs) noexcept
{
    return &lhs == &rhs;
}

inline bool operator!=(runtime_tensor_type &lhs, runtime_tensor_type &rhs) noexcept
{
    return &lhs != &rhs;
}

class NNCASE_API runtime_tensor
{
public:
    runtime_tensor() noexcept;
    runtime_tensor(datatype_t datatype, runtime_shape_t shape, runtime_shape_t strides, runtime_tensor_type &tensor_type, std::shared_ptr<void> data) noexcept;

    datatype_t datatype() const noexcept { return datatype_; }
    const runtime_shape_t &shape() const noexcept { return shape_; }
    const runtime_shape_t &strides() const noexcept { return strides_; }
    runtime_tensor_type &tensor_type() const noexcept { return *tensor_type_; }
    bool empty() const noexcept;
    bool is_host() const noexcept;

    const std::shared_ptr<void> &data() const noexcept { return data_; }
    std::shared_ptr<void> &data() noexcept { return data_; }

    template <class T = void>
    T *data_as() const noexcept { return reinterpret_cast<T *>(data_.get()); }

    bool can_copy_to_without_staging(const runtime_tensor &dest) const noexcept;
    result<void> copy_to(const runtime_tensor &dest) const noexcept;
    result<runtime_tensor> as_host() const noexcept;

    void reset() noexcept;

private:
    datatype_t datatype_;
    runtime_shape_t shape_;
    runtime_shape_t strides_;
    runtime_tensor_type *tensor_type_;
    std::shared_ptr<void> data_;
};

NNCASE_API bool operator==(const runtime_tensor &lhs, const runtime_tensor &rhs) noexcept;
NNCASE_API bool operator!=(const runtime_tensor &lhs, const runtime_tensor &rhs) noexcept;

namespace host_runtime_tensor
{
NNCASE_API runtime_tensor_type &tensor_type() noexcept;
NNCASE_API result<runtime_tensor> create(datatype_t datatype, runtime_shape_t shape) noexcept;
NNCASE_API result<runtime_tensor> create(datatype_t datatype, runtime_shape_t shape, gsl::span<gsl::byte> data, bool copy) noexcept;
NNCASE_API result<runtime_tensor> create(datatype_t datatype, runtime_shape_t shape, gsl::span<gsl::byte> data, std::function<void(gsl::span<gsl::byte>)> data_deleter) noexcept;
NNCASE_API result<runtime_tensor> create(datatype_t datatype, runtime_shape_t shape, runtime_shape_t strides) noexcept;
NNCASE_API result<runtime_tensor> create(datatype_t datatype, runtime_shape_t shape, runtime_shape_t strides, gsl::span<gsl::byte> data, bool copy) noexcept;
NNCASE_API result<runtime_tensor> create(datatype_t datatype, runtime_shape_t shape, runtime_shape_t strides, gsl::span<gsl::byte> data, std::function<void(gsl::span<gsl::byte>)> data_deleter) noexcept;
NNCASE_API result<gsl::span<gsl::byte>> buffer(const runtime_tensor &tensor) noexcept;
}

END_NS_NNCASE_RUNTIME
