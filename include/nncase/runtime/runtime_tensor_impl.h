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
#include "runtime_tensor.h"

BEGIN_NS_NNCASE_RUNTIME

namespace detail
{
class NNCASE_API runtime_tensor_impl
{
public:
    virtual ~runtime_tensor_impl() = default;

    virtual datatype_t datatype() const noexcept = 0;
    virtual const runtime_shape_t &shape() const noexcept = 0;
    virtual const runtime_shape_t &strides() const noexcept = 0;
    virtual runtime_tensor_type &tensor_type() const noexcept = 0;
    bool is_host() const noexcept;
    bool is_contiguous() const noexcept;

    bool can_copy_to_without_staging(const runtime_tensor &dest) const noexcept;
    result<void> copy_to(runtime_tensor &dest) noexcept;
    result<runtime_tensor> copy_as_host() noexcept;

    virtual bool can_copy_from_different_type(const runtime_tensor_impl &src) const noexcept;
    virtual bool can_copy_to_different_type(const runtime_tensor_impl &dest) const noexcept;

    virtual result<void> copy_to_same_type(runtime_tensor_impl &dest) noexcept;
    virtual result<void> copy_from_different_type(runtime_tensor_impl &src) noexcept;
    virtual result<void> copy_to_different_type(runtime_tensor_impl &dest) noexcept;
    virtual result<void> copy_from_host(runtime_tensor_impl &src) noexcept;
    virtual result<void> copy_to_host(runtime_tensor_impl &dest) noexcept;
};
}

END_NS_NNCASE_RUNTIME
