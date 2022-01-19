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
#include <nncase/runtime/dbg.h>
#include <nncase/runtime/error.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/runtime_tensor.h>
#include <nncase/runtime/runtime_tensor_impl.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::detail;

namespace
{
runtime_shape_t empty_shape;
runtime_tensor_type empty_runtime_tensor_type { "empty" };
quant_param_t empty_quant_param {};
}

runtime_tensor::runtime_tensor() noexcept
{
}

runtime_tensor::runtime_tensor(std::shared_ptr<runtime_tensor_impl> impl) noexcept
    : impl_(std::move(impl))
{
}

datatype_t runtime_tensor::datatype() const noexcept
{
    if (impl_)
        return impl_->datatype();
    return (datatype_t)0;
}

const quant_param_t &runtime_tensor::quant_param() const noexcept
{
    if (impl_)
        return impl_->quant_param();
    return empty_quant_param;
}

void runtime_tensor::quant_param(const quant_param_t &quant) const noexcept
{
    if (impl_)
        impl_->quant_param(quant);
}

const runtime_shape_t &runtime_tensor::shape() const noexcept
{
    if (impl_)
        return impl_->shape();
    return empty_shape;
}

const runtime_shape_t &runtime_tensor::strides() const noexcept
{
    if (impl_)
        return impl_->strides();
    return empty_shape;
}

runtime_tensor_type &runtime_tensor::tensor_type() const noexcept
{
    if (impl_)
        return impl_->tensor_type();
    return empty_runtime_tensor_type;
}

result<runtime_tensor> runtime_tensor::as_host() noexcept
{
    CHECK_WITH_ERR(!empty(), std::errc::not_supported);
    if (is_host())
        return ok(*this);
    return impl_->copy_as_host();
}

bool runtime_tensor::is_host() const noexcept
{
    if (empty())
        return false;
    return impl_->is_host();
}

bool runtime_tensor::is_contiguous() const noexcept
{
    if (empty())
        return false;
    return impl_->is_contiguous();
}

void runtime_tensor::reset() noexcept
{
    impl_.reset();
}

bool runtime_tensor::empty() const noexcept
{
    return !impl_;
}

bool runtime_tensor::can_copy_to_without_staging(const runtime_tensor &dest) const noexcept
{
    if (empty())
        return false;
    return impl_->can_copy_to_without_staging(dest);
}

result<void> runtime_tensor::copy_to(runtime_tensor &dest) noexcept
{
    CHECK_WITH_ERR(!empty(), std::errc::not_supported);
    return impl_->copy_to(dest);
}

bool runtime::operator==(const runtime_tensor &lhs, const runtime_tensor &rhs) noexcept
{
    return lhs.impl() == rhs.impl();
}

bool runtime::operator!=(const runtime_tensor &lhs, const runtime_tensor &rhs) noexcept
{
    return !(lhs == rhs);
}
