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
#include <nncase/runtime/host_runtime_tensor.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/runtime_tensor.h>
#include <nncase/runtime/runtime_tensor_impl.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::detail;

bool runtime_tensor_impl::is_host() const noexcept {
    return tensor_type() == host_runtime_tensor::tensor_type();
}

bool runtime_tensor_impl::is_contiguous() const noexcept {
    return this->strides() == get_default_strides(this->shape());
}

bool runtime_tensor_impl::can_copy_to_without_staging(
    const runtime_tensor &dest) const noexcept {
    if (dest.empty() || datatype() != dest.datatype() ||
        shape() != dest.shape())
        return false;

    return tensor_type() == dest.tensor_type() || is_host() || dest.is_host() ||
           can_copy_to_different_type(*dest.impl()) ||
           dest.impl()->can_copy_from_different_type(*this);
}

result<void> runtime_tensor_impl::copy_to(runtime_tensor &dest) noexcept {
    CHECK_WITH_ERR(!dest.empty(), std::errc::not_supported);
    CHECK_WITH_ERR(datatype() == dest.datatype(),
                   nncase_errc::datatype_mismatch);
    CHECK_WITH_ERR(shape() == dest.shape(), nncase_errc::shape_mismatch);

    if (tensor_type() == dest.tensor_type())
        return copy_to_same_type(*dest.impl());
    if (is_host())
        return dest.impl()->copy_from_host(*this);
    if (dest.is_host())
        return copy_to_host(*dest.impl());
    if (can_copy_to_different_type(*dest.impl()))
        return copy_to_different_type(*dest.impl());
    if (dest.impl()->can_copy_from_different_type(*this))
        return dest.impl()->copy_from_different_type(*this);

    // staging
    try_var(host, copy_as_host());
    return dest.impl()->copy_from_host(*host.impl());
}

result<runtime_tensor> runtime_tensor_impl::copy_as_host() noexcept {
    try_var(host, host_runtime_tensor::create(datatype(), shape()));
    try_(copy_to_host(*host.impl()));
    return ok(host);
}

bool runtime_tensor_impl::can_copy_from_different_type(
    NNCASE_UNUSED const runtime_tensor_impl &src) const noexcept {
    return false;
}

bool runtime_tensor_impl::can_copy_to_different_type(
    NNCASE_UNUSED const runtime_tensor_impl &dest) const noexcept {
    return false;
}

result<void> runtime_tensor_impl::copy_to_same_type(
    NNCASE_UNUSED runtime_tensor_impl &dest) noexcept {
    return err(std::errc::not_supported);
}

result<void> runtime_tensor_impl::copy_from_different_type(
    NNCASE_UNUSED runtime_tensor_impl &src) noexcept {
    return err(std::errc::not_supported);
}

result<void> runtime_tensor_impl::copy_to_different_type(
    NNCASE_UNUSED runtime_tensor_impl &dest) noexcept {
    return err(std::errc::not_supported);
}

result<void> runtime_tensor_impl::copy_from_host(
    NNCASE_UNUSED runtime_tensor_impl &src) noexcept {
    return err(std::errc::not_supported);
}

result<void> runtime_tensor_impl::copy_to_host(
    NNCASE_UNUSED runtime_tensor_impl &dest) noexcept {
    return err(std::errc::not_supported);
}
