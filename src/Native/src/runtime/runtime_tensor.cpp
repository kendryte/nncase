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

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::detail;

runtime_tensor::runtime_tensor() noexcept : impl_(nullptr) {}

runtime_tensor::runtime_tensor(tensor impl) noexcept : impl_(std::move(impl)) {}

typecode_t runtime_tensor::datatype() const noexcept {
    auto type = impl_->dtype().as<prim_type_t>().expect("Not a prim type");
    return type->typecode();
}

gsl::span<const size_t> runtime_tensor::shape() const noexcept {
    return impl_->shape();
}

gsl::span<const size_t> runtime_tensor::strides() const noexcept {
    return impl_->strides();
}

result<runtime_tensor> runtime_tensor::to_host() noexcept {
    CHECK_WITH_ERR(!empty(), std::errc::not_supported);
    if (is_host())
        return ok(*this);
    checked_try_var(host_tensor, impl_->to_host());
    return ok(runtime_tensor(std::move(host_tensor)));
}

bool runtime_tensor::is_host() const noexcept {
    if (empty())
        return false;
    return impl_.is_a<host_buffer_t>();
}

bool runtime_tensor::is_contiguous() const noexcept {
    if (empty())
        return false;
    return impl_->is_contiguous();
}

void runtime_tensor::reset() noexcept { impl_ = nullptr; }

bool runtime_tensor::empty() const noexcept { return impl_.empty(); }

bool runtime_tensor::can_copy_to_without_staging(
    const runtime_tensor &dest) const noexcept {
    if (empty())
        return false;
    return is_host() || dest.is_host();
}

result<void> runtime_tensor::copy_to(runtime_tensor &dest) noexcept {
    CHECK_WITH_ERR(!empty(), std::errc::not_supported);
    return impl_->copy_to(dest.impl_);
}

bool runtime::operator==(const runtime_tensor &lhs,
                         const runtime_tensor &rhs) noexcept {
    return lhs.impl().equals(rhs.impl());
}

bool runtime::operator!=(const runtime_tensor &lhs,
                         const runtime_tensor &rhs) noexcept {
    return !(lhs == rhs);
}
