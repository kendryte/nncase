/* Copyright 2020 Canaan Inc.
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
#include <nncase/kernels/tensor_compute.h>
#include <nncase/runtime/error.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/runtime_tensor.h>

using namespace nncase;
using namespace nncase::runtime;

namespace
{
class host_runtime_tensor_type : public runtime_tensor_type
{
public:
    bool can_copy_from_different_type(NNCASE_UNUSED const runtime_tensor &dest, NNCASE_UNUSED const runtime_tensor &src) noexcept override
    {
        return true;
    }

    bool can_copy_to_different_type(NNCASE_UNUSED const runtime_tensor &dest, NNCASE_UNUSED const runtime_tensor &src) noexcept override
    {
        return true;
    }

    result<void> copy_to_same_type(const runtime_tensor &src, const runtime_tensor &dest) noexcept override
    {
        auto buffer_src = host_runtime_tensor::buffer(src).unwrap();
        auto buffer_dest = host_runtime_tensor::buffer(dest).unwrap();
        if (src.datatype() != dest.datatype())
            return err(nncase_errc::datatype_mismatch);
        if (src.shape() != dest.shape())
            return err(nncase_errc::shape_mismatch);

        return kernels::copy(src.datatype(), buffer_src.data(), buffer_dest.data(), src.shape(), src.strides(), dest.strides());
    }

    result<void> copy_from_different_type(const runtime_tensor &src, const runtime_tensor &dest) noexcept override
    {
        return src.tensor_type().copy_to_host(src, dest);
    }

    result<void> copy_to_different_type(const runtime_tensor &src, const runtime_tensor &dest) noexcept override
    {
        return dest.tensor_type().copy_from_host(src, dest);
    }

    result<void> copy_from_host(const runtime_tensor &src, const runtime_tensor &dest) noexcept override
    {
        return copy_to_same_type(src, dest);
    }

    result<void> copy_to_host(const runtime_tensor &src, const runtime_tensor &dest) noexcept override
    {
        return copy_to_same_type(src, dest);
    }
};

class empty_runtime_tensor_type : public runtime_tensor_type
{
};

host_runtime_tensor_type host_runtime_tensor_type_;
empty_runtime_tensor_type empty_runtime_tensor_type_;
}

bool runtime_tensor_type::can_copy_from_different_type(NNCASE_UNUSED const runtime_tensor &src, NNCASE_UNUSED const runtime_tensor &dest) noexcept
{
    return false;
}

bool runtime_tensor_type::can_copy_to_different_type(NNCASE_UNUSED const runtime_tensor &src, NNCASE_UNUSED const runtime_tensor &dest) noexcept
{
    return false;
}

result<void> runtime_tensor_type::copy_to_same_type(NNCASE_UNUSED const runtime_tensor &src, NNCASE_UNUSED const runtime_tensor &dest) noexcept
{
    return err(std::errc::not_supported);
}

result<void> runtime_tensor_type::copy_from_different_type(NNCASE_UNUSED const runtime_tensor &src, NNCASE_UNUSED const runtime_tensor &dest) noexcept
{
    return err(std::errc::not_supported);
}

result<void> runtime_tensor_type::copy_to_different_type(NNCASE_UNUSED const runtime_tensor &src, NNCASE_UNUSED const runtime_tensor &dest) noexcept
{
    return err(std::errc::not_supported);
}

result<void> runtime_tensor_type::copy_from_host(NNCASE_UNUSED const runtime_tensor &src, NNCASE_UNUSED const runtime_tensor &dest) noexcept
{
    return err(std::errc::not_supported);
}

result<void> runtime_tensor_type::copy_to_host(NNCASE_UNUSED const runtime_tensor &src, NNCASE_UNUSED const runtime_tensor &dest) noexcept
{
    return err(std::errc::not_supported);
}

runtime_tensor_type &host_runtime_tensor::tensor_type() noexcept
{
    return host_runtime_tensor_type_;
}

runtime_tensor::runtime_tensor() noexcept
    : datatype_(dt_uint8), tensor_type_(&empty_runtime_tensor_type_)
{
}

runtime_tensor::runtime_tensor(datatype_t datatype, runtime_shape_t shape, runtime_shape_t strides, runtime_tensor_type &tensor_type, std::shared_ptr<void> data) noexcept
    : datatype_(datatype), shape_(std::move(shape)), strides_(std::move(strides)), tensor_type_(&tensor_type), data_(std::move(data))
{
}

bool runtime_tensor::can_copy_to_without_staging(const runtime_tensor &dest) const noexcept
{
    if (datatype() != dest.datatype()
        || shape() != dest.shape())
        return false;

    return tensor_type() == dest.tensor_type()
        || is_host() || dest.is_host()
        || tensor_type().can_copy_to_different_type(*this, dest)
        || dest.tensor_type().can_copy_from_different_type(*this, dest);
}

result<void> runtime_tensor::copy_to(const runtime_tensor &dest) const noexcept
{
    if (empty() || dest.empty())
        return err(std::errc::not_supported);
    if (datatype() != dest.datatype())
        return err(nncase_errc::datatype_mismatch);
    if (shape() != dest.shape())
        return err(nncase_errc::shape_mismatch);

    if (tensor_type() == dest.tensor_type())
        return tensor_type().copy_to_same_type(*this, dest);
    if (is_host())
        return dest.tensor_type().copy_from_host(*this, dest);
    if (dest.is_host())
        return tensor_type().copy_to_host(*this, dest);
    if (tensor_type().can_copy_to_different_type(*this, dest))
        return tensor_type().copy_to_different_type(*this, dest);
    if (dest.tensor_type().can_copy_from_different_type(*this, dest))
        return dest.tensor_type().copy_from_different_type(*this, dest);

    // staging
    try_var(host, as_host());
    return dest.tensor_type().copy_from_host(host, dest);
}

result<runtime_tensor> runtime_tensor::as_host() const noexcept
{
    if (empty())
        return err(std::errc::not_supported);
    if (is_host())
        return ok(*this);
    try_var(host, host_runtime_tensor::create(datatype(), shape()));
    try_(tensor_type().copy_to_host(*this, host));
    return ok(host);
}

bool runtime_tensor::is_host() const noexcept
{
    return tensor_type() == host_runtime_tensor::tensor_type();
}

void runtime_tensor::reset() noexcept
{
    *this = runtime_tensor();
}

bool runtime_tensor::empty() const noexcept
{
    return tensor_type() == empty_runtime_tensor_type_;
}

bool runtime::operator==(const runtime_tensor &lhs, const runtime_tensor &rhs) noexcept
{
    return lhs.datatype() == rhs.datatype() && lhs.shape() == rhs.shape()
        && lhs.tensor_type() == rhs.tensor_type() && lhs.data() == rhs.data();
}

bool runtime::operator!=(const runtime_tensor &lhs, const runtime_tensor &rhs) noexcept
{
    return !(lhs == rhs);
}

result<runtime_tensor> host_runtime_tensor::create(datatype_t datatype, runtime_shape_t shape, runtime_shape_t strides) noexcept
{
    auto size = xt::compute_strides(shape, xt::layout_type::row_major, strides) * get_bytes(datatype);
    std::shared_ptr<uint8_t> buffer(new (std::nothrow) uint8_t[size], std::default_delete<uint8_t[]>());
    if (!buffer)
        return err(std::errc::not_enough_memory);
    return ok(runtime_tensor(datatype, std::move(shape), std::move(strides), host_runtime_tensor_type_, std::move(buffer)));
}

result<runtime_tensor> host_runtime_tensor::create(datatype_t datatype, runtime_shape_t shape, runtime_shape_t strides, gsl::span<gsl::byte> data, bool copy) noexcept
{
    auto size = xt::compute_strides(shape, xt::layout_type::row_major, strides) * get_bytes(datatype);
    if (data.size_bytes() != size)
        return err(std::errc::invalid_argument);

    std::shared_ptr<gsl::byte> buffer;
    if (copy)
    {
        buffer.reset(new (std::nothrow) gsl::byte[size], std::default_delete<gsl::byte[]>());
        if (!buffer)
            return err(std::errc::not_enough_memory);
        try_(kernels::copy(datatype, data.data(), buffer.get(), shape, strides, strides));
    }
    else
    {
        buffer.reset(data.data(), [](NNCASE_UNUSED gsl::byte *ptr) {});
    }

    return ok(runtime_tensor(datatype, std::move(shape), std::move(strides), host_runtime_tensor_type_, std::move(buffer)));
}

result<runtime_tensor> host_runtime_tensor::create(datatype_t datatype, runtime_shape_t shape, runtime_shape_t strides, gsl::span<gsl::byte> data, data_deleter_t data_deleter) noexcept
{
    auto size = xt::compute_strides(shape, xt::layout_type::row_major, strides) * get_bytes(datatype);
    if (data.size_bytes() != size)
        return err(std::errc::invalid_argument);
    return ok(runtime_tensor(datatype, std::move(shape), std::move(strides), host_runtime_tensor_type_,
        std::shared_ptr<gsl::byte>(data.data(), data_deleter)));
}

result<runtime_tensor> host_runtime_tensor::create(datatype_t datatype, runtime_shape_t shape) noexcept
{
    return create(datatype, shape, get_default_strides(shape));
}

result<runtime_tensor> host_runtime_tensor::create(datatype_t datatype, runtime_shape_t shape, gsl::span<gsl::byte> data, bool copy) noexcept
{
    return create(datatype, shape, get_default_strides(shape), data, copy);
}

result<runtime_tensor> host_runtime_tensor::create(datatype_t datatype, runtime_shape_t shape, gsl::span<gsl::byte> data, data_deleter_t data_deleter) noexcept
{
    return create(datatype, shape, get_default_strides(shape), data, std::move(data_deleter));
}

result<gsl::span<gsl::byte>> host_runtime_tensor::buffer(const runtime_tensor &tensor) noexcept
{
    if (tensor.is_host())
    {
        auto size = get_bytes(tensor.datatype(), tensor.shape());
        return ok(gsl::span<gsl::byte>(tensor.data_as<gsl::byte>(), size));
    }
    else
    {
        return err(std::errc::invalid_argument);
    }
}
