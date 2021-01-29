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
    bool can_copy_from_different_type(runtime_tensor &dest, runtime_tensor &src) noexcept override
    {
        return true;
    }

    bool can_copy_to_different_type(runtime_tensor &dest, runtime_tensor &src) noexcept override
    {
        return true;
    }

    result<void> copy_to_same_type(runtime_tensor &src, runtime_tensor &dest) noexcept override
    {
        auto buffer_src = host_runtime_tensor::buffer(src);
        auto buffer_dest = host_runtime_tensor::buffer(src);
        if (buffer_src.size() != buffer_dest.size())
            return err(nncase_errc::shape_mismatch);
        std::memcpy(buffer_dest.data(), buffer_src.data(), buffer_src.size_bytes());
        return ok();
    }

    result<void> copy_from_different_type(runtime_tensor &src, runtime_tensor &dest) noexcept override
    {
        return src.tensor_type().copy_to_host(src, dest);
    }

    result<void> copy_to_different_type(runtime_tensor &src, runtime_tensor &dest) noexcept override
    {
        return dest.tensor_type().copy_from_host(src, dest);
    }

    result<void> copy_from_host(runtime_tensor &src, runtime_tensor &dest) noexcept override
    {
        return copy_to_same_type(src, dest);
    }

    result<void> copy_to_host(runtime_tensor &src, runtime_tensor &dest) noexcept override
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

bool runtime_tensor_type::can_copy_from_different_type(runtime_tensor &src, runtime_tensor &dest) noexcept
{
    return false;
}

bool runtime_tensor_type::can_copy_to_different_type(runtime_tensor &src, runtime_tensor &dest) noexcept
{
    return false;
}

result<void> runtime_tensor_type::copy_to_same_type(runtime_tensor &src, runtime_tensor &dest) noexcept
{
    return err(std::errc::not_supported);
}

result<void> runtime_tensor_type::copy_from_different_type(runtime_tensor &src, runtime_tensor &dest) noexcept
{
    return err(std::errc::not_supported);
}

result<void> runtime_tensor_type::copy_to_different_type(runtime_tensor &src, runtime_tensor &dest) noexcept
{
    return err(std::errc::not_supported);
}

result<void> runtime_tensor_type::copy_from_host(runtime_tensor &src, runtime_tensor &dest) noexcept
{
    return err(std::errc::not_supported);
}

result<void> runtime_tensor_type::copy_to_host(runtime_tensor &src, runtime_tensor &dest) noexcept
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

runtime_tensor::runtime_tensor(datatype_t datatype, const runtime_shape_t &shape, runtime_tensor_type &tensor_type, std::shared_ptr<void> data) noexcept
    : datatype_(datatype), shape_(shape), tensor_type_(&tensor_type), data_(std::move(data))
{
}

bool runtime_tensor::can_copy_to_without_staging(runtime_tensor &dest) noexcept
{
    if (datatype() != dest.datatype()
        || shape() != dest.shape())
        return false;

    return tensor_type() == dest.tensor_type()
        || is_host() || dest.is_host()
        || tensor_type().can_copy_to_different_type(*this, dest)
        || dest.tensor_type().can_copy_from_different_type(*this, dest);
}

result<void> runtime_tensor::copy_to(runtime_tensor &dest) noexcept
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

result<runtime_tensor> runtime_tensor::as_host() noexcept
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

result<runtime_tensor> host_runtime_tensor::create(datatype_t datatype, const runtime_shape_t &shape) noexcept
{
    auto size = get_bytes(datatype, shape);
    std::shared_ptr<uint8_t> buffer(new (std::nothrow) uint8_t[size], std::default_delete<uint8_t[]>());
    if (!buffer)
        return err(std::errc::not_enough_memory);
    return ok(runtime_tensor(datatype, shape, host_runtime_tensor_type_, std::move(buffer)));
}

result<runtime_tensor> host_runtime_tensor::create(datatype_t datatype, const runtime_shape_t &shape, gsl::span<gsl::byte> data, bool copy) noexcept
{
    auto size = get_bytes(datatype, shape);
    if (data.size_bytes() != size)
        return err(std::errc::invalid_argument);

    std::shared_ptr<uint8_t> buffer;
    if (copy)
    {
        buffer.reset(new (std::nothrow) uint8_t[size], std::default_delete<uint8_t[]>());
        if (!buffer)
            return err(std::errc::not_enough_memory);
        std::memcpy(buffer.get(), data.data(), size);
    }
    else
    {
        buffer.reset((uint8_t *)data.data(), [](uint8_t *ptr) {});
    }

    return ok(runtime_tensor(datatype, shape, host_runtime_tensor_type_, std::move(buffer)));
}

result<runtime_tensor> host_runtime_tensor::create(datatype_t datatype, const runtime_shape_t &shape, gsl::span<gsl::byte> data, std::function<void(gsl::span<gsl::byte>)> data_deleter) noexcept
{
    auto size = get_bytes(datatype, shape);
    if (data.size_bytes() != size)
        return err(std::errc::invalid_argument);
    return ok(runtime_tensor(datatype, shape, host_runtime_tensor_type_,
        std::shared_ptr<uint8_t>((uint8_t *)data.data(), [=](uint8_t *ptr) { data_deleter(data); })));
}

gsl::span<gsl::byte> host_runtime_tensor::buffer(runtime_tensor &tensor) noexcept
{
    assert(tensor.is_host());
    auto size = get_bytes(tensor.datatype(), tensor.shape());
    return { tensor.data_as<gsl::byte>(), size };
}
