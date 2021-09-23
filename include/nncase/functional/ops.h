#pragma once
#include <nncase/runtime/runtime_tensor.h>

#ifndef NNCASE_FUNCTIONAL_IMPL_PLATFORM_HEADER
#include <nncase/functional/ops.platform.h>
#else
#include NNCASE_FUNCTIONAL_IMPL_PLATFORM_HEADER
#endif

namespace nncase::F
{

NNCASE_API inline result<runtime::runtime_tensor> square(runtime::runtime_tensor &input, datatype_t dtype) noexcept
{
    return impl::unary(input, dtype, unary_square);
}

NNCASE_API inline result<runtime::runtime_tensor> sqrt(runtime::runtime_tensor &input, datatype_t dtype) noexcept
{
    return impl::unary(input, dtype, unary_sqrt);
}

NNCASE_API inline result<runtime::runtime_tensor> log(runtime::runtime_tensor &input, datatype_t dtype) noexcept
{
    return impl::unary(input, dtype, unary_log);
}

NNCASE_API inline result<runtime::runtime_tensor> exp(runtime::runtime_tensor &input, datatype_t dtype) noexcept
{
    return impl::unary(input, dtype, unary_exp);
}

NNCASE_API inline result<runtime::runtime_tensor> sin(runtime::runtime_tensor &input, datatype_t dtype) noexcept
{
    return impl::unary(input, dtype, unary_sin);
}

NNCASE_API inline result<runtime::runtime_tensor> cos(runtime::runtime_tensor &input, datatype_t dtype) noexcept
{
    return impl::unary(input, dtype, unary_cos);
}

NNCASE_API inline result<runtime::runtime_tensor> round(runtime::runtime_tensor &input, datatype_t dtype) noexcept
{
    return impl::unary(input, dtype, unary_round);
}

NNCASE_API inline result<runtime::runtime_tensor> floor(runtime::runtime_tensor &input, datatype_t dtype) noexcept
{
    return impl::unary(input, dtype, unary_floor);
}

NNCASE_API inline result<runtime::runtime_tensor> ceil(runtime::runtime_tensor &input, datatype_t dtype) noexcept
{
    return impl::unary(input, dtype, unary_ceil);
}

NNCASE_API inline result<runtime::runtime_tensor> abs(runtime::runtime_tensor &input, datatype_t dtype) noexcept
{
    return impl::unary(input, dtype, unary_abs);
}

NNCASE_API inline result<runtime::runtime_tensor> neg(runtime::runtime_tensor &input, datatype_t dtype) noexcept
{
    return impl::unary(input, dtype, unary_neg);
}

}