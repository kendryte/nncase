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

NNCASE_API inline result<runtime::runtime_tensor> add(runtime::runtime_tensor &input_a, runtime::runtime_tensor &input_b, datatype_t dtype) noexcept
{
    return impl::binary(input_a, input_b, dtype, binary_add);
}

NNCASE_API inline result<runtime::runtime_tensor> sub(runtime::runtime_tensor &input_a, runtime::runtime_tensor &input_b, datatype_t dtype) noexcept
{
    return impl::binary(input_a, input_b, dtype, binary_sub);
}

NNCASE_API inline result<runtime::runtime_tensor> mul(runtime::runtime_tensor &input_a, runtime::runtime_tensor &input_b, datatype_t dtype) noexcept
{
    return impl::binary(input_a, input_b, dtype, binary_mul);
}

NNCASE_API inline result<runtime::runtime_tensor> div(runtime::runtime_tensor &input_a, runtime::runtime_tensor &input_b, datatype_t dtype) noexcept
{
    return impl::binary(input_a, input_b, dtype, binary_div);
}

NNCASE_API inline result<runtime::runtime_tensor> min(runtime::runtime_tensor &input_a, runtime::runtime_tensor &input_b, datatype_t dtype) noexcept
{
    return impl::binary(input_a, input_b, dtype, binary_min);
}

NNCASE_API inline result<runtime::runtime_tensor> max(runtime::runtime_tensor &input_a, runtime::runtime_tensor &input_b, datatype_t dtype) noexcept
{
    return impl::binary(input_a, input_b, dtype, binary_max);
}

NNCASE_API inline result<runtime::runtime_tensor> quantize(runtime::runtime_tensor &input, datatype_t dtype) noexcept
{
    return impl::quantize(input, dtype);
}

NNCASE_API inline result<runtime::runtime_tensor> dequantize(runtime::runtime_tensor &input, datatype_t dtype) noexcept
{
    return impl::dequantize(input, dtype);
}

NNCASE_API inline result<runtime::runtime_tensor> crop(runtime::runtime_tensor &input, std::vector<runtime_shape_t> &bbox, size_t out_h, size_t out_w, image_resize_mode_t resize_mode) noexcept
{
    return impl::crop(input, bbox, out_h, out_w, resize_mode);
}

NNCASE_API inline result<runtime::runtime_tensor> resize(runtime::runtime_tensor &input, size_t out_h, size_t out_w, image_resize_mode_t resize_mode) noexcept
{
    return impl::resize(input, out_h, out_w, resize_mode);
}

NNCASE_API inline result<runtime::runtime_tensor> pad(runtime::runtime_tensor &input, std::vector<padding> &padding, pad_mode_t pad_mode, float fill_v) noexcept
{
    return impl::pad(input, padding, pad_mode, fill_v);
}

}