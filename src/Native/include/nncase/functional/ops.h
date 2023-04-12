/**
 * @file ops.h
 * @date 2021-09-27
 *
 * @copyright 2019-2021 Canaan Inc.
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
 *
 */
#pragma once
#include <nncase/runtime/runtime_tensor.h>

#ifndef NNCASE_FUNCTIONAL_IMPL_PLATFORM_HEADER
#include <nncase/functional/ops.platform.h>
#else
#include NNCASE_FUNCTIONAL_IMPL_PLATFORM_HEADER
#endif

namespace nncase::F {

/**
 * @brief unary_square
 *
 * @param input runtime_tensor
 * @param dtype output tensor datatype
 * @return result<runtime::runtime_tensor>
 */
NNCASE_API inline result<runtime::runtime_tensor>
square(runtime::runtime_tensor &input, datatype_t dtype) noexcept {
    return impl::unary(input, dtype, unary_square);
}
/**
 * @brief unary_sqrt
 *
 * @param input runtime_tensor
 * @param dtype output tensor datatype
 * @return result<runtime::runtime_tensor>
 */
NNCASE_API inline result<runtime::runtime_tensor>
sqrt(runtime::runtime_tensor &input, datatype_t dtype) noexcept {
    return impl::unary(input, dtype, unary_sqrt);
}
/**
 * @brief unary_log
 *
 * @param input runtime_tensor
 * @param dtype output tensor datatype
 * @return result<runtime::runtime_tensor>
 */
NNCASE_API inline result<runtime::runtime_tensor>
log(runtime::runtime_tensor &input, datatype_t dtype) noexcept {
    return impl::unary(input, dtype, unary_log);
}
/**
 * @brief unary_exp
 *
 * @param input runtime_tensor
 * @param dtype output tensor datatype
 * @return result<runtime::runtime_tensor>
 */
NNCASE_API inline result<runtime::runtime_tensor>
exp(runtime::runtime_tensor &input, datatype_t dtype) noexcept {
    return impl::unary(input, dtype, unary_exp);
}
/**
 * @brief unary_sin
 *
 * @param input runtime_tensor
 * @param dtype output tensor datatype
 * @return result<runtime::runtime_tensor>
 */
NNCASE_API inline result<runtime::runtime_tensor>
sin(runtime::runtime_tensor &input, datatype_t dtype) noexcept {
    return impl::unary(input, dtype, unary_sin);
}
/**
 * @brief unary_cos
 *
 * @param input runtime_tensor
 * @param dtype output tensor datatype
 * @return result<runtime::runtime_tensor>
 */
NNCASE_API inline result<runtime::runtime_tensor>
cos(runtime::runtime_tensor &input, datatype_t dtype) noexcept {
    return impl::unary(input, dtype, unary_cos);
}
/**
 * @brief unary_round
 *
 * @param input runtime_tensor
 * @param dtype output tensor datatype
 * @return result<runtime::runtime_tensor>
 */
NNCASE_API inline result<runtime::runtime_tensor>
round(runtime::runtime_tensor &input, datatype_t dtype) noexcept {
    return impl::unary(input, dtype, unary_round);
}
/**
 * @brief unary_floor
 *
 * @param input runtime_tensor
 * @param dtype output tensor datatype
 * @return result<runtime::runtime_tensor>
 */
NNCASE_API inline result<runtime::runtime_tensor>
floor(runtime::runtime_tensor &input, datatype_t dtype) noexcept {
    return impl::unary(input, dtype, unary_floor);
}
/**
 * @brief unary_ceil
 *
 * @param input runtime_tensor
 * @param dtype output tensor datatype
 * @return result<runtime::runtime_tensor>
 */
NNCASE_API inline result<runtime::runtime_tensor>
ceil(runtime::runtime_tensor &input, datatype_t dtype) noexcept {
    return impl::unary(input, dtype, unary_ceil);
}
/**
 * @brief unary_abs
 *
 * @param input runtime_tensor
 * @param dtype output tensor datatype
 * @return result<runtime::runtime_tensor>
 */
NNCASE_API inline result<runtime::runtime_tensor>
abs(runtime::runtime_tensor &input, datatype_t dtype) noexcept {
    return impl::unary(input, dtype, unary_abs);
}
/**
 * @brief unary_neg
 *
 * @param input runtime_tensor
 * @param dtype output tensor datatype
 * @return result<runtime::runtime_tensor>
 */
NNCASE_API inline result<runtime::runtime_tensor>
neg(runtime::runtime_tensor &input, datatype_t dtype) noexcept {
    return impl::unary(input, dtype, unary_neg);
}

/**
 * @brief binary add
 *        temporary not support
 * @param input_a runtime_tensor
 * @param input_b runtime_tensor
 * @param dtype datatype, output tensor datatype
 * @return result<runtime_tensor>
 */
NNCASE_API inline result<runtime::runtime_tensor>
add(runtime::runtime_tensor &input_a, runtime::runtime_tensor &input_b,
    datatype_t dtype) noexcept {
    return impl::binary(input_a, input_b, dtype, binary_add);
}
/**
 * @brief binary sub
 *        temporary not support
 * @param input_a runtime_tensor
 * @param input_b runtime_tensor
 * @param dtype datatype, output tensor datatype
 * @return result<runtime_tensor>
 */
NNCASE_API inline result<runtime::runtime_tensor>
sub(runtime::runtime_tensor &input_a, runtime::runtime_tensor &input_b,
    datatype_t dtype) noexcept {
    return impl::binary(input_a, input_b, dtype, binary_sub);
}
/**
 * @brief binary mul
 *        temporary not support
 * @param input_a runtime_tensor
 * @param input_b runtime_tensor
 * @param dtype datatype, output tensor datatype
 * @return result<runtime_tensor>
 */
NNCASE_API inline result<runtime::runtime_tensor>
mul(runtime::runtime_tensor &input_a, runtime::runtime_tensor &input_b,
    datatype_t dtype) noexcept {
    return impl::binary(input_a, input_b, dtype, binary_mul);
}
/**
 * @brief binary div
 *        temporary not support
 * @param input_a runtime_tensor
 * @param input_b runtime_tensor
 * @param dtype datatype, output tensor datatype
 * @return result<runtime_tensor>
 */
NNCASE_API inline result<runtime::runtime_tensor>
div(runtime::runtime_tensor &input_a, runtime::runtime_tensor &input_b,
    datatype_t dtype) noexcept {
    return impl::binary(input_a, input_b, dtype, binary_div);
}
/**
 * @brief binary min
 *        temporary not support
 * @param input_a runtime_tensor
 * @param input_b runtime_tensor
 * @param dtype datatype, output tensor datatype
 * @return result<runtime_tensor>
 */
NNCASE_API inline result<runtime::runtime_tensor>
min(runtime::runtime_tensor &input_a, runtime::runtime_tensor &input_b,
    datatype_t dtype) noexcept {
    return impl::binary(input_a, input_b, dtype, binary_min);
}
/**
 * @brief binary max
 *        temporary not support
 * @param input_a runtime_tensor
 * @param input_b runtime_tensor
 * @param dtype datatype, output tensor datatype
 * @return result<runtime_tensor>
 */
NNCASE_API inline result<runtime::runtime_tensor>
max(runtime::runtime_tensor &input_a, runtime::runtime_tensor &input_b,
    datatype_t dtype) noexcept {
    return impl::binary(input_a, input_b, dtype, binary_max);
}

/**
 * @brief quantize float or bfloat tensor to uint8 or int8
 *
 * @param input runtime_tensor
 * @param dtype datatype, output tensor datatype
 * @return result<runtime_tensor>
 */
NNCASE_API inline result<runtime::runtime_tensor>
quantize(runtime::runtime_tensor &input, datatype_t dtype) noexcept {
    return impl::quantize(input, dtype);
}
/**
 * @brief dequantize uint8 or int8 tensor to float or bfloat
 *
 * @param input runtime_tensor
 * @param dtype datatype, output tensor datatype
 * @return result<runtime_tensor>
 */
NNCASE_API inline result<runtime::runtime_tensor>
dequantize(runtime::runtime_tensor &input, datatype_t dtype) noexcept {
    return impl::dequantize(input, dtype);
}

/**
 * @brief give bboxs, crop new tensor from current tensor.
 *
 * @param input
 * @param bbox runtime tensor, shape should be [1,1,roi_amounts,4], layout
 * should be [y0, x0, y1, x1]
 * @param out_h output tensor height
 * @param out_w output tensor width
 * @param resize_mode resize mode
 * @return result<runtime_tensor>
 */
NNCASE_API inline result<runtime::runtime_tensor>
crop(runtime::runtime_tensor &input, runtime::runtime_tensor &bbox,
     size_t out_h, size_t out_w, image_resize_mode_t resize_mode,
     bool align_corners, bool half_pixel_centers) noexcept {
    return impl::crop(input, bbox, out_h, out_w, resize_mode, align_corners,
                      half_pixel_centers);
}

/**
 * @brief resize tensor to new height or width
 *
 * @param input
 * @param out_h
 * @param out_w
 * @param resize_mode
 * @return result<runtime_tensor>
 */
NNCASE_API inline result<runtime::runtime_tensor>
resize(runtime::runtime_tensor &input, size_t out_h, size_t out_w,
       image_resize_mode_t resize_mode, bool align_corners,
       bool half_pixel_centers) noexcept {
    return impl::resize(input, out_h, out_w, resize_mode, align_corners,
                        half_pixel_centers);
}

/**
 * @brief padding value on the input tensor
 *        temporary not support
 * @param input
 * @param padding vector for padding param, from last to frist. eg. vector [
 * {2,3}, {1,3} ] mean pad {2,3} in last dim, pad {1,3} in last second dim
 * @param pad_mode
 * @param fill_v const fill value
 * @return result<runtime_tensor>
 */
NNCASE_API inline result<runtime::runtime_tensor>
pad(runtime::runtime_tensor &input, runtime_paddings_t &paddings,
    pad_mode_t pad_mode, float fill_v) noexcept {
    return impl::pad(input, paddings, pad_mode, fill_v);
}

} // namespace nncase::F