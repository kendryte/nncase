/**
 * @file ops.platform.h
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

namespace nncase::F::impl {

result<runtime::runtime_tensor> unary(runtime::runtime_tensor &input,
                                      datatype_t dtype,
                                      unary_op_t op_type) noexcept;

result<runtime::runtime_tensor> binary(runtime::runtime_tensor &input_a,
                                       runtime::runtime_tensor &input_b,
                                       datatype_t dtype,
                                       binary_op_t op_type) noexcept;

result<runtime::runtime_tensor> quantize(runtime::runtime_tensor &input,
                                         datatype_t dtype) noexcept;

result<runtime::runtime_tensor> dequantize(runtime::runtime_tensor &input,
                                           datatype_t dtype) noexcept;

result<runtime::runtime_tensor>
crop(runtime::runtime_tensor &input, runtime::runtime_tensor &bbox,
     size_t out_h, size_t out_w, image_resize_mode_t resize_mode,
     bool align_corners, bool half_pixel_centers) noexcept;

result<runtime::runtime_tensor> resize(runtime::runtime_tensor &input,
                                       size_t out_h, size_t out_w,
                                       image_resize_mode_t resize_mode,
                                       bool align_corners,
                                       bool half_pixel_centers) noexcept;

result<runtime::runtime_tensor> pad(runtime::runtime_tensor &input,
                                    runtime_paddings_t &paddings,
                                    pad_mode_t pad_mode, float fill_v) noexcept;
} // namespace nncase::F::impl
