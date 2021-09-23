#pragma once
#include <nncase/runtime/runtime_tensor.h>

namespace nncase::F::impl
{

result<runtime::runtime_tensor> unary(runtime::runtime_tensor &input, datatype_t dtype, unary_op_t op_type) noexcept;

result<runtime::runtime_tensor> binary(runtime::runtime_tensor &input_a, runtime::runtime_tensor &input_b, datatype_t dtype, binary_op_t op_type) noexcept;

result<runtime::runtime_tensor> quantize(runtime::runtime_tensor &input, datatype_t dtype) noexcept;

result<runtime::runtime_tensor> dequantize(runtime::runtime_tensor &input, datatype_t dtype) noexcept;

result<runtime::runtime_tensor> crop(runtime::runtime_tensor &input, runtime::runtime_tensor &bbox, size_t out_h, size_t out_w, image_resize_mode_t resize_mode) noexcept;

result<runtime::runtime_tensor> resize(runtime::runtime_tensor &input, size_t out_h, size_t out_w, image_resize_mode_t resize_mode) noexcept;

result<runtime::runtime_tensor> pad(runtime::runtime_tensor &input, std::vector<padding> &padding, pad_mode_t pad_mode, float fill_v) noexcept;
}
