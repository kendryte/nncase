#pragma once
#include <nncase/kernels/kernel_context.h>
#include <nncase/runtime/datatypes.h>
#include <nncase/runtime/error.h>
#include <nncase/runtime/result.h>
#include <nncase/runtime/runtime_tensor.h>

namespace nncase::F::impl
{

result<runtime::runtime_tensor> unary(runtime::runtime_tensor &input, unary_op_t op_type) noexcept;

}
