#pragma once
#include <nncase/runtime/runtime_tensor.h>

namespace nncase::F::impl
{

result<runtime::runtime_tensor> unary(runtime::runtime_tensor &input, datatype_t dtype, unary_op_t op_type) noexcept;

}
