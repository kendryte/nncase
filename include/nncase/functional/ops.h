#pragma once
#include <nncase/kernels/kernel_context.h>
#include <nncase/runtime/datatypes.h>
#include <nncase/runtime/error.h>
#include <nncase/runtime/result.h>
#include <nncase/runtime/runtime_tensor.h>

namespace nncase::F
{

NNCASE_API result<runtime::runtime_tensor> square(runtime::runtime_tensor input) noexcept;

NNCASE_API result<runtime::runtime_tensor> sqrt(runtime::runtime_tensor input) noexcept;

NNCASE_API result<runtime::runtime_tensor> log(runtime::runtime_tensor input) noexcept;

NNCASE_API result<runtime::runtime_tensor> exp(runtime::runtime_tensor input) noexcept;

NNCASE_API result<runtime::runtime_tensor> sin(runtime::runtime_tensor input) noexcept;

NNCASE_API result<runtime::runtime_tensor> cos(runtime::runtime_tensor input) noexcept;

NNCASE_API result<runtime::runtime_tensor> round(runtime::runtime_tensor input) noexcept;

NNCASE_API result<runtime::runtime_tensor> floor(runtime::runtime_tensor input) noexcept;

NNCASE_API result<runtime::runtime_tensor> ceil(runtime::runtime_tensor input) noexcept;

NNCASE_API result<runtime::runtime_tensor> abs(runtime::runtime_tensor input) noexcept;

NNCASE_API result<runtime::runtime_tensor> neg(runtime::runtime_tensor input) noexcept;

}