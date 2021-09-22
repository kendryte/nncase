#include <nncase/functional/ops.h>

#ifndef NNCASE_FUNCTIONAL_IMPL_PLATFORM_HEADER
#include <nncase/functional/ops.platform.h>
#else
#include NNCASE_FUNCTIONAL_IMPL_PLATFORM_HEADER
#endif

using namespace nncase;
using namespace nncase::F;
using namespace nncase::runtime;

NNCASE_API result<runtime::runtime_tensor> square(runtime::runtime_tensor input) noexcept
{
    return impl::unary(input, unary_square);
}

NNCASE_API result<runtime::runtime_tensor> sqrt(runtime::runtime_tensor input) noexcept
{
    return impl::unary(input, unary_sqrt);
}

NNCASE_API result<runtime::runtime_tensor> log(runtime::runtime_tensor input) noexcept
{
    return impl::unary(input, unary_log);
}

NNCASE_API result<runtime::runtime_tensor> exp(runtime::runtime_tensor input) noexcept
{
    return impl::unary(input, unary_exp);
}

NNCASE_API result<runtime::runtime_tensor> sin(runtime::runtime_tensor input) noexcept
{
    return impl::unary(input, unary_sin);
}

NNCASE_API result<runtime::runtime_tensor> cos(runtime::runtime_tensor input) noexcept
{
    return impl::unary(input, unary_cos);
}

NNCASE_API result<runtime::runtime_tensor> round(runtime::runtime_tensor input) noexcept
{
    return impl::unary(input, unary_round);
}

NNCASE_API result<runtime::runtime_tensor> floor(runtime::runtime_tensor input) noexcept
{
    return impl::unary(input, unary_floor);
}

NNCASE_API result<runtime::runtime_tensor> ceil(runtime::runtime_tensor input) noexcept
{
    return impl::unary(input, unary_ceil);
}

NNCASE_API result<runtime::runtime_tensor> abs(runtime::runtime_tensor input) noexcept
{
    return impl::unary(input, unary_abs);
}

NNCASE_API result<runtime::runtime_tensor> neg(runtime::runtime_tensor input) noexcept
{
    return impl::unary(input, unary_neg);
}
