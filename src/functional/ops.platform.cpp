#include <nncase/functional/ops.platform.h>

using namespace nncase;
using namespace nncase::F;

result<runtime::runtime_tensor> F::impl::unary(NNCASE_UNUSED runtime::runtime_tensor &input, NNCASE_UNUSED unary_op_t op_type) noexcept
{
    return err(std::errc::not_supported);
}