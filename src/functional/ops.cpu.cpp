#include <nncase/functional/ops.platform.h>

namespace nncase::F::impl
{

result<runtime::runtime_tensor> unary(NNCASE_UNUSED runtime::runtime_tensor &input, NNCASE_UNUSED datatype_t dtype, NNCASE_UNUSED unary_op_t op_type) noexcept
{
    return err(std::errc::not_supported);
}

result<runtime::runtime_tensor> binary(NNCASE_UNUSED runtime::runtime_tensor &input_a, NNCASE_UNUSED runtime::runtime_tensor &input_b, NNCASE_UNUSED datatype_t dtype, NNCASE_UNUSED binary_op_t op_type) noexcept
{
    return err(std::errc::not_supported);
}

result<runtime::runtime_tensor> quantize(NNCASE_UNUSED runtime::runtime_tensor &input, NNCASE_UNUSED datatype_t dtype) noexcept
{
    return err(std::errc::not_supported);
}

result<runtime::runtime_tensor> dequantize(NNCASE_UNUSED runtime::runtime_tensor &input, NNCASE_UNUSED datatype_t dtype) noexcept
{
    return err(std::errc::not_supported);
}

result<runtime::runtime_tensor> crop(NNCASE_UNUSED runtime::runtime_tensor &input, NNCASE_UNUSED runtime::runtime_tensor &bbox, NNCASE_UNUSED size_t out_h, NNCASE_UNUSED size_t out_w, NNCASE_UNUSED image_resize_mode_t resize_mode) noexcept
{
    return err(std::errc::not_supported);
}

result<runtime::runtime_tensor> resize(NNCASE_UNUSED runtime::runtime_tensor &input, NNCASE_UNUSED size_t out_h, NNCASE_UNUSED size_t out_w, NNCASE_UNUSED image_resize_mode_t resize_mode) noexcept
{
    return err(std::errc::not_supported);
}

result<runtime::runtime_tensor> pad(NNCASE_UNUSED runtime::runtime_tensor &input, NNCASE_UNUSED std::vector<padding> &padding, NNCASE_UNUSED pad_mode_t pad_mode, NNCASE_UNUSED float fill_v) noexcept
{
    return err(std::errc::not_supported);
}
}