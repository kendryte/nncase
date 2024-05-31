#include "shape_infer.h"
#include <nncase/kernels/stackvm/tensor_ops.h>

using namespace nncase;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;

result<value_t> nncase::kernels::stackvm::conv2d_shape(
    value_t input, value_t weights, value_t padding, value_t stride,
    value_t dilation, [[maybe_unused]] value_t groups, value_t output,
    kernel_context &) {
    try_dims(in_shape, input);
    try_dims(w_shape, weights);
    try_strides(strides_value, stride);
    try_paddings(pads, padding);
    try_strides(strides, stride);
    try_strides(dilations, dilation);
    try_output(out_mem, output, dt_int64, dims_t{4});
    auto out_shape =
        conv2d_infer_shape(in_shape, w_shape, strides_value, dilations, pads);
    for (int i = 0; i < 4; ++i) {
        OUT_CAST(int64_t, out_mem)[i] = out_shape[i];
    }
    KERNEL_FINISH;
}

size_t compute_out_size(int input_size, int weights_size,
                        const strides_t &strides, dims_t out_paddings,
                        paddings_t paddings, const strides_t &dilations,
                        int offset) {
    return (strides[offset] * (input_size - 1L)) + out_paddings[offset] +
           (((weights_size - 1L) * dilations[offset]) + 1L) -
           paddings[offset].before - paddings[offset].after;
}

dims_t conv2d_transpose_infer_shape(std::span<const size_t> in_shape,
                                    std::span<const size_t> w_shape,
                                    const strides_t &strides,
                                    paddings_t paddings,
                                    const dims_t &outPadding,
                                    const strides_t &dilations, int group) {
    auto in = in_shape[0];
    auto ih = in_shape[2];
    auto iw = in_shape[3];
    auto oc = w_shape[0] * group;
    auto wh = w_shape[2];
    auto ww = w_shape[3];

    auto oh =
        compute_out_size(ih, wh, strides, outPadding, paddings, dilations, 0);
    auto ow =
        compute_out_size(iw, ww, strides, outPadding, paddings, dilations, 1);
    auto out_shape = dims_t{in, oc, oh, ow};
    return out_shape;
}

result<value_t> nncase::kernels::stackvm::conv2d_transpose_shape(
    value_t input, value_t weights, value_t stride, value_t dilation,
    value_t padding, value_t output_padding, value_t groups, value_t output,
    kernel_context &) {
    try_dims(input_shape, input);
    try_dims(weights_shape, weights);
    try_strides(strides_value, stride);
    try_paddings(pads, padding);
    try_dims(out_padding, output_padding);
    try_to_integer(groups_value, groups);
    try_strides(strides, stride);
    try_strides(dilations, dilation);

    auto out_shape =
        conv2d_transpose_infer_shape(input_shape, weights_shape, strides, pads,
                                     out_padding, dilations, groups_value);
    try_output(out_mem, output, dt_int64, dims_t{4});
    for (int i = 0; i < 4; ++i) {
        OUT_CAST(int64_t, out_mem)[i] = out_shape[i];
    }
    KERNEL_FINISH;
}

result<dims_t> to_dims(tensor shape) {
    try_dims(shape_value, shape);
    return ok(shape_value);
}

result<value_t> nncase::kernels::stackvm::broadcast_shape(value_t inputs,
                                                          value_t output,
                                                          kernel_context &) {
    try_tuple_input(tuple_mem, inputs);
    auto begin = inputs_tuple->fields().begin();
    auto out_shape = std::accumulate(
        std::next(begin), inputs_tuple->fields().end(),
        to_dims(begin->as<tensor>().unwrap()).unwrap(),
        [&](auto sum, auto field) {
            auto shape = to_dims(field.template as<tensor>().unwrap()).unwrap();
            auto result = kernels::detail::get_binary_output_shape(shape, sum);

            return dims_t(result.begin(), result.end());
        });
    try_output(out_mem, output, dt_int64, dims_t{out_shape.size()});
    for (int i = 0; i < out_shape.size(); ++i) {
        OUT_CAST(int64_t, out_mem)[i] = out_shape[i];
    }

    KERNEL_FINISH;
}

#define WRITE_OUT_SHAPE                                                        \
    try_output(out_mem, output, dt_int64, dims_t{out_shape.size()});           \
    for (int i = 0; i < out_shape.size(); ++i) {                               \
        OUT_CAST(int64_t, out_mem)[i] = out_shape[i];                          \
    }

result<value_t> nncase::kernels::stackvm::mat_mul_shape(value_t lhs,
                                                        value_t rhs,
                                                        value_t output,
                                                        kernel_context &) {
    try_dims(lhs_shape, lhs);
    try_dims(rhs_shape, rhs);
    try_var(out_shape, matmul_infer_shape(lhs_shape, rhs_shape));
    WRITE_OUT_SHAPE;
    KERNEL_FINISH;
}

inline int get_windowed_output_size(int size, int filter, int stride,
                                    int dilation, bool same, bool ceilMode) {
    auto effectiveFilterSize = ((filter - 1) * dilation) + 1;
    auto falseBranch = !ceilMode
                           ? ((size - effectiveFilterSize + stride) / stride)
                           : ceil(size - effectiveFilterSize + stride / stride);
    auto trueBranch = (size + stride - 1) / stride;
    return same ? trueBranch : falseBranch;
}

inline padding get_windowed_padding(int32_t input_size, int32_t output_size,
                                    int32_t filter, int32_t stride,
                                    int32_t dilation, bool lower) {
    auto effective_filter_size = (filter - 1) * dilation + 1;
    int padding = std::max(0, (output_size - 1) * stride +
                                  effective_filter_size - input_size);
    auto before = padding / 2;
    auto after = padding - padding / 2;
    if (lower) {
        return {std::max(before, after), std::min(before, after)};
    }
    return {before, after};
}

result<value_t> nncase::kernels::stackvm::get_paddings(
    value_t input_shape, value_t weights_shape, value_t strides,
    value_t dilations, value_t same, value_t lower, value_t output,
    [[maybe_unused]] kernel_context &) {
    try_dims(in_shape, input_shape);
    try_dims(w_shape, weights_shape);
    try_strides(strides_value, strides);
    try_strides(dilations_value, dilations);
    try_to_scalar_v(same, bool);
    try_to_scalar_v(lower, bool);
    auto out_h =
        get_windowed_output_size(in_shape[2], w_shape[2], strides_value[0],
                                 dilations_value[0], same_value, false);
    auto out_w =
        get_windowed_output_size(in_shape[3], w_shape[3], strides_value[1],
                                 dilations_value[1], same_value, false);
    auto pad_h =
        get_windowed_padding(in_shape[2], out_h, w_shape[2], strides_value[0],
                             dilations_value[0], lower_value);
    auto pad_w =
        get_windowed_padding(in_shape[3], out_w, w_shape[3], strides_value[1],
                             dilations_value[1], lower_value);
    auto out_shape = dims_t{2, 2};
    try_out_mem(output, dt_int64, out_shape);
    OUT_CAST(int64_t, output_mem)[0] = pad_h.before;
    OUT_CAST(int64_t, output_mem)[1] = pad_h.after;
    OUT_CAST(int64_t, output_mem)[2] = pad_w.before;
    OUT_CAST(int64_t, output_mem)[3] = pad_w.after;
    KERNEL_FINISH;
}

result<value_t> nncase::kernels::stackvm::reshape_shape(value_t input_shape,
                                                        value_t shape,
                                                        value_t output,
                                                        kernel_context &) {
    try_dims(in_shape, input_shape);
    try_axes(shape_value, shape);
    auto out_shape = reshape_shape_infer(in_shape, shape_value);
    WRITE_OUT_SHAPE;
    KERNEL_FINISH;
}

result<value_t>
nncase::kernels::stackvm::transpose_shape(value_t input_shape, value_t perm,
                                          value_t output,
                                          [[maybe_unused]] kernel_context &) {
    try_dims(in_shape, input_shape);
    try_dims(perm_value, perm);
    auto out_shape = transpose_infer_shape(in_shape, perm_value);
    WRITE_OUT_SHAPE;
    KERNEL_FINISH;
}

result<value_t>
nncase::kernels::stackvm::squeeze_shape(value_t input_shape, value_t dim,
                                        value_t output,
                                        [[maybe_unused]] kernel_context &) {
    try_dims(in_shape, input_shape);
    try_positive_axes(dim_value, dim, in_shape.size());
    auto out_shape = squeeze_infer_shape(in_shape, dim_value);
    WRITE_OUT_SHAPE;
    KERNEL_FINISH;
}

result<value_t>
nncase::kernels::stackvm::unsqueeze_shape(value_t input_shape, value_t dim,
                                          value_t output,
                                          [[maybe_unused]] kernel_context &) {
    try_dims(in_shape, input_shape);
    try_axes(dim_value, dim);
    auto out_shape = unsqueeze_infer_shape(in_shape, dim_value);
    WRITE_OUT_SHAPE;
    KERNEL_FINISH;
}