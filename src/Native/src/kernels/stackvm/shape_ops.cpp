#include "shape_infer.h"
#include <nncase/kernels/stackvm/tensor_ops.h>

using namespace nncase;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;

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

#define WRITE_OUT_SHAPE                                                        \
    try_output(out_mem, output, dt_int64, dims_t{out_shape.size()});           \
    for (int i = 0; i < out_shape.size(); ++i) {                               \
        OUT_CAST(int64_t, out_mem)[i] = out_shape[i];                          \
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
