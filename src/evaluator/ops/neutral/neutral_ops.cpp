/* Copyright 2020 Canaan Inc.
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
 */
#include <nncase/ir/evaluator.h>
#include <nncase/ir/op_utils.h>
#include <nncase/ir/ops/binary.h>
#include <nncase/ir/ops/bitcast.h>
#include <nncase/ir/ops/concat.h>
#include <nncase/ir/ops/conv2d.h>
#include <nncase/ir/ops/conv2d_transpose.h>
#include <nncase/ir/ops/dequantize.h>
#include <nncase/ir/ops/matmul.h>
//#include <nncase/ir/ops/nnil_method.h>
#include <nncase/ir/ops/clamp.h>
#include <nncase/ir/ops/convert.h>
#include <nncase/ir/ops/pad.h>
#include <nncase/ir/ops/quantize.h>
#include <nncase/ir/ops/reduce.h>
#include <nncase/ir/ops/reduce_window2d.h>
#include <nncase/ir/ops/resize_image.h>
#include <nncase/ir/ops/slice.h>
#include <nncase/ir/ops/table_lookup.h>
#include <nncase/ir/ops/transpose.h>
#include <nncase/ir/ops/unary.h>
#include <nncase/ir/runtime_type_utils.h>
#include <nncase/kernels/neutral/neutral_kernels.h>

using namespace nncase;
using namespace nncase::schedule;
using namespace nncase::ir;
using namespace nncase::kernels;

#define ELEM_SIZE_IMPL(type, KERNEL)                            \
    switch (get_bytes(type))                                    \
    {                                                           \
    case 1:                                                     \
        KERNEL(uint8_t);                                        \
        break;                                                  \
    case 2:                                                     \
        KERNEL(uint16_t);                                       \
        break;                                                  \
    case 4:                                                     \
        KERNEL(uint32_t);                                       \
        break;                                                  \
    default:                                                    \
        throw std::runtime_error("Not supported element type"); \
    }

#define FP_OR_Q_IMPL(type, KERNEL)                              \
    switch (type)                                               \
    {                                                           \
    case dt_float32:                                            \
        KERNEL(float);                                          \
        break;                                                  \
    case dt_int8:                                               \
        KERNEL(uint8_t);                                        \
        break;                                                  \
    case dt_uint8:                                              \
        KERNEL(uint8_t);                                        \
        break;                                                  \
    default:                                                    \
        throw std::runtime_error("Not supported element type"); \
    }

namespace
{
void nop_evaluator(ir::node &, evaluator &)
{
}
}

namespace nncase::ir
{
template <datatype_t src_type>
void ConvertIfDestTypeMatches(convert &node, evaluator &context)
{
    switch (node.new_type())
    {
#define CONVERT_IF_TYPES_MATCH(type)                                                      \
    case (type):                                                                          \
        ConvertIfTypesMatch<to_cpp_type_t<src_type>, to_cpp_type_t<type>>(node, context); \
        break;
        CONVERT_IF_TYPES_MATCH(dt_int8)
        CONVERT_IF_TYPES_MATCH(dt_uint8)
        CONVERT_IF_TYPES_MATCH(dt_uint32)
        CONVERT_IF_TYPES_MATCH(dt_float32)
        CONVERT_IF_TYPES_MATCH(dt_bfloat16)
#undef CONVERT_IF_TYPES_MATCH
    default:
        throw std::runtime_error("Not supported element type");
        break;
    }
}

template <typename src_type, typename dst_type>
void ConvertIfTypesMatch(convert &node, evaluator &context)
{
    auto input = context.memory_at<src_type>(node.input()).span;
    auto output = context.memory_at<dst_type>(node.output()).span;

    auto input_ptr = input.data();
    auto output_ptr = output.data();
    for (size_t i = 0; i < input.size(); i++)
    {
        output_ptr[i] = static_cast<dst_type>(input_ptr[i]);
    }
}

void register_neutral_evaluators()
{
    register_evaluator(op_input_node, nop_evaluator);
    register_evaluator(op_output_node, nop_evaluator);
    register_evaluator(op_ignore_node, nop_evaluator);
    register_evaluator(op_constant, nop_evaluator);

    register_evaluator(op_binary, [](ir::node &node, evaluator &context) {
        auto &rnode = static_cast<binary &>(node);

        assert(rnode.input_a().type() == dt_float32);
        assert(rnode.input_b().type() == dt_float32);
        auto input_a = context.memory_at<float>(rnode.input_a()).span;
        auto input_b = context.memory_at<float>(rnode.input_b()).span;
        auto output = context.memory_at<float>(rnode.output()).span;

        auto binary = [&](auto binary_op) {
            neutral::binary(input_a.data(), input_b.data(), output.data(), to(rnode.input_a().shape()), to(rnode.input_b().shape()),
                to(rnode.output().shape()), rnode.fused_activation(), binary_op);
        };

        switch (rnode.binary_op())
        {
        case binary_add:
            binary([](auto a, auto b) { return a + b; });
            break;
        case binary_sub:
            binary([](auto a, auto b) { return a - b; });
            break;
        case binary_mul:
            binary([](auto a, auto b) { return a * b; });
            break;
        case binary_div:
            binary([](auto a, auto b) { return a / b; });
            break;
        case binary_min:
            binary([](auto a, auto b) { return std::min(a, b); });
            break;
        case binary_max:
            binary([](auto a, auto b) { return std::max(a, b); });
            break;
        case binary_pow:
            binary([](auto a, auto b) { return std::pow(a, b); });
            break;
        case binary_floor_div:
            binary([](auto a, auto b) { return std::floor(std::divides<float>()(a, b)); });
            break;
        case binary_floor_mod:
            binary([](auto a, auto b) {
                auto trunc_mod = std::fmod(a, b);
                return trunc_mod != 0 && ((b < 0) != (trunc_mod < 0))
                    ? trunc_mod + b
                    : trunc_mod;
            });
            break;
        default:
            throw std::runtime_error("Not supported binary");
        }
    });

    register_evaluator(op_concat, [](ir::node &node, evaluator &context) {
        auto &rnode = static_cast<concat &>(node);

        std::vector<uint8_t *> inputs;
        for (auto in : rnode.inputs())
            inputs.emplace_back(context.memory_at<uint8_t>(*in).span.data());

        auto output = context.memory_at<uint8_t>(rnode.output()).span;

        auto elem_size = (uint32_t)get_bytes(rnode.output().type());
        uint64_t inner_size, outer_size;
        get_concat_params(rnode.output().shape(), elem_size, (size_t)rnode.axis(), inner_size, outer_size);
        neutral::concat(xtl::span<uint8_t *>(inputs), output.data(), rnode.concat_dims(), inner_size, outer_size);
    });

    register_evaluator(op_conv2d, [](ir::node &node, evaluator &context) {
        auto &rnode = static_cast<conv2d &>(node);

        assert(rnode.input().type() == dt_float32);
        auto input = context.memory_at<float>(rnode.input()).span;
        auto weights = context.memory_at<float>(rnode.input()).span;
        auto bias = context.memory_at<float>(rnode.input()).span;
        auto output = context.memory_at<float>(rnode.output()).span;

        neutral::conv2d(input.data(), output.data(), weights.data(), bias.data(), to(rnode.input().shape()),
            rnode.groups(), rnode.output_channels(), rnode.filter_h(), rnode.filter_w(), rnode.stride_h(), rnode.stride_w(),
            rnode.dilation_h(), rnode.dilation_w(), rnode.padding_h(), rnode.padding_w(), rnode.fused_activation());
    });

    register_evaluator(op_conv2d_transpose, [](ir::node &node, evaluator &context) {
        auto &rnode = static_cast<conv2d_transpose &>(node);

        assert(rnode.input().type() == dt_float32);
        auto input = context.memory_at<float>(rnode.input()).span;
        auto weights = context.memory_at<float>(rnode.input()).span;
        auto bias = context.memory_at<float>(rnode.input()).span;
        auto output = context.memory_at<float>(rnode.output()).span;

        neutral::conv2d_transpose(input.data(), output.data(), weights.data(), bias.data(), to(rnode.input().shape()),
            rnode.groups(), to(rnode.output().shape()), rnode.filter_h(), rnode.filter_w(), rnode.stride_h(), rnode.stride_w(),
            rnode.dilation_h(), rnode.dilation_w(), rnode.padding_h(), rnode.padding_w(), rnode.fused_activation());
    });

    register_evaluator(op_dequantize, [](ir::node &node, evaluator &context) {
        auto &rnode = static_cast<dequantize &>(node);

        auto output = context.memory_at<float>(rnode.output()).span;

        switch (rnode.input().type())
        {
#define DEQUANTIZE(type)                                                                                                \
    case type:                                                                                                          \
    {                                                                                                                   \
        auto input = context.memory_at<to_cpp_type_t<type>>(rnode.input()).span;                                        \
        neutral::dequantize(input.data(), output.data(), xt::compute_size(rnode.input().shape()), rnode.quant_param()); \
        break;                                                                                                          \
    }
            DEQUANTIZE(dt_uint8)
            DEQUANTIZE(dt_int8)
            DEQUANTIZE(dt_int32)
        default:
            assert(false && "not supported type!");

#undef DEQUANTIZE
        }
    });

    register_evaluator(op_matmul, [](ir::node &node, evaluator &context) {
        auto &rnode = static_cast<matmul &>(node);

        assert(rnode.input_a().type() == dt_float32);
        assert(rnode.input_b().type() == dt_float32);
        auto input_a = context.memory_at<float>(rnode.input_a()).span;
        auto input_b = context.memory_at<float>(rnode.input_b()).span;
        auto output = context.memory_at<float>(rnode.output()).span;
        auto bias = context.memory_at<float>(rnode.bias()).span;

        auto &a_shape = rnode.input_a().shape();
        auto &b_shape = rnode.input_b().shape();

        neutral::matmul(input_a.data(), input_b.data(), output.data(), bias.data(), (int32_t)a_shape[0], (int32_t)a_shape[1], (int32_t)b_shape[1], rnode.fused_activation());
    });

    register_evaluator(op_pad, [](ir::node &node, evaluator &context) {
        auto &rnode = static_cast<pad &>(node);

        auto input = context.memory_at<uint8_t>(rnode.input()).span;
        auto output = context.memory_at<uint8_t>(rnode.output()).span;

#define PAD_KERNEL(T) \
    kernels::neutral::pad(reinterpret_cast<const T *>(input.data()), reinterpret_cast<T *>(output.data()), to(rnode.input().shape()), to(rnode.paddings()), rnode.pad_value().as<T>());

        ELEM_SIZE_IMPL(rnode.input().type(), PAD_KERNEL);
#undef PAD_KERNEL
    });

    register_evaluator(op_quantize, [](ir::node &node, evaluator &context) {
        auto &rnode = static_cast<quantize &>(node);

        auto input = context.memory_at<float>(rnode.input()).span;
        auto output = context.memory_at<uint8_t>(rnode.output()).span;

        neutral::quantize(input.data(), output.data(), xt::compute_size(rnode.input().shape()), rnode.quant_param());
    });

    register_evaluator(op_reduce, [](ir::node &node, evaluator &context) {
        auto &rnode = static_cast<reduce &>(node);

        assert(rnode.input().type() == dt_float32);
        auto input = context.memory_at<float>(rnode.input()).span;
        auto output = context.memory_at<float>(rnode.output()).span;

        auto reduced_shape = get_reduced_shape(rnode.input().shape(), rnode.axis(), true);

        auto reduce = [&](auto reduce_op) {
            neutral::reduce(input.data(), output.data(), rnode.init_value(), to(rnode.input().shape()), to(reduced_shape), reduce_op);
        };

        switch (rnode.reduce_op())
        {
        case reduce_mean:
        {
            reduce([](auto a, auto b) { return a + b; });
            auto mul = (float)output.size() / input.size();
            neutral::unary(output.data(), output.data(), output.size(), [mul](auto a) { return a * mul; });
            break;
        }
        case reduce_min:
            reduce([](auto a, auto b) { return std::min(a, b); });
            break;
        case reduce_max:
            reduce([](auto a, auto b) { return std::max(a, b); });
            break;
        case reduce_sum:
            reduce([](auto a, auto b) { return a + b; });
            break;
        default:
            throw std::runtime_error("Not supported reduce");
        }
    });

    register_evaluator(op_reduce_window2d, [](ir::node &node, evaluator &context) {
        auto &rnode = static_cast<reduce_window2d &>(node);

        assert(rnode.input().type() == dt_float32);
        auto input = context.memory_at<float>(rnode.input()).span;
        auto output = context.memory_at<float>(rnode.output()).span;

        auto reduce = [&](auto binary_op, auto output_op) {
            neutral::reduce_window2d(
                input.data(), output.data(), rnode.init_value(), to(rnode.input().shape()), rnode.filter_h(), rnode.filter_w(),
                rnode.stride_h(), rnode.stride_w(), rnode.dilation_h(), rnode.dilation_w(), rnode.padding_h(), rnode.padding_w(), rnode.fused_activation(),
                binary_op, output_op);
        };

        switch (rnode.reduce_op())
        {
        case reduce_mean:
            reduce([](auto a, auto b) { return a + b; }, [](auto v, auto k) { return v / k; });
            break;
        case reduce_min:
            reduce([](auto a, auto b) { return std::min(a, b); }, [](auto v, auto) { return v; });
            break;
        case reduce_max:
            reduce([](auto a, auto b) { return std::max(a, b); }, [](auto v, auto) { return v; });
            break;
        case reduce_sum:
            reduce([](auto a, auto b) { return a + b; }, [](auto v, auto) { return v; });
            break;
        default:
            throw std::runtime_error("Not supported reduce");
        }
    });

    register_evaluator(op_bitcast, [](ir::node &node, evaluator &context) {
        auto &rnode = static_cast<bitcast &>(node);

        auto input = context.memory_at<uint8_t>(rnode.input()).span;
        auto output = context.memory_at<uint8_t>(rnode.output()).span;

        std::copy(input.begin(), input.end(), output.begin());
    });

    register_evaluator(op_resize_image, [](ir::node &node, evaluator &context) {
        auto &rnode = static_cast<resize_image &>(node);

        if (rnode.mode() == image_resize_bilinear)
        {
            auto input = context.memory_at<uint8_t>(rnode.input()).span;
            auto output = context.memory_at<uint8_t>(rnode.output()).span;

#define RESIZE_BL_KERNEL(T) \
    kernels::neutral::resize_bilinear(reinterpret_cast<const T *>(input.data()), reinterpret_cast<T *>(output.data()), to(rnode.input().shape()), rnode.new_size()[0], rnode.new_size()[1], rnode.align_corners());

            FP_OR_Q_IMPL(rnode.input().type(), RESIZE_BL_KERNEL);
#undef RESIZE_BL_KERNEL
        }
        else
        {
            auto input = context.memory_at<uint8_t>(rnode.input()).span;
            auto output = context.memory_at<uint8_t>(rnode.output()).span;

#define RESIZE_NN_KERNEL(T) \
    kernels::neutral::resize_nearest_neighbor(reinterpret_cast<const T *>(input.data()), reinterpret_cast<T *>(output.data()), to(rnode.input().shape()), rnode.new_size()[0], rnode.new_size()[1]);

            FP_OR_Q_IMPL(rnode.input().type(), RESIZE_NN_KERNEL);
#undef RESIZE_NN_KERNEL
        }
    });

    register_evaluator(op_slice, [](ir::node &node, evaluator &context) {
        auto &rnode = static_cast<slice &>(node);

        auto input = context.memory_at<uint8_t>(rnode.input()).span;
        auto output = context.memory_at<uint8_t>(rnode.output()).span;

#define STRIDED_SLICE_KERNEL(T)                                                                                \
    neutral::strided_slice<T>(reinterpret_cast<const T *>(input.data()), reinterpret_cast<T *>(output.data()), \
        to(rnode.input().shape()), to(rnode.begin(), 0), to(rnode.end()), to(rnode.strides()));

        ELEM_SIZE_IMPL(rnode.input().type(), STRIDED_SLICE_KERNEL);
    });

    register_evaluator(op_transpose, [](ir::node &node, evaluator &context) {
        auto &rnode = static_cast<transpose &>(node);

        auto input = context.memory_at<uint8_t>(rnode.input()).span;
        auto output = context.memory_at<uint8_t>(rnode.output()).span;

        runtime_shape_t in_shape, perm;
        extend_transpose_shape(rnode.input().shape(), rnode.perm(), in_shape, perm);

#define TRANSPOSE_KERNEL(T) \
    neutral::transpose<T>(reinterpret_cast<const T *>(input.data()), reinterpret_cast<T *>(output.data()), in_shape, perm);

        ELEM_SIZE_IMPL(rnode.input().type(), TRANSPOSE_KERNEL);
    });

    register_evaluator(op_unary, [](ir::node &node, evaluator &context) {
        auto &rnode = static_cast<unary &>(node);

        assert(rnode.input().type() == dt_float32);
        auto input = context.memory_at<float>(rnode.input()).span;
        auto output = context.memory_at<float>(rnode.output()).span;

        auto unary = [&](auto unary_op) {
            neutral::unary(input.data(), output.data(), input.size(), unary_op);
        };

        switch (rnode.unary_op())
        {
        case unary_abs:
            unary([](auto a) { return fabs(a); });
            break;
        case unary_ceil:
            unary([](auto a) { return ceilf(a); });
            break;
        case unary_cos:
            unary([](auto a) { return cosf(a); });
            break;
        case unary_exp:
            unary([](auto a) { return expf(a); });
            break;
        case unary_floor:
            unary([](auto a) { return floorf(a); });
            break;
        case unary_log:
            unary([](auto a) { return logf(a); });
            break;
        case unary_neg:
            unary([](auto a) { return -a; });
            break;
        case unary_round:
            unary([](auto a) {
#if 0
                return round(a);
#else
                // bankers rounding method for tensorflow/tflite
                auto floor_val = std::floor(a);
                auto diff = a - floor_val;
                if ((diff < 0.5f) ||
                    ((diff == 0.5f) && (static_cast<int>(floor_val) % 2 == 0))) {
                    return floor_val;
                } else {
                    return floor_val = floor_val + 1.0f;
                }
#endif
            });
            break;
        case unary_rsqrt:
            unary([](auto a) { return 1.f / sqrtf(a); });
            break;
        case unary_sin:
            unary([](auto a) { return sinf(a); });
            break;
        case unary_sqrt:
            unary([](auto a) { return sqrt(a); });
            break;
        case unary_square:
            unary([](auto a) { return a * a; });
            break;
        case unary_tanh:
            unary([](auto a) { return tanh(a); });
            break;
        default:
            throw std::runtime_error("Not supported unary");
        }
    });

    register_evaluator(op_table_lookup1d, [](ir::node &node, evaluator &context) {
        auto &rnode = static_cast<table_lookup1d &>(node);

        assert(rnode.input().type() == dt_uint8);
        auto input = context.memory_at<uint8_t>(rnode.input()).span;
        auto table = context.memory_at<uint8_t>(rnode.table()).span;
        auto output = context.memory_at<uint8_t>(rnode.output()).span;

        kernels::neutral::table_lookup1d(input.data(), output.data(), input.size(), table.data());
    });

    register_evaluator(op_clamp, [](ir::node &node, evaluator &context) {
        auto &rnode = static_cast<clamp &>(node);

        assert(rnode.input().type() == dt_float32);
        auto input = context.memory_at<float>(rnode.input()).span;
        auto input_low = context.memory_at<float>(rnode.input_low()).span;
        auto input_high = context.memory_at<float>(rnode.input_high()).span;
        auto output = context.memory_at<float>(rnode.output()).span;

        const float *input_ptr = input.data();
        float low = input_low.data()[0];
        float high = input_high.data()[0];
        float *output_ptr = output.data();
        for (size_t i = 0; i < input.size(); i++)
        {
            output_ptr[i] = std::clamp(input_ptr[i], low, high);
        }
    });

    register_evaluator(op_convert, [](ir::node &node, evaluator &context) {
        auto &rnode = static_cast<convert &>(node);
        switch (rnode.input().type())
        {
#define CONVERT_IF_DEST_TYPE_MATCHES(type)              \
    case (type):                                        \
        ConvertIfDestTypeMatches<type>(rnode, context); \
        break;
            CONVERT_IF_DEST_TYPE_MATCHES(dt_int8)
            CONVERT_IF_DEST_TYPE_MATCHES(dt_uint8)
            CONVERT_IF_DEST_TYPE_MATCHES(dt_uint32)
            CONVERT_IF_DEST_TYPE_MATCHES(dt_float32)
            CONVERT_IF_DEST_TYPE_MATCHES(dt_bfloat16)
#undef CONVERT_IF_DEST_TYPE_MATCHES
        default:
            throw std::runtime_error("Not supported element type");
            break;
        }
    });
}

}
