#include <ir/evaluator.h>
#include <ir/op_utils.h>
#include <ir/ops/binary.h>
#include <ir/ops/concat.h>
#include <ir/ops/conv2d.h>
#include <ir/ops/dequantize.h>
#include <ir/ops/fake_dequantize.h>
#include <ir/ops/fake_quantize.h>
#include <ir/ops/matmul.h>
#include <ir/ops/pad.h>
#include <ir/ops/quantize.h>
#include <ir/ops/reduce.h>
#include <ir/ops/reduce_window2d.h>
#include <ir/ops/resize_image.h>
#include <ir/ops/transpose.h>
#include <kernels/neutral/neutral_kernels.h>

using namespace nncase;
using namespace nncase::scheduler;
using namespace nncase::ir;
using namespace nncase::kernels;

#define ELEM_SIZE_IMPL(type, KERNEL)                            \
    switch (runtime::get_bytes(type))                           \
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

namespace
{
void nop_evaluator(ir::node &, evaluate_context &)
{
}
}

namespace nncase
{
namespace ir
{
    void register_neutral_evaluators()
    {
        register_evaluator(op_binary, [](ir::node &node, evaluate_context &context) {
            auto &rnode = static_cast<binary &>(node);

            assert(rnode.input_a().type() == dt_float32);
            assert(rnode.input_b().type() == dt_float32);
            auto input_a = context.memory_at<float>(rnode.input_a());
            auto input_b = context.memory_at<float>(rnode.input_b());
            auto output = context.memory_at<float>(rnode.output());

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
            default:
                throw std::runtime_error("Not supported binary");
            }
        });

        register_evaluator(op_concat, [](ir::node &node, evaluate_context &context) {
            auto &rnode = static_cast<concat &>(node);

            std::vector<uint8_t *> inputs;
            for (auto &&in : rnode.inputs())
                inputs.emplace_back(context.memory_at<uint8_t>(in).data());

            auto output = context.memory_at<uint8_t>(rnode.output());

            auto elem_size = (uint32_t)runtime::get_bytes(rnode.output().type());
            uint64_t inner_size, outer_size;
            get_concat_params(rnode.output().shape(), elem_size, rnode.axis(), inner_size, outer_size);
            neutral::concat(xtl::span<uint8_t *>(inputs), output.data(), rnode.concat_dims(), inner_size, outer_size);
        });

        register_evaluator(op_conv2d, [](ir::node &node, evaluate_context &context) {
            auto &rnode = static_cast<conv2d &>(node);

            assert(rnode.input().type() == dt_float32);
            auto input = context.memory_at<float>(rnode.input());
            auto output = context.memory_at<float>(rnode.output());

            neutral::conv2d(input.data(), output.data(), rnode.weights().data(), rnode.bias().data(), to(rnode.input().shape()),
                rnode.groups(), rnode.output_channels(), rnode.filter_h(), rnode.filter_w(), rnode.stride_h(), rnode.stride_w(),
                rnode.dilation_h(), rnode.dilation_w(), rnode.padding_h(), rnode.padding_w(), rnode.fused_activation());
        });

        register_evaluator(op_dequantize, [](ir::node &node, evaluate_context &context) {
            auto &rnode = static_cast<dequantize &>(node);

            auto input = context.memory_at<uint8_t>(rnode.input());
            auto output = context.memory_at<float>(rnode.output());

            neutral::dequantize(input.data(), output.data(), xt::compute_size(rnode.input().shape()), rnode.quant_param());
        });

        register_evaluator(op_fake_dequantize, [](ir::node &node, evaluate_context &context) {
            auto &rnode = static_cast<fake_dequantize &>(node);

            auto input = context.memory_at<float>(rnode.input());
            auto output = context.memory_at<float>(rnode.output());

            std::copy(input.begin(), input.end(), output.begin());
        });

        register_evaluator(op_fake_quantize, [](ir::node &node, evaluate_context &context) {
            auto &rnode = static_cast<fake_quantize &>(node);

            auto input = context.memory_at<float>(rnode.input());
            auto output = context.memory_at<float>(rnode.output());

            std::copy(input.begin(), input.end(), output.begin());
        });

        register_evaluator(op_matmul, [](ir::node &node, evaluate_context &context) {
            auto &rnode = static_cast<matmul &>(node);

            assert(rnode.input_a().type() == dt_float32);
            assert(rnode.input_b().type() == dt_float32);
            auto input_a = context.memory_at<float>(rnode.input_a());
            auto input_b = context.memory_at<float>(rnode.input_b());
            auto output = context.memory_at<float>(rnode.output());

            auto &a_shape = rnode.input_a().shape();
            auto &b_shape = rnode.input_b().shape();

            neutral::matmul(input_a.data(), input_b.data(), output.data(), rnode.bias().data(), a_shape[0], a_shape[1], b_shape[1], rnode.fused_activation());
        });

        register_evaluator(op_input_node, nop_evaluator);
        register_evaluator(op_output_node, nop_evaluator);
        register_evaluator(op_constant, nop_evaluator);

        register_evaluator(op_pad, [](ir::node &node, evaluate_context &context) {
            auto &rnode = static_cast<pad &>(node);

            auto input = context.memory_at<uint8_t>(rnode.input());
            auto output = context.memory_at<uint8_t>(rnode.output());

#define PAD_KERNEL(T) \
    kernels::neutral::pad(reinterpret_cast<const T *>(input.data()), reinterpret_cast<T *>(output.data()), to(rnode.input().shape()), to(rnode.paddings()), rnode.pad_value().as<T>());

            ELEM_SIZE_IMPL(rnode.input().type(), PAD_KERNEL);
#undef PAD_KERNEL
        });

        register_evaluator(op_quantize, [](ir::node &node, evaluate_context &context) {
            auto &rnode = static_cast<quantize &>(node);

            auto input = context.memory_at<float>(rnode.input());
            auto output = context.memory_at<uint8_t>(rnode.output());

            neutral::quantize(input.data(), output.data(), xt::compute_size(rnode.input().shape()), rnode.quant_param());
        });

        register_evaluator(op_reduce, [](ir::node &node, evaluate_context &context) {
            auto &rnode = static_cast<reduce &>(node);

            assert(rnode.input().type() == dt_float32);
            auto input = context.memory_at<float>(rnode.input());
            auto output = context.memory_at<float>(rnode.output());

            auto reduced_shape = get_reduced_shape(rnode.input().shape(), rnode.axis(), true);

            auto reduce = [&](auto reduce_op) {
                neutral::reduce(input.data(), output.data(), rnode.init_value(), to(rnode.input().shape()), to(reduced_shape), reduce_op);
            };

            switch (rnode.reduce_op())
            {
            case reduce_mean:
            {
                reduce([](auto a, auto b) { return a + b; });
                auto mul = (float)input.size() / output.size();
                neutral::unary(output.data(), output.data(), output.size(), [mul](auto a) { return a * mul; });
                break;
            }
            case reduce_min:
                reduce([](auto a, auto b) { return std::min(a, b); });
                break;
            case reduce_max:
                reduce([](auto a, auto b) { return std::max(a, b); });
                break;
            default:
                throw std::runtime_error("Not supported reduce");
            }
        });

        register_evaluator(op_reduce_window2d, [](ir::node &node, evaluate_context &context) {
            auto &rnode = static_cast<reduce_window2d &>(node);

            assert(rnode.input().type() == dt_float32);
            auto input = context.memory_at<float>(rnode.input());
            auto output = context.memory_at<float>(rnode.output());

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
                reduce([](auto a, auto b) { return std::min(a, b); }, [](auto v, auto k) { return v; });
                break;
            case reduce_max:
                reduce([](auto a, auto b) { return std::max(a, b); }, [](auto v, auto k) { return v; });
                break;
            default:
                throw std::runtime_error("Not supported reduce");
            }
        });

        register_evaluator(op_resize_image, [](ir::node &node, evaluate_context &context) {
            auto &rnode = static_cast<resize_image &>(node);

            if (rnode.mode() == image_resize_bilinear)
            {
                auto input = context.memory_at<float>(rnode.input());
                auto output = context.memory_at<float>(rnode.output());

                kernels::neutral::resize_bilinear(input.data(), output.data(), to(rnode.input().shape()), rnode.new_size()[0], rnode.new_size()[1], rnode.align_corners());
            }
            else
            {
                auto input = context.memory_at<uint8_t>(rnode.input());
                auto output = context.memory_at<uint8_t>(rnode.output());

#define RESIZE_NN_KERNEL(T) \
    kernels::neutral::resize_nearest_neighbor(reinterpret_cast<const T *>(input.data()), reinterpret_cast<T *>(output.data()), to(rnode.input().shape()), rnode.new_size()[0], rnode.new_size()[1]);

                ELEM_SIZE_IMPL(rnode.input().type(), RESIZE_NN_KERNEL);
#undef RESIZE_NN_KERNEL
            }
        });

        register_evaluator(op_transpose, [](ir::node &node, evaluate_context &context) {
            auto &rnode = static_cast<transpose &>(node);

            auto input = context.memory_at<uint8_t>(rnode.input());
            auto output = context.memory_at<uint8_t>(rnode.output());

            runtime_shape_t in_shape, perm;
            extend_transpose_shape(rnode.input().shape(), rnode.perm(), in_shape, perm);

#define TRANSPOSE_KERNEL(T) \
    neutral::transpose<T>(reinterpret_cast<const T *>(input.data()), reinterpret_cast<T *>(output.data()), in_shape, perm);

            ELEM_SIZE_IMPL(rnode.input().type(), TRANSPOSE_KERNEL);
        });
    }
}
}
