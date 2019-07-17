#include <kernels/neutral/neutral_kernels.h>
#include <runtime/kernel_registry.h>
#include <targets/neutral/neutral_ops_body.h>

using namespace nncase;
using namespace nncase::runtime;

#define ELEM_SIZE_IMPL(type, KERNEL)  \
    switch (runtime::get_bytes(type)) \
    {                                 \
    case 1:                           \
        KERNEL(uint8_t);              \
        break;                        \
    case 2:                           \
        KERNEL(uint16_t);             \
        break;                        \
    case 4:                           \
        KERNEL(uint32_t);             \
        break;                        \
    default:                          \
        return kcr_error;             \
    }

namespace nncase
{
namespace targets
{
    namespace neutral
    {
        kernel_call_result binary(binary_options &options, interpreter &interpreter, interpreter_step_t step)
        {
            auto input_a = interpreter.memory_at<float>(options.input_a);
            auto input_b = interpreter.memory_at<float>(options.input_b);
            auto output = interpreter.memory_at<float>(options.output);

            auto binary = [&](auto op) {
                kernels::neutral::binary(input_a.data(), input_b.data(), output.data(), options.in_a_shape, options.in_b_shape, options.out_shape, options.fused_activation, op);
            };

            switch (options.binary_op)
            {
            case binary_add:
                binary([](auto a, auto b) { return a + b; });
                return kcr_done;
            case binary_sub:
                binary([](auto a, auto b) { return a - b; });
                return kcr_done;
            case binary_mul:
                binary([](auto a, auto b) { return a * b; });
                return kcr_done;
            case binary_div:
                binary([](auto a, auto b) { return a / b; });
                return kcr_done;
            default:
                return kcr_error;
            }
        }

        kernel_call_result concat(concat_options &options, interpreter &interpreter, interpreter_step_t step)
        {
            auto output = interpreter.memory_at<uint8_t>(options.output);
            xtl::span<const uint8_t *> inputs = { (const uint8_t **)alloca(options.inputs_count * sizeof(uint8_t *)), options.inputs_count };
            for (size_t i = 0; i < options.inputs_count; i++)
                inputs[i] = interpreter.memory_at<uint8_t>(options.inputs[i]).data();
            kernels::neutral::concat(inputs, output.data(), options.dims, options.inner_size, options.outer_size);
            return kcr_done;
        }

        kernel_call_result conv2d(conv2d_options &options, interpreter &interpreter, interpreter_step_t step)
        {
            auto input = interpreter.memory_at<float>(options.input);
            auto output = interpreter.memory_at<float>(options.output);
            kernels::neutral::conv2d(input.data(), output.data(), options.weights.data(), options.bias.data(), options.in_shape, options.groups, options.out_channels, options.filter_h,
                options.filter_w, options.stride_h, options.stride_w, options.dilation_h, options.dilation_w, options.padding_h, options.padding_w, options.fused_activation);
            return kcr_done;
        }

        kernel_call_result dequantize(dequantize_options &options, interpreter &interpreter, interpreter_step_t step)
        {
            auto input = interpreter.memory_at<uint8_t>(options.input);
            auto output = interpreter.memory_at<float>(options.output);

            kernels::neutral::dequantize(input.data(), output.data(), input.size(), options.quant_param);
            return kcr_done;
        }

        kernel_call_result matmul(matmul_options &options, interpreter &interpreter, interpreter_step_t step)
        {
            auto input_a = interpreter.memory_at<float>(options.input_a);
            auto input_b = interpreter.memory_at<float>(options.input_b);
            auto output = interpreter.memory_at<float>(options.output);
            kernels::neutral::matmul(input_a.data(), input_b.data(), output.data(), options.bias.data(), options.a_rows, options.a_cols, options.b_cols, options.fused_activation);
            return kcr_done;
        }

        kernel_call_result memory_copy(memory_copy_options &options, interpreter &interpreter, interpreter_step_t step)
        {
            auto input = interpreter.memory_at<float>(options.input);
            auto output = interpreter.memory_at<float>(options.output);

            std::copy(input.begin(), input.end(), output.begin());
            return kcr_done;
        }

        kernel_call_result pad(pad_options &options, interpreter &interpreter, interpreter_step_t step)
        {
            auto input = interpreter.memory_at<uint8_t>(options.input);
            auto output = interpreter.memory_at<uint8_t>(options.output);

#define PAD_KERNEL(T) \
    kernels::neutral::pad(reinterpret_cast<const T *>(input.data()), reinterpret_cast<T *>(output.data()), options.in_shape, options.paddings, options.pad_value.as<T>());

            ELEM_SIZE_IMPL(options.input.datatype, PAD_KERNEL);
            return kcr_done;
#undef PAD_KERNEL
        }

        kernel_call_result quantize(quantize_options &options, interpreter &interpreter, interpreter_step_t step)
        {
            auto input = interpreter.memory_at<float>(options.input);
            auto output = interpreter.memory_at<uint8_t>(options.output);

            kernels::neutral::quantize(input.data(), output.data(), input.size(), options.quant_param);
            return runtime::kcr_done;
        }

        kernel_call_result reduce(reduce_options &options, interpreter &interpreter, interpreter_step_t step)
        {
            auto input = interpreter.memory_at<float>(options.input);
            auto output = interpreter.memory_at<float>(options.output);

            auto reduce = [&](auto op) {
                kernels::neutral::reduce(input.data(), output.data(), options.init_value, options.in_shape, options.out_shape, op);
            };

            switch (options.reduce_op)
            {
            case reduce_mean:
            {
                reduce([](auto a, auto b) { return a + b; });
                auto mul = (float)input.size() / output.size();
                kernels::neutral::unary(output.data(), output.data(), output.size(), [mul](auto a) { return a * mul; });
                return kcr_done;
            }
            case reduce_min:
                reduce([](auto a, auto b) { return std::min(a, b); });
                return kcr_done;
            case reduce_max:
                reduce([](auto a, auto b) { return std::max(a, b); });
                return kcr_done;
            default:
                return kcr_error;
            }
        }

        kernel_call_result reduce_window2d(reduce_window2d_options &options, interpreter &interpreter, interpreter_step_t step)
        {
            auto input = interpreter.memory_at<float>(options.input);
            auto output = interpreter.memory_at<float>(options.output);

            auto reduce = [&](auto binary_op, auto window_op) {
                kernels::neutral::reduce_window2d(input.data(), output.data(), options.init_value, options.in_shape, options.filter_h, options.filter_w, options.stride_h,
                    options.stride_w, options.dilation_h, options.dilation_w, options.padding_h, options.padding_w, options.fused_activation, binary_op, window_op);
            };

            switch (options.reduce_op)
            {
            case reduce_mean:
                reduce([](auto a, auto b) { return a + b; }, [](auto v, auto k) { return v / k; });
                return kcr_done;
            case reduce_min:
                reduce([](auto a, auto b) { return std::min(a, b); }, [](auto v, auto k) { return v; });
                return kcr_done;
            case reduce_max:
                reduce([](auto a, auto b) { return std::max(a, b); }, [](auto v, auto k) { return v; });
                return kcr_done;
            default:
                return kcr_error;
            }
        }

        kernel_call_result resize_bilinear(resize_bilinear_options &options, interpreter &interpreter, interpreter_step_t step)
        {
            auto input = interpreter.memory_at<float>(options.input);
            auto output = interpreter.memory_at<float>(options.output);

            kernels::neutral::resize_bilinear(input.data(), output.data(), options.in_shape, options.out_h, options.out_w, options.align_corners);
            return kcr_done;
        }

        kernel_call_result resize_nearest_neighbor(resize_nearest_neighbor_options &options, runtime::interpreter &interpreter, runtime::interpreter_step_t step)
        {
            auto input = interpreter.memory_at<uint8_t>(options.input);
            auto output = interpreter.memory_at<uint8_t>(options.output);

#define RESIZE_NN_KERNEL(T) \
    kernels::neutral::resize_nearest_neighbor(reinterpret_cast<const T *>(input.data()), reinterpret_cast<T *>(output.data()), options.in_shape, options.out_h, options.out_w);

            ELEM_SIZE_IMPL(options.input.datatype, RESIZE_NN_KERNEL);
            return kcr_done;
#undef RESIZE_NN_KERNEL
        }

        kernel_call_result softmax(softmax_options &options, interpreter &interpreter, interpreter_step_t step)
        {
            auto input = interpreter.memory_at<float>(options.input);
            auto output = interpreter.memory_at<float>(options.output);

            kernels::neutral::softmax(input.data(), output.data(), options.beta, options.outer_size, options.inner_size);
            return kcr_done;
        }

        kernel_call_result transpose(transpose_options &options, interpreter &interpreter, interpreter_step_t step)
        {
            auto input = interpreter.memory_at<uint8_t>(options.input);
            auto output = interpreter.memory_at<uint8_t>(options.output);

#define TRANSPOSE_KERNEL(T) \
    kernels::neutral::transpose(reinterpret_cast<const T *>(input.data()), reinterpret_cast<T *>(output.data()), options.in_shape, options.perm);

            ELEM_SIZE_IMPL(options.input.datatype, TRANSPOSE_KERNEL);
            return kcr_done;
#undef TRANSPOSE_KERNEL
        }
    }
}
}
