#include <kernels/cpu/cpu_kernels.h>
#include <runtime/kernel_registry.h>
#include <targets/cpu/cpu_ops_body.h>

using namespace nncase;
using namespace nncase::runtime;

namespace nncase
{
namespace targets
{
    namespace cpu
    {
        kernel_call_result cpu_conv2d(cpu_conv2d_options &options, interpreter &interpreter, interpreter_step_t step)
        {
            auto input = interpreter.memory_at<float>(options.input);
            auto output = interpreter.memory_at<float>(options.output);
            kernels::cpu::conv2d(input.data(), output.data(), options.weights.data(), options.bias.data(), options.in_shape, options.out_channels, options.filter_h,
                options.filter_w, options.stride_h, options.stride_w, options.dilation_h, options.dilation_w, options.padding_h, options.padding_w, options.fused_activation);
            return kcr_done;
        }

        kernel_call_result cpu_depthwise_conv2d(cpu_depthwise_conv2d_options &options, interpreter &interpreter, interpreter_step_t step)
        {
            auto input = interpreter.memory_at<float>(options.input);
            auto output = interpreter.memory_at<float>(options.output);
            kernels::cpu::depthwise_conv2d(input.data(), output.data(), options.weights.data(), options.bias.data(), options.in_shape, options.filter_h,
                options.filter_w, options.stride_h, options.stride_w, options.dilation_h, options.dilation_w, options.padding_h, options.padding_w, options.fused_activation);
            return kcr_done;
        }

        runtime::kernel_call_result cpu_reduce_window2d(cpu_reduce_window2d_options &options, runtime::interpreter &interpreter, runtime::interpreter_step_t step)
        {
            auto input = interpreter.memory_at<float>(options.input);
            auto output = interpreter.memory_at<float>(options.output);

            auto reduce = [&](auto binary_op, auto window_op) {
                kernels::cpu::reduce_window2d(input.data(), output.data(), options.init_value, options.in_shape, options.filter_h, options.filter_w, options.stride_h,
                    options.stride_w, options.dilation_h, options.dilation_w, options.padding_h, options.padding_w, options.fused_activation, binary_op, window_op);
            };

            switch (options.reduce_op)
            {
            case reduce_mean:
                reduce([](auto a, auto b) { return a + b; }, [](auto v, auto k) { return v / k; });
                return runtime::kcr_done;
            case reduce_min:
                reduce([](auto a, auto b) { return std::min(a, b); }, [](auto v, auto k) { return v; });
                return runtime::kcr_done;
            case reduce_max:
                reduce([](auto a, auto b) { return std::max(a, b); }, [](auto v, auto k) { return v; });
                return kcr_done;
            default:
                return kcr_error;
            }
        }

        kernel_call_result cpu_quantized_conv2d(cpu_quantized_conv2d_options &options, interpreter &interpreter, interpreter_step_t step)
        {
            auto input = interpreter.memory_at<uint8_t>(options.input);
            auto output = interpreter.memory_at<uint8_t>(options.output);
            kernels::cpu::quantized_conv2d(input.data(), output.data(), options.weights.data(), options.bias.data(), options.in_shape, options.out_channels, options.filter_h,
                options.filter_w, options.stride_h, options.stride_w, options.dilation_h, options.dilation_w, options.padding_h, options.padding_w,
                options.input_offset, options.filter_offset, options.output_mul, options.output_shift, options.output_offset);
            return kcr_done;
        }

        kernel_call_result cpu_quantized_depthwise_conv2d(cpu_quantized_depthwise_conv2d_options &options, interpreter &interpreter, interpreter_step_t step)
        {
            auto input = interpreter.memory_at<uint8_t>(options.input);
            auto output = interpreter.memory_at<uint8_t>(options.output);
            kernels::cpu::quantized_depthwise_conv2d(input.data(), output.data(), options.weights.data(), options.bias.data(), options.in_shape, options.filter_h,
                options.filter_w, options.stride_h, options.stride_w, options.dilation_h, options.dilation_w, options.padding_h, options.padding_w,
                options.input_offset, options.filter_offset, options.output_mul, options.output_shift, options.output_offset);
            return kcr_done;
        }
    }
}
}
