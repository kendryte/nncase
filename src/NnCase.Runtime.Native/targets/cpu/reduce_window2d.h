#pragma once
#include "../../kernels/neutral/neutral_kernels.h"
#include "../node_body.h"

namespace nncase
{
namespace targets
{
    namespace cpu
    {
        struct cpu_reduce_window2d_options : simple_node_body<runtime::rop_cpu_reduce_window2d, cpu_reduce_window2d_options>
        {
            memory_range input;
            memory_range output;
            reduce_op reduce_op;
            runtime_shape_t in_shape;
            padding padding_h;
            padding padding_w;
            int32_t filter_h;
            int32_t filter_w;
            int32_t stride_h;
            int32_t stride_w;
            int32_t dilation_h;
            int32_t dilation_w;
            float init_value;
            value_range<float> fused_activation;
        };

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
                return runtime::kcr_done;
            default:
                return runtime::kcr_error;
            }
        }
    }
}
}
