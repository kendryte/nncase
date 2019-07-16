#pragma once
#include "../../kernels/cpu/cpu_kernels.h"
#include "../node_body.h"

namespace nncase
{
namespace targets
{
    namespace cpu
    {
        struct cpu_depthwise_conv2d_options
        {
            memory_range input;
            memory_range output;
            runtime_shape_t in_shape;
            padding padding_h;
            padding padding_w;
            int32_t filter_h;
            int32_t filter_w;
            int32_t stride_h;
            int32_t stride_w;
            int32_t dilation_h;
            int32_t dilation_w;
            value_range<float> fused_activation;
            xtl::span<const float> weights;
            xtl::span<const float> bias;

            void deserialize(runtime::span_reader &reader)
            {
                reader.read(input);
                reader.read(output);
                reader.read(in_shape);
                reader.read(padding_h);
                reader.read(padding_w);
                reader.read(filter_h);
                reader.read(filter_w);
                reader.read(stride_h);
                reader.read(stride_w);
                reader.read(dilation_h);
                reader.read(dilation_w);
                reader.read_span(weights, in_shape[3] * filter_h * filter_w);
                reader.read_span(bias, in_shape[3]);
            }
        };

        runtime::kernel_call_result cpu_depthwise_conv2d(cpu_depthwise_conv2d_options &options, runtime::interpreter &interpreter, runtime::interpreter_step_t step)
        {
            auto input = interpreter.memory_at<float>(options.input);
            auto output = interpreter.memory_at<float>(options.output);
            kernels::cpu::depthwise_conv2d(input.data(), output.data(), options.weights.data(), options.bias.data(), options.in_shape, options.filter_h,
                options.filter_w, options.stride_h, options.stride_w, options.dilation_h, options.dilation_w, options.padding_h, options.padding_w, options.fused_activation);
            return runtime::kcr_done;
        }
    }
}
}
