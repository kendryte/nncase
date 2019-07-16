#pragma once
#include "../../kernels/neutral/neutral_kernels.h"
#include "../node_body.h"

namespace nncase
{
namespace targets
{
    namespace neutral
    {
        struct binary_options : public simple_node_body<runtime::rop_binary, binary_options>
        {
            memory_range input_a;
            memory_range input_b;
            memory_range output;
            binary_op binary_op;
            runtime_shape_t in_a_shape;
            runtime_shape_t in_b_shape;
            runtime_shape_t out_shape;
            value_range<float> fused_activation;
        };

        runtime::kernel_call_result binary(binary_options &options, runtime::interpreter &interpreter, runtime::interpreter_step_t step)
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
                return runtime::kcr_done;
            case binary_sub:
                binary([](auto a, auto b) { return a - b; });
                return runtime::kcr_done;
            case binary_mul:
                binary([](auto a, auto b) { return a * b; });
                return runtime::kcr_done;
            case binary_div:
                binary([](auto a, auto b) { return a / b; });
                return runtime::kcr_done;
            default:
                return runtime::kcr_error;
            }
        }
    }
}
}
