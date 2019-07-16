#pragma once
#include "../../kernels/neutral/neutral_kernels.h"
#include "../node_body.h"

namespace nncase
{
namespace targets
{
    namespace neutral
    {
        struct reduce_options : public simple_node_body<runtime::rop_reduce, reduce_options>
        {
            memory_range input;
            memory_range output;
            reduce_op reduce_op;
            runtime_shape_t in_shape;
            runtime_shape_t out_shape;
            float init_value;
        };

        runtime::kernel_call_result reduce(reduce_options &options, runtime::interpreter &interpreter, runtime::interpreter_step_t step)
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
                return runtime::kcr_done;
            }
            case reduce_min:
                reduce([](auto a, auto b) { return std::min(a, b); });
                return runtime::kcr_done;
            case reduce_max:
                reduce([](auto a, auto b) { return std::max(a, b); });
                return runtime::kcr_done;
            default:
                return runtime::kcr_error;
            }
        }
    }
}
}
