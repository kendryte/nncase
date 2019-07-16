#pragma once
#include "../../kernels/neutral/neutral_kernels.h"
#include "../node_body.h"

namespace nncase
{
namespace targets
{
    namespace neutral
    {
        struct softmax_options : public simple_node_body<runtime::rop_softmax, softmax_options>
        {
            memory_range input;
            memory_range output;
            int32_t inner_size;
            int32_t outer_size;
            float beta;
        };

        runtime::kernel_call_result softmax(softmax_options &options, runtime::interpreter &interpreter, runtime::interpreter_step_t step)
        {
            auto input = interpreter.memory_at<float>(options.input);
            auto output = interpreter.memory_at<float>(options.output);

            kernels::neutral::softmax(input.data(), output.data(), options.beta, options.outer_size, options.inner_size);
            return runtime::kcr_done;
        }
    }
}
}
