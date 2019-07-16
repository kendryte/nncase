#pragma once
#include "../../kernels/neutral/neutral_kernels.h"
#include "../node_body.h"

namespace nncase
{
namespace targets
{
    namespace neutral
    {
        struct memory_copy_options : public simple_node_body<runtime::rop_memory_copy, memory_copy_options>
        {
            memory_range input;
            memory_range output;
        };

        runtime::kernel_call_result memory_copy(memory_copy_options &options, runtime::interpreter &interpreter, runtime::interpreter_step_t step)
        {
            auto input = interpreter.memory_at<float>(options.input);
            auto output = interpreter.memory_at<float>(options.output);

            std::copy(input.begin(), input.end(), output.begin());
            return runtime::kcr_done;
        }
    }
}
}
