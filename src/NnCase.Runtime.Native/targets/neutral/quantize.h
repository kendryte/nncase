#pragma once
#include "../../kernels/neutral/neutral_kernels.h"
#include "../node_body.h"

namespace nncase
{
namespace targets
{
    namespace neutral
    {
        struct quantize_options : public simple_node_body<runtime::rop_quantize, quantize_options>
        {
            memory_range input;
            memory_range output;
            quant_param quant_param;
        };

        runtime::kernel_call_result quantize(quantize_options &options, runtime::interpreter &interpreter, runtime::interpreter_step_t step)
        {
            auto input = interpreter.memory_at<float>(options.input);
            auto output = interpreter.memory_at<uint8_t>(options.output);

            kernels::neutral::quantize(input.data(), output.data(), input.size(), options.quant_param);
            return runtime::kcr_done;
        }
    }
}
}
