#pragma once
#include "../../kernels/neutral/neutral_kernels.h"
#include "../node_body.h"

namespace nncase
{
namespace targets
{
    namespace neutral
    {
        struct dequantize_options : public simple_node_body<runtime::rop_dequantize, dequantize_options>
        {
            memory_range input;
            memory_range output;
            quant_param quant_param;
        };

        runtime::kernel_call_result dequantize(dequantize_options &options, runtime::interpreter &interpreter, runtime::interpreter_step_t step)
        {
            auto input = interpreter.memory_at<uint8_t>(options.input);
            auto output = interpreter.memory_at<float>(options.output);

            kernels::neutral::dequantize(input.data(), output.data(), input.size(), options.quant_param);
            return runtime::kcr_done;
        }
    }
}
}
