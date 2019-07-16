#pragma once
#include "../../kernels/neutral/neutral_kernels.h"
#include "../node_body.h"

namespace nncase
{
namespace targets
{
    namespace neutral
    {
        struct pad_options : public simple_node_body<runtime::rop_pad, pad_options>
        {
            memory_range input;
            memory_range output;
            runtime_shape_t in_shape;
            runtime_paddings_t paddings;
            scalar pad_value;
        };

        runtime::kernel_call_result pad(pad_options &options, runtime::interpreter &interpreter, runtime::interpreter_step_t step)
        {
            auto input = interpreter.memory_at<uint8_t>(options.input);
            auto output = interpreter.memory_at<uint8_t>(options.output);

            return kernels::neutral::pad(runtime::get_bytes(options.input.datatype), input.data(), output.data(), options.in_shape, options.paddings, options.pad_value);
        }
    }
}
}
