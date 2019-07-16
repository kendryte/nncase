#pragma once
#include "../../kernels/neutral/neutral_kernels.h"
#include "../node_body.h"

namespace nncase
{
namespace targets
{
    namespace neutral
    {
        struct transpose_options : public simple_node_body<runtime::rop_transpose, transpose_options>
        {
            memory_range input;
            memory_range output;
            runtime_shape_t in_shape;
            runtime_shape_t perm;
        };

        runtime::kernel_call_result transpose(transpose_options &options, runtime::interpreter &interpreter, runtime::interpreter_step_t step)
        {
            auto input = interpreter.memory_at<uint8_t>(options.input);
            auto output = interpreter.memory_at<uint8_t>(options.output);

            return kernels::neutral::transpose(runtime::get_bytes(options.input.datatype), input.data(), output.data(), options.in_shape, options.perm);
        }
    }
}
}
