#pragma once
#include "../../kernels/neutral/neutral_kernels.h"
#include "../node_body.h"

namespace nncase
{
namespace targets
{
    namespace neutral
    {
        struct resize_nearest_neighbor_options : public simple_node_body<runtime::rop_resize_nearest_neighbor, resize_nearest_neighbor_options>
        {
            memory_range input;
            memory_range output;
            runtime_shape_t in_shape;
            int32_t out_h;
            int32_t out_w;
            bool align_corners;
        };

        runtime::kernel_call_result resize_nearest_neighbor(resize_nearest_neighbor_options &options, runtime::interpreter &interpreter, runtime::interpreter_step_t step)
        {
            auto input = interpreter.memory_at<uint8_t>(options.input);
            auto output = interpreter.memory_at<uint8_t>(options.output);

            return kernels::neutral::resize_nearest_neighbor(runtime::get_bytes(options.input.datatype), input.data(), output.data(), options.in_shape,
                options.out_h, options.out_w);
        }
    }
}
}
