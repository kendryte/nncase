#pragma once
#include "../../kernels/neutral/neutral_kernels.h"
#include "../node_body.h"

namespace nncase
{
namespace targets
{
    namespace neutral
    {
        struct resize_bilinear_options : public simple_node_body<runtime::rop_resize_bilinear, resize_bilinear_options>
        {
            memory_range input;
            memory_range output;
            runtime_shape_t in_shape;
            int32_t out_h;
            int32_t out_w;
            bool align_corners;
        };

        runtime::kernel_call_result resize_bilinear(resize_bilinear_options &options, runtime::interpreter &interpreter, runtime::interpreter_step_t step)
        {
            auto input = interpreter.memory_at<float>(options.input);
            auto output = interpreter.memory_at<float>(options.output);
 
            kernels::neutral::resize_bilinear(input.data(), output.data(), options.in_shape, options.out_h, options.out_w, options.align_corners);
            return runtime::kcr_done;
        }
    }
}
}
