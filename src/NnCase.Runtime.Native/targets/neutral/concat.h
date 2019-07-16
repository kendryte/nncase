#pragma once
#include "../../kernels/neutral/neutral_kernels.h"
#include "../node_body.h"

namespace nncase
{
namespace targets
{
    namespace neutral
    {
        struct concat_options
        {
            memory_range output;
            uint32_t inner_size;
            uint32_t outer_size;
            uint32_t inputs_count;
            xtl::span<const memory_range> inputs;
            xtl::span<const int32_t> dims;

            void deserialize(runtime::span_reader &reader)
            {
                reader.read(output);
                reader.read(inner_size);
                reader.read(outer_size);
                reader.read(inputs_count);
                reader.read_span(inputs, inputs_count);
                reader.read_span(dims, inputs_count);
            }
        };

        runtime::kernel_call_result concat(concat_options &options, runtime::interpreter &interpreter, runtime::interpreter_step_t step)
        {
            auto output = interpreter.memory_at<uint8_t>(options.output);
            xtl::span<const uint8_t *> inputs = { (const uint8_t **)alloca(options.inputs_count * sizeof(uint8_t *)), options.inputs_count };
            for (size_t i = 0; i < options.inputs_count; i++)
                inputs[i] = interpreter.memory_at<uint8_t>(options.inputs[i]).data();
            kernels::neutral::concat(inputs, output.data(), options.dims, options.inner_size, options.outer_size);
            return runtime::kcr_done;
        }
    }
}
}
