#pragma once
#include "../../kernels/neutral/neutral_kernels.h"
#include "../node_body.h"

namespace nncase
{
namespace targets
{
    namespace neutral
    {
        struct matmul_options
        {
            memory_range input_a;
            memory_range input_b;
            memory_range output;
            int32_t a_rows;
            int32_t a_cols;
            int32_t b_cols;
            value_range<float> fused_activation;
            xtl::span<const float> bias;

            void deserialize(runtime::span_reader &reader)
            {
                reader.read(input_a);
                reader.read(input_b);
                reader.read(output);
                reader.read(a_rows);
                reader.read(a_cols);
                reader.read(b_cols);
                reader.read(fused_activation);
                reader.read_span(bias, b_cols);
            }
        };

        runtime::kernel_call_result matmul(matmul_options &options, runtime::interpreter &interpreter, runtime::interpreter_step_t step)
        {
            auto input_a = interpreter.memory_at<float>(options.input_a);
            auto input_b = interpreter.memory_at<float>(options.input_b);
            auto output = interpreter.memory_at<float>(options.output);
            kernels::neutral::matmul(input_a.data(), input_b.data(), output.data(), options.bias.data(), options.a_rows, options.a_cols, options.b_cols, options.fused_activation);
            return runtime::kcr_done;
        }
    }
}
}
