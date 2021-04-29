/* Copyright 2019-2020 Canaan Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <kernels/neutral/neutral_kernels.h>
#include <runtime/kernel_registry.h>
#include <runtime/neutral/neutral_ops_body.h>

using namespace nncase;
using namespace nncase::runtime;

#define ELEM_SIZE_IMPL(type, KERNEL)  \
    switch (runtime::get_bytes(type)) \
    {                                 \
    case 1:                           \
        KERNEL(uint8_t);              \
        break;                        \
    case 2:                           \
        KERNEL(uint16_t);             \
        break;                        \
    case 4:                           \
        KERNEL(uint32_t);             \
        break;                        \
    default:                          \
        return kcr_error;             \
    }

#define FP_OR_Q_IMPL(type, KERNEL) \
    switch (type)                  \
    {                              \
    case dt_float32:               \
        KERNEL(float);             \
        break;                     \
    case dt_uint8:                 \
        KERNEL(uint8_t);           \
        break;                     \
    default:                       \
        return kcr_error;          \
    }

namespace nncase
{
namespace runtime
{
    namespace neutral
    {
        kernel_call_result binary(binary_options &options, interpreter_t &interpreter, interpreter_step_t step)
        {
            auto input_a = interpreter.memory_at<float>(options.input_a);
            auto input_b = interpreter.memory_at<float>(options.input_b);
            auto output = interpreter.memory_at<float>(options.output);

            auto binary = [&](auto op) {
                kernels::neutral::binary(input_a.data(), input_b.data(), output.data(), options.in_a_shape, options.in_b_shape, options.out_shape, options.fused_activation, op);
            };

            switch (options.binary_op)
            {
            case binary_add:
                binary([](auto a, auto b) { return a + b; });
                return kcr_done;
            case binary_sub:
                binary([](auto a, auto b) { return a - b; });
                return kcr_done;
            case binary_mul:
                binary([](auto a, auto b) { return a * b; });
                return kcr_done;
            case binary_div:
                binary([](auto a, auto b) { return a / b; });
                return kcr_done;
            case binary_min:
                binary([](auto a, auto b) { return std::min(a, b); });
                return kcr_done;
            case binary_max:
                binary([](auto a, auto b) { return std::max(a, b); });
                return kcr_done;
            default:
                return kcr_error;
            }
        }

        kernel_call_result quantized_binary(quantized_binary_options &options, interpreter_t &interpreter, interpreter_step_t step)
        {
            auto input_a = interpreter.memory_at<uint8_t>(options.input_a);
            auto input_b = interpreter.memory_at<uint8_t>(options.input_b);
            auto output = interpreter.memory_at<uint8_t>(options.output);

            auto binary = [&](auto op) {
                kernels::neutral::quantized_binary(input_a.data(), input_b.data(), output.data(), options.in_a_shape, options.in_b_shape, options.out_shape,
                    options.input_a_offset, options.input_a_mul, options.input_a_shift, options.input_b_offset, options.input_b_mul, options.input_b_shift,
                    options.output_mul, options.output_shift, options.output_offset, op);
            };

            std::cout << "Binary op: " << options.binary_op << std::endl;
            switch (options.binary_op)
            {
            case binary_add:
                binary([](auto a, auto b) { return a + b; });
                return kcr_done;
            case binary_sub:
                binary([](auto a, auto b) { return a - b; });
                return kcr_done;
            case binary_mul:
                binary([](auto a, auto b) { return a * b; });
                return kcr_done;
            case binary_div:
                binary([](auto a, auto b) { return (a + b / 2) / b; });
                return kcr_done;
            case binary_min:
                binary([](auto a, auto b) { return std::min(a, b); });
                return kcr_done;
            case binary_max:
                binary([](auto a, auto b) { return std::max(a, b); });
                return kcr_done;
            default:
                return kcr_error;
            }
        }

        kernel_call_result concat(concat_options &options, interpreter_t &interpreter, interpreter_step_t step)
        {
            auto output = interpreter.memory_at<uint8_t>(options.output);
            kernels::neutral::concat(options.inputs, output.data(), options.dims, options.inner_size, options.outer_size,
                [&](const memory_range &range) { return interpreter.memory_at<uint8_t>(range).data(); });
            return kcr_done;
        }

        kernel_call_result conv2d(conv2d_options &options, interpreter_t &interpreter, interpreter_step_t step)
        {
            auto input = interpreter.memory_at<float>(options.input);
            auto output = interpreter.memory_at<float>(options.output);
            kernels::neutral::conv2d(input.data(), output.data(), options.weights.data(), options.bias.data(), options.in_shape, options.groups, options.out_channels, options.filter_h,
                options.filter_w, options.stride_h, options.stride_w, options.dilation_h, options.dilation_w, options.padding_h, options.padding_w, options.fused_activation);
            return kcr_done;
        }

        kernel_call_result quantized_conv2d(quantized_conv2d_options &options, interpreter_t &interpreter, interpreter_step_t step)
        {
            auto input = interpreter.memory_at<uint8_t>(options.input);
            auto output = interpreter.memory_at<uint8_t>(options.output);
            kernels::neutral::quantized_conv2d(input.data(), output.data(), options.weights.data(), options.bias.data(), options.input_offset, options.filter_offset,
                options.output_mul, options.output_shift, options.output_offset, options.in_shape, options.groups, options.out_channels, options.filter_h,
                options.filter_w, options.stride_h, options.stride_w, options.dilation_h, options.dilation_w, options.padding_h, options.padding_w);
            return kcr_done;
        }

        kernel_call_result conv2d_transpose(conv2d_transpose_options &options, interpreter_t &interpreter, interpreter_step_t step)
        {
            auto input = interpreter.memory_at<float>(options.input);
            auto output = interpreter.memory_at<float>(options.output);
            kernels::neutral::conv2d_transpose(input.data(), output.data(), options.weights.data(), options.bias.data(), options.in_shape, options.groups, options.out_shape, options.filter_h,
                options.filter_w, options.stride_h, options.stride_w, options.dilation_h, options.dilation_w, options.padding_h, options.padding_w, options.fused_activation);
            return kcr_done;
        }

        kernel_call_result dequantize(dequantize_options &options, interpreter_t &interpreter, interpreter_step_t step)
        {
            auto input = interpreter.memory_at<uint8_t>(options.input);
            auto output = interpreter.memory_at<float>(options.output);

            kernels::neutral::dequantize(input.data(), output.data(), input.size(), options.quant_param);
            return kcr_done;
        }

        kernel_call_result matmul(matmul_options &options, interpreter_t &interpreter, interpreter_step_t step)
        {
            auto input_a = interpreter.memory_at<float>(options.input_a);
            auto input_b = interpreter.memory_at<float>(options.input_b);
            auto output = interpreter.memory_at<float>(options.output);
            kernels::neutral::matmul(input_a.data(), input_b.data(), output.data(), options.bias.data(), options.a_rows, options.a_cols, options.b_cols, options.fused_activation);
            return kcr_done;
        }

        kernel_call_result quantized_matmul(quantized_matmul_options &options, interpreter_t &interpreter, interpreter_step_t step)
        {
            auto input_a = interpreter.memory_at<uint8_t>(options.input_a);
            auto input_b = interpreter.memory_at<uint8_t>(options.input_b);
            auto output = interpreter.memory_at<uint8_t>(options.output);
            kernels::neutral::quantized_matmul(input_a.data(), input_b.data(), output.data(), options.bias.data(), options.a_rows, options.a_cols, options.b_cols,
                options.input_a_offset, options.input_b_offset, options.output_mul, options.output_shift, options.output_offset);
            return kcr_done;
        }

        kernel_call_result memory_copy(memory_copy_options &options, interpreter_t &interpreter, interpreter_step_t step)
        {
            auto input = interpreter.memory_at<float>(options.input);
            auto output = interpreter.memory_at<float>(options.output);

            std::copy(input.begin(), input.end(), output.begin());
            return kcr_done;
        }

        kernel_call_result pad(pad_options &options, interpreter_t &interpreter, interpreter_step_t step)
        {
            auto input = interpreter.memory_at<uint8_t>(options.input);
            auto output = interpreter.memory_at<uint8_t>(options.output);

#define PAD_KERNEL(T) \
    kernels::neutral::pad(reinterpret_cast<const T *>(input.data()), reinterpret_cast<T *>(output.data()), options.in_shape, options.paddings, options.pad_value.as<T>());

            ELEM_SIZE_IMPL(options.input.datatype, PAD_KERNEL);
            return kcr_done;
#undef PAD_KERNEL
        }

        kernel_call_result quantize(quantize_options &options, interpreter_t &interpreter, interpreter_step_t step)
        {
            auto input = interpreter.memory_at<float>(options.input);
            auto output = interpreter.memory_at<uint8_t>(options.output);

            kernels::neutral::quantize(input.data(), output.data(), input.size(), options.quant_param);
            return runtime::kcr_done;
        }

        kernel_call_result reduce(reduce_options &options, interpreter_t &interpreter, interpreter_step_t step)
        {
            auto input = interpreter.memory_at<float>(options.input);
            auto output = interpreter.memory_at<float>(options.output);

            auto reduce = [&](auto op) {
                kernels::neutral::reduce(input.data(), output.data(), options.init_value, options.in_shape, options.out_shape, op);
            };

            switch (options.reduce_op)
            {
            case reduce_mean:
            {
                reduce([](auto a, auto b) { return a + b; });
                auto mul = (float)output.size() / input.size();
                kernels::neutral::unary(output.data(), output.data(), output.size(), [mul](auto a) { return a * mul; });
                return kcr_done;
            }
            case reduce_min:
                reduce([](auto a, auto b) { return std::min(a, b); });
                return kcr_done;
            case reduce_max:
                reduce([](auto a, auto b) { return std::max(a, b); });
                return kcr_done;
            case reduce_sum:
                reduce([](auto a, auto b) { return a + b; });
                return kcr_done;
            default:
                return kcr_error;
            }
        }

        kernel_call_result reduce_window2d(reduce_window2d_options &options, interpreter_t &interpreter, interpreter_step_t step)
        {
            auto input = interpreter.memory_at<float>(options.input);
            auto output = interpreter.memory_at<float>(options.output);

            auto reduce = [&](auto binary_op, auto window_op) {
                kernels::neutral::reduce_window2d(input.data(), output.data(), options.init_value, options.in_shape, options.filter_h, options.filter_w, options.stride_h,
                    options.stride_w, options.dilation_h, options.dilation_w, options.padding_h, options.padding_w, options.fused_activation, binary_op, window_op);
            };

            switch (options.reduce_op)
            {
            case reduce_mean:
                reduce([](auto a, auto b) { return a + b; }, [](auto v, auto k) { return v / k; });
                return kcr_done;
            case reduce_min:
                reduce([](auto a, auto b) { return std::min(a, b); }, [](auto v, auto k) { return v; });
                return kcr_done;
            case reduce_max:
                reduce([](auto a, auto b) { return std::max(a, b); }, [](auto v, auto k) { return v; });
                return kcr_done;
            case reduce_sum:
                reduce([](auto a, auto b) { return a + b; }, [](auto v, auto k) { return v; });
                return kcr_done;
            default:
                return kcr_error;
            }
        }

        kernel_call_result resize_image(resize_image_options &options, interpreter_t &interpreter, interpreter_step_t step)
        {
            auto input = interpreter.memory_at<uint8_t>(options.input);
            auto output = interpreter.memory_at<uint8_t>(options.output);

            if (options.mode == image_resize_bilinear)
            {
#define RESIZE_BL_KERNEL(T) \
    kernels::neutral::resize_bilinear(reinterpret_cast<const T *>(input.data()), reinterpret_cast<T *>(output.data()), options.in_shape, options.out_h, options.out_w, options.align_corners);

                FP_OR_Q_IMPL(options.input.datatype, RESIZE_BL_KERNEL);
                return kcr_done;
#undef RESIZE_BL_KERNEL
            }
            else
            {
#define RESIZE_NN_KERNEL(T) \
    kernels::neutral::resize_nearest_neighbor(reinterpret_cast<const T *>(input.data()), reinterpret_cast<T *>(output.data()), options.in_shape, options.out_h, options.out_w);

                FP_OR_Q_IMPL(options.input.datatype, RESIZE_NN_KERNEL);
                return kcr_done;
#undef RESIZE_NN_KERNEL
            }
        }

        kernel_call_result softmax(softmax_options &options, interpreter_t &interpreter, interpreter_step_t step)
        {
            auto input = interpreter.memory_at<float>(options.input);
            auto output = interpreter.memory_at<float>(options.output);

            kernels::neutral::softmax(input.data(), output.data(), options.beta, options.outer_size, options.inner_size);
            return kcr_done;
        }

        kernel_call_result transpose(transpose_options &options, interpreter_t &interpreter, interpreter_step_t step)
        {
            auto input = interpreter.memory_at<uint8_t>(options.input);
            auto output = interpreter.memory_at<uint8_t>(options.output);

#define TRANSPOSE_KERNEL(T) \
    kernels::neutral::transpose(reinterpret_cast<const T *>(input.data()), reinterpret_cast<T *>(output.data()), options.in_shape, options.perm);

            ELEM_SIZE_IMPL(options.input.datatype, TRANSPOSE_KERNEL);
            return kcr_done;
#undef TRANSPOSE_KERNEL
        }

        kernel_call_result strided_slice(strided_slice_options &options, interpreter_t &interpreter, interpreter_step_t step)
        {
            auto input = interpreter.memory_at<uint8_t>(options.input);
            auto output = interpreter.memory_at<uint8_t>(options.output);

#define STRIDED_SLICE_KERNEL(T) \
    kernels::neutral::strided_slice(reinterpret_cast<const T *>(input.data()), reinterpret_cast<T *>(output.data()), options.in_shape, options.begin, options.end, options.strides);

            ELEM_SIZE_IMPL(options.input.datatype, STRIDED_SLICE_KERNEL);
            return kcr_done;
#undef STRIDED_SLICE_KERNEL
        }

        kernel_call_result unary(unary_options &options, interpreter_t &interpreter, interpreter_step_t step)
        {
            auto input = interpreter.memory_at<float>(options.input);
            auto output = interpreter.memory_at<float>(options.output);

            auto unary = [&](auto unary_op) {
                kernels::neutral::unary(input.data(), output.data(), input.size(), unary_op);
            };

            switch (options.unary_op)
            {
            case unary_abs:
                unary([](auto a) { return fabs(a); });
                return kcr_done;
            case unary_ceil:
                unary([](auto a) { return ceilf(a); });
                return kcr_done;
            case unary_cos:
                unary([](auto a) { return cosf(a); });
                return kcr_done;
            case unary_exp:
                unary([](auto a) { return expf(a); });
                return kcr_done;
            case unary_floor:
                unary([](auto a) { return floorf(a); });
                return kcr_done;
            case unary_log:
                unary([](auto a) { return logf(a); });
                return kcr_done;
            case unary_neg:
                unary([](auto a) { return -a; });
                return kcr_done;
            case unary_rsqrt:
                unary([](auto a) { return 1.f / sqrtf(a); });
                return kcr_done;
            case unary_sin:
                unary([](auto a) { return sinf(a); });
                return kcr_done;
            case unary_square:
                unary([](auto a) { return a * a; });
                return kcr_done;
            default:
                return kcr_error;
            }
        }

        kernel_call_result nnil_unary_method(nnil_unary_method_options &options, interpreter_t &interpreter, interpreter_step_t step)
        {
            auto input = interpreter.memory_at<float>(options.input);
            auto output = interpreter.memory_at<float>(options.output);

            kernels::neutral::nnil_unary_method(input.data(), output.data(), input.size(), options.body);
            return kcr_done;
        }

        kernel_call_result table_lookup1d(table_lookup1d_options &options, interpreter_t &interpreter, interpreter_step_t step)
        {
            if (options.input.datatype != dt_uint8)
                return kcr_error;

            auto input = interpreter.memory_at<uint8_t>(options.input);
            auto table = interpreter.memory_at<uint8_t>(options.table);
            auto output = interpreter.memory_at<uint8_t>(options.output);

            kernels::neutral::table_lookup1d(input.data(), output.data(), input.size(), table.data());
            return kcr_done;
        }

        kernel_call_result split(split_options &options, interpreter_t &interpreter, interpreter_step_t step) {
            auto splits = std::vector<int64_t>();
            for(auto split : options.splits) {
                splits.push_back(split);
            }

            if(options.input.datatype == dt_uint8) {
                auto input = interpreter.memory_at<uint8_t>(options.input).data();
                auto outputs = std::vector<uint8_t *>();
                for(auto out : options.outputs) {
                    if(out.size == 0) {
                        outputs.push_back(nullptr);
                        continue;
                    }

                    auto span = interpreter.memory_at<uint8_t>(out);
                    outputs.push_back(span.data());
                }
                kernels::neutral::split(input, outputs, options.input_shape, options.axis, splits);
            } else {
                auto input = interpreter.memory_at<float>(options.input).data();
                auto outputs = std::vector<float *>();
                for(auto out : options.outputs) {
                    if(out.size == 0) {
                        outputs.push_back(nullptr);
                        continue;
                    }

                    auto span = interpreter.memory_at<float>(out);
                    outputs.push_back(span.data());
                }
                kernels::neutral::split(input, outputs, options.input_shape, options.axis, splits);
            }
            return kcr_done;
        }

        kernel_call_result upsample(upsample_options &options, interpreter_t &interpreter, interpreter_step_t step) {
            auto scales = std::vector<float>();
            for(auto scale : options.scales) {
                scales.push_back(std::floor(scale));
            }
            if (options.input.datatype == dt_uint8) {
                auto input = interpreter.memory_at<uint8_t>(options.input);
                auto output = interpreter.memory_at<uint8_t>(options.output);
                kernels::neutral::upsample(input.data(), output.data(), options.input_shape, scales);
            } else {
                auto input = interpreter.memory_at<float>(options.input);
                auto output = interpreter.memory_at<float>(options.output);
                kernels::neutral::upsample(input.data(), output.data(), options.input_shape, scales);
            }
            return kcr_done;
        }
    }
}
}
