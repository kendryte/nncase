/* Copyright 2019-2021 Canaan Inc.
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
#include "../tflite_importer.h"
#include <nncase/ir/ops/bitcast.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/matmul.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_TFLITE_LOWER(FULLY_CONNECTED)
{
    auto &input_a = get_tensor(op.inputs(), 0);
    auto &input_b = get_tensor(op.inputs(), 1);
    [[maybe_unused]] int not_f32 = 0;

    auto &output = get_tensor(op.outputs(), 0);
    auto &options = *op.builtin_options_as_FullyConnectedOptions();

    assert(options.weights_format() == tflite::FullyConnectedOptionsWeightsFormat_DEFAULT);

    dequantize *input_a_dequant, *input_b_dequant, *bias_dequant;
    transpose *input_b_trans;
    quantize *output_quant;

    // input_a dequantize
    if (input_a.type() != tflite::TensorType_FLOAT32)
    {
        not_f32 = 1;
        quant_param_t input_a_dequant_paras = to_quant_param(input_a.quantization());
        input_a_dequant = graph_.emplace<dequantize>(to_data_type(input_a.type()), get_shape(input_a.shape()), dt_float32,
            input_a_dequant_paras);
        input_a_dequant->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/input_a_dequant");
    }

    // input_b dequantize
    if (input_b.type() != tflite::TensorType_FLOAT32)
    {
        quant_param_t input_b_dequant_paras = to_quant_param(input_b.quantization());
        input_b_dequant = graph_.emplace<dequantize>(to_data_type(input_b.type()), get_shape(input_b.shape()), dt_float32, input_b_dequant_paras);
        input_b_dequant->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/input_b_dequant");
        input_b_trans = graph_.emplace<transpose>(input_b_dequant->output().type(), get_shape(input_b.shape()), axis_t { 1, 0 });
        input_b_trans->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/input_b_transpose_0");
        input_b_trans->input().connect(input_b_dequant->output());
        link_input_tensor(&input_b_dequant->input(), op.inputs()->Get(1));
    }
    else
    {
        input_b_trans = graph_.emplace<transpose>(to_data_type(input_b.type()), get_shape(input_b.shape()), axis_t { 1, 0 });
        input_b_trans->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/input_b_transpose_1");
        link_input_tensor(&input_b_trans->input(), op.inputs()->Get(1));
    }

    auto rshape_input_a = graph_.emplace<bitcast>(dt_float32, get_shape(input_a.shape()), dt_float32,
        axis_t { -1, (int32_t)input_b_trans->output().shape()[0] });
    auto fc = graph_.emplace<matmul>(rshape_input_a->output().shape(), input_b_trans->output().shape(),
        to_float_clamp_range(options.fused_activation_function()));

    // bias dequantize
    if ((op.inputs()->size() == 3) && (op.inputs()->Get(2) != -1))
    {
        auto &bias = get_tensor(op.inputs(), 2);
        if (bias.type() != tflite::TensorType_FLOAT32)
        {
            quant_param_t bias_dequant_paras = to_quant_param(bias.quantization());
            bias_dequant = graph_.emplace<dequantize>(to_data_type(bias.type()), get_shape(bias.shape()), dt_float32, bias_dequant_paras);
            bias_dequant->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/bias_quant");
            fc->bias().connect(bias_dequant->output());
            link_input_tensor(&bias_dequant->input(), op.inputs()->Get(2));
        }
        else
        {
            link_input_tensor(&fc->bias(), op.inputs()->Get(2));
        }
    }

    // input_a?dequant connect
    if (not_f32)
    {
        rshape_input_a->input().connect(input_a_dequant->output());
        link_input_tensor(&input_a_dequant->input(), op.inputs()->Get(0));
    }
    else
    {
        link_input_tensor(&rshape_input_a->input(), op.inputs()->Get(0));
    }

    auto rshape_output = graph_.emplace<bitcast>(dt_float32, fc->output().shape(), get_shape(output.shape()));

    rshape_input_a->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/reshape");
    fc->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/fc");
    rshape_output->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/reshape");

    fc->input_a().connect(rshape_input_a->output());
    fc->input_b().connect(input_b_trans->output());
    rshape_output->input().connect(fc->output());

    if (rshape_output->output().type() != to_data_type(output.type()))
    {
        quant_param_t output_quant_paras = to_quant_param(output.quantization());
        output_quant = graph_.emplace<quantize>(dt_float32, rshape_output->output().shape(), to_data_type(output.type()), output_quant_paras);
        output_quant->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "/output_quant");
        output_quant->input().connect(rshape_output->output());
        link_output_tensor(op.outputs()->Get(0), &output_quant->output());
    }
    else
    {
        link_output_tensor(op.outputs()->Get(0), &rshape_output->output());
    }
}

DEFINE_TFLITE_LOWER(BATCH_MATMUL)
{
    auto &input_a = get_tensor(op.inputs(), 0);
    auto &input_b = get_tensor(op.inputs(), 1);
    auto &output = get_tensor(op.outputs(), 0);

    auto input_a_type = to_data_type(input_a.type());
    auto a_type = input_a_type;
    auto input_a_shape = get_shape(input_a.shape());
    int32_t input_a_shape_size = input_a_shape.size();
    assert(input_a_shape_size >= 2);

    auto input_b_type = to_data_type(input_b.type());
    auto b_type = input_b_type;
    auto input_b_shape = get_shape(input_b.shape());
    int32_t input_b_shape_size = input_b_shape.size();
    assert(input_b_shape_size >= 2);

    auto output_type = to_data_type(output.type());
    auto output_shape = get_shape(output.shape());

    auto &options = *op.builtin_options_as_BatchMatMulOptions();
    bool adj_a = options.adj_x();
    bool adj_b = options.adj_y();

    // dequantize input_a
    dequantize *input_a_dequant = nullptr;
    if (input_a_type == dt_uint8 || input_a_type == dt_int8)
    {
        auto dequant_paras = to_quant_param(input_a.quantization());
        input_a_dequant = graph_.emplace<dequantize>(input_a_type, input_a_shape, dt_float32, dequant_paras);
        input_a_dequant->name(std::string(output.name()->string_view()) + "/input_a_dequant");
        link_input_tensor(&input_a_dequant->input(), op.inputs()->Get(0));
        a_type = dt_float32;
    }

    // transpose input_a
    transpose *input_a_tp = nullptr;
    if (adj_a)
    {
        axis_t perm(input_a_shape_size);
        std::iota(std::begin(perm), std::end(perm), 0);
        std::swap(perm[input_a_shape_size - 2], perm[input_a_shape_size - 1]);
        input_a_tp = graph_.emplace<transpose>(a_type, input_a_shape, perm);
        input_a_tp->name(std::string(output.name()->string_view()) + "/input_a_transpose");
        if (input_a_dequant)
        {
            input_a_tp->input().connect(input_a_dequant->output());
        }
        else
        {
            link_input_tensor(&input_a_tp->input(), op.inputs()->Get(0));
        }
    }

    // reshape input_a as [batch, m, k]
    shape_t shape = adj_a ? input_a_tp->output().shape() : input_a_shape;
    shape_t new_a_shape { 1, 1, 1 };
    if (input_a_shape_size == 2)
    {
        new_a_shape[1] = shape[0];
        new_a_shape[2] = shape[1];
    }
    else if (input_a_shape_size == 3)
    {
        new_a_shape.assign(shape.begin(), shape.end());
    }
    else
    {
        for (size_t i = 0; i < input_a_shape_size - 2; i++)
        {
            new_a_shape[0] *= shape[i];
        }

        new_a_shape[1] = shape[input_a_shape_size - 2];
        new_a_shape[2] = shape[input_a_shape_size - 1];
    }

    auto input_a_bc = graph_.emplace<bitcast>(a_type, shape, new_a_shape);
    input_a_bc->name(std::string(output.name()->string_view()) + "/input_a_reshape");
    if (input_a_dequant == nullptr && input_a_tp == nullptr)
    {
        link_input_tensor(&input_a_bc->input(), op.inputs()->Get(0));
    }
    else if (input_a_tp == nullptr)
    {
        input_a_bc->input().connect(input_a_dequant->output());
    }
    else
    {
        input_a_bc->input().connect(input_a_tp->output());
    }

    // dequantize input_b
    dequantize *input_b_dequant = nullptr;
    if (input_b_type == dt_uint8 || input_b_type == dt_int8)
    {
        auto dequant_paras = to_quant_param(input_b.quantization());
        input_b_dequant = graph_.emplace<dequantize>(input_b_type, input_b_shape, dt_float32, dequant_paras);
        input_b_dequant->name(std::string(output.name()->string_view()) + "/input_b_dequant");
        link_input_tensor(&input_b_dequant->input(), op.inputs()->Get(1));
        b_type = dt_float32;
    }

    // transpose input_b
    transpose *input_b_tp = nullptr;
    if (adj_b)
    {
        axis_t perm(input_b_shape_size);
        std::iota(std::begin(perm), std::end(perm), 0);
        std::swap(perm[input_b_shape_size - 2], perm[input_b_shape_size - 1]);
        input_b_tp = graph_.emplace<transpose>(b_type, input_b_shape, perm);
        input_b_tp->name(std::string(output.name()->string_view()) + "/input_b_transpose");
        if (input_b_dequant)
        {
            input_b_tp->input().connect(input_b_dequant->output());
        }
        else
        {
            link_input_tensor(&input_b_tp->input(), op.inputs()->Get(1));
        }
    }

    // reshape input_b as [batch, k, n]
    shape = adj_b ? input_b_tp->output().shape() : input_b_shape;
    shape_t new_b_shape { 1, 1, 1 };
    if (input_b_shape_size == 2)
    {
        new_b_shape[1] = shape[0];
        new_b_shape[2] = shape[1];
    }
    else if (input_b_shape_size == 3)
    {
        new_b_shape.assign(shape.begin(), shape.end());
    }
    else
    {
        for (size_t i = 0; i < input_b_shape_size - 2; i++)
        {
            new_b_shape[0] *= shape[i];
        }

        new_b_shape[1] = shape[input_b_shape_size - 2];
        new_b_shape[2] = shape[input_b_shape_size - 1];
    }

    auto input_b_bc = graph_.emplace<bitcast>(b_type, shape, new_b_shape);
    input_b_bc->name(std::string(output.name()->string_view()) + "/input_b_reshape");
    if (input_b_dequant == nullptr && input_b_tp == nullptr)
    {
        link_input_tensor(&input_b_bc->input(), op.inputs()->Get(1));
    }
    else if (input_b_tp == nullptr)
    {
        input_b_bc->input().connect(input_b_dequant->output());
    }
    else
    {
        input_b_bc->input().connect(input_b_tp->output());
    }

    // bias
    auto dim = input_b_bc->output().shape().back();
    std::vector<float> bias_value(dim, 0.f);
    shape_t bias_shape = { dim };
    auto bias = graph_.emplace<constant>(dt_float32, bias_shape, bias_value);
    bias->name(std::string(output.name()->string_view()) + "/bias");

    // matmul
    auto mm = graph_.emplace<matmul>(input_a_bc->output().shape(), input_b_bc->output().shape(), value_range<float>::full());
    mm->name(std::string(output.name()->string_view()) + "/matmul");
    mm->input_a().connect(input_a_bc->output());
    mm->input_b().connect(input_b_bc->output());
    mm->bias().connect(bias->output());

    // reshape output
    auto output_bc = graph_.emplace<bitcast>(mm->output().type(), mm->output().shape(), output_shape);
    output_bc->name(std::string(output.name()->string_view()) + "/output_reshape");
    output_bc->input().connect(mm->output());

    // quantize output
    if (output_bc->output().type() != output_type)
    {
        auto quant_paras = to_quant_param(output.quantization());
        auto output_quant = graph_.emplace<quantize>(output_bc->output().type(), output_bc->output().shape(), output_type, quant_paras);
        output_quant->name(std::string(output.name()->string_view()) + "/output_quant");
        output_quant->input().connect(output_bc->output());
        link_output_tensor(op.outputs()->Get(0), &output_quant->output());
    }
    else
    {
        link_output_tensor(op.outputs()->Get(0), &output_bc->output());
    }
}