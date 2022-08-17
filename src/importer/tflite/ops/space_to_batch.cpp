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
#include <iostream>
#include <nncase/ir/ops/batch_to_space.h>
#include <nncase/ir/ops/bitcast.h>
#include <nncase/ir/ops/pad.h>
#include <nncase/ir/ops/slice.h>
#include <nncase/ir/ops/space_to_batch.h>
#include <nncase/ir/ops/transpose.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_TFLITE_LOWER(SPACE_TO_BATCH_ND)
{
    auto &input = get_tensor(op.inputs(), 0);
    auto &output = get_tensor(op.outputs(), 0);

    shape_t new_shape;
    if (input.shape()->size() > 4)
    {
        throw std::runtime_error("Only support [3,4]dims space_to_batch_nd");
    }

    auto block_shape = load_axis<int32_t>(get_tensor(op.inputs(), 1));
    auto paddings = load_tensor<int32_t, 2>(get_tensor(op.inputs(), 2));
    auto in_shape = get_shape(input.shape());
    auto out_shape = get_shape(output.shape());
    auto spatial_size = block_shape.size();
    auto remaining_shape_size = in_shape.size() - spatial_size - 1;

    xt::svector<padding> new_paddings;
    // batch
    new_paddings.push_back(padding::zero());
    // spatial
    for (size_t i = 0; i < spatial_size; i++)
        new_paddings.push_back(padding { paddings(i, 0), paddings(i, 1) });
    if (spatial_size == 1)
        new_paddings.push_back(padding::zero());
    // remaining
    for (size_t i = 0; i < remaining_shape_size; i++)
        new_paddings.push_back(padding::zero());

    auto block_size_h = block_shape.data()[0];
    int32_t block_size_w = 1;

    // set real block_size for import, will be fixed in stage 2
    auto real_block_size_h = block_size_h;
    auto real_block_size_w = block_size_w;
    input_connector *input_conn;
    output_connector *output_conn;
    transpose *tp1, *tp2;
    if (input.shape()->size() == 3)
    {
        // if this condition is true, the shape of tflite is error
        if (out_shape[0] / in_shape[0] != block_size_h * block_size_w)
        {
            real_block_size_h = 1;
        }
        auto in_bitc = graph_.emplace<bitcast>(to_data_type(input.type()), get_shape(input.shape()), shape_t { get_shape(input.shape())[0], 1, get_shape(input.shape())[1], get_shape(input.shape())[2] });
        in_bitc->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "_s2b_in_bitc");
        tp1 = graph_.emplace<transpose>(in_bitc->output().type(), in_bitc->output().shape(), axis_t { 0, 3, 1, 2 });
        tp1->input().connect(in_bitc->output());
        input_conn = &in_bitc->input();

        // We set 'h' as expand axis if the input dim is '3', because the args used in s2b of '3D model' is single .
        // ! So, swap them for 'h,w'
        std::swap(block_size_h, block_size_w);
        std::swap(real_block_size_h, real_block_size_w);
        new_paddings.insert(new_paddings.begin() + 1, padding::zero());
    }
    else
    {
        block_size_w = block_shape.data()[1];
        real_block_size_w = block_shape.data()[1];
        tp1 = graph_.emplace<transpose>(to_data_type(input.type()), get_shape(input.shape()), axis_t { 0, 3, 1, 2 });
        input_conn = &tp1->input();
    }
    tp1->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "_s2b_in_tp");

    auto pad_value = input.type() != tflite::TensorType_FLOAT32 ? static_cast<int8_t>(input.quantization()->zero_point()->data()[0]) : 0.f;
    auto s2b = graph_.emplace<space_to_batch>(tp1->output().type(), tp1->output().shape(), block_size_h, block_size_w, new_paddings[1], new_paddings[2], pad_value, real_block_size_h, real_block_size_w);
    tp2 = graph_.emplace<transpose>(s2b->output().type(), s2b->output().shape(), axis_t { 0, 2, 3, 1 });
    s2b->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()));
    tp2->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "_s2b_out_tp");
    s2b->input().connect(tp1->output());
    tp2->input().connect(s2b->output());

    output_conn = &tp2->output();

    if (input.shape()->size() == 3)
    {
        auto out_bitc = graph_.emplace<bitcast>(tp2->output().type(), tp2->output().shape(), shape_t { tp2->output().shape()[0], tp2->output().shape()[2], tp2->output().shape()[3] });
        out_bitc->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "_s2b_outbitc");
        out_bitc->input().connect(tp2->output());
        output_conn = &out_bitc->output();
    }
    link_input_tensor(input_conn, op.inputs()->Get(0));
    link_output_tensor(op.outputs()->Get(0), output_conn);
}

DEFINE_TFLITE_LOWER(BATCH_TO_SPACE_ND)
{
    auto &input = get_tensor(op.inputs(), 0);
    auto &output = get_tensor(op.outputs(), 0);
    auto block_shape = load_axis<int32_t>(get_tensor(op.inputs(), 1));
    auto crops = load_tensor<int32_t, 2>(get_tensor(op.inputs(), 2));
    auto in_shape = get_shape(input.shape());
    auto out_shape = get_shape(output.shape());

    if (input.shape()->size() > 4)
    {
        throw std::runtime_error("Only support [3,4]dims space_to_batch_nd");
    }
    auto block_size_h = block_shape.data()[0];
    auto block_size_w = 1;

    std::vector<size_t> shape_expend;
    size_t block_shape_produt = std::accumulate(block_shape.begin(), block_shape.end(), 1, std::multiplies<size_t>());

    for (size_t i = 0; i < block_shape.size(); i++)
    {
        shape_expend.push_back(block_shape[i]);
    }
    shape_expend.push_back(in_shape[0] / block_shape_produt);
    for (size_t i = 1; i < in_shape.size(); i++)
    {
        shape_expend.push_back(in_shape[i]);
    }

    std::vector<int32_t> permute;
    permute.push_back(block_shape.size());
    for (size_t i = 0; i < block_shape.size(); i++)
    {
        permute.push_back(block_shape.size() + 1 + i); // input_shape[i+1]
        permute.push_back(i); // block_shape[i]
    }
    for (size_t i = block_shape.size() * 2 + 1; i < shape_expend.size(); i++)
    {
        permute.push_back(i);
    }
    // shape_shrink
    std::vector<int32_t> shape_shrink;
    shape_shrink.push_back(shape_expend[block_shape.size()]);
    for (size_t i = 0; i < block_shape.size(); i++)
    {
        shape_shrink.push_back(block_shape[i] * in_shape[i + 1]);
    }
    for (size_t i = block_shape.size() + 1; i < in_shape.size(); i++)
    {
        shape_shrink.push_back(in_shape[i]);
    }

    std::vector<int32_t> crop_begs, crop_ends;
    crop_begs.push_back(0);
    crop_ends.push_back(shape_shrink[0]);
    for (size_t i = 0; i < crops.shape()[0]; i++)
    {
        crop_begs.push_back(crops(i, 0));
        crop_ends.push_back(shape_shrink[i + 1] - crops(i, 1));
    }
    for (size_t i = block_shape.size() + 1; i < in_shape.size(); i++)
    {
        crop_begs.push_back(0);
        crop_ends.push_back(shape_shrink[i]);
    }

    //    crops.data()
    std::array<int32_t, 2> crop_h { crops.data()[0], crops.data()[1] };
    std::array<int32_t, 2> crop_w { crops.data()[2], crops.data()[3] };

    // set real block_size for import, will be fixed in stage 2
    NNCASE_UNUSED auto real_block_size_h = block_size_h;
    NNCASE_UNUSED auto real_block_size_w = block_size_w;

    input_connector *input_conn;
    output_connector *output_conn;
    transpose *tp1, *tp2;

    if (out_shape.size() == 3)
    {
        // if this condition is true, the shape of tflite is error
        if (in_shape[0] / out_shape[0] != block_size_h * block_size_w)
        {
            real_block_size_h = 1;
        }
        auto in_bitc = graph_.emplace<bitcast>(to_data_type(input.type()), get_shape(input.shape()), shape_t { get_shape(input.shape())[0], 1, get_shape(input.shape())[1], get_shape(input.shape())[2] });
        tp1 = graph_.emplace<transpose>(in_bitc->output().type(), in_bitc->output().shape(), axis_t { 0, 3, 1, 2 });
        in_bitc->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "_b2s_in_bitc");

        tp1->input().connect(in_bitc->output());
        input_conn = &in_bitc->input();

        // We set 'h' as expand axis if the input dim is '3', because the args used in b2s of '3D model' is single .
        // ! So, swap them for 'h,w'
        std::swap(block_size_h, block_size_w);
        std::swap(real_block_size_h, real_block_size_w);
        crop_begs.insert(crop_begs.begin() + 1, 0);
        crop_ends.insert(crop_ends.begin() + 1, 1);
        crop_w = { 0, 0 };
        std::swap(crop_h, crop_w);
    }
    else
    {
        block_size_w = block_shape.data()[1];
        real_block_size_w = block_shape.data()[1];
        crop_w = { crops.data()[2], crops.data()[3] };
        tp1 = graph_.emplace<transpose>(to_data_type(input.type()), get_shape(input.shape()), axis_t { 0, 3, 1, 2 });
        input_conn = &tp1->input();
    }
    tp1->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "_b2s_in_tp");

    auto b2s = graph_.emplace<batch_to_space>(tp1->output().type(), tp1->output().shape(), block_size_h, block_size_w, axis_t(crop_begs.size(), 1), crop_begs, crop_ends, crop_h, crop_w, real_block_size_h, real_block_size_w);
    b2s->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()));

    tp2 = graph_.emplace<transpose>(b2s->output().type(), b2s->output().shape(), axis_t { 0, 2, 3, 1 });
    tp2->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "_b2s_out_tp");
    b2s->input().connect(tp1->output());
    tp2->input().connect(b2s->output());
    output_conn = &tp2->output();

    if (input.shape()->size() == 3)
    {
        auto out_bitc = graph_.emplace<bitcast>(tp2->output().type(), tp2->output().shape(), shape_t { tp2->output().shape()[0], tp2->output().shape()[2], tp2->output().shape()[3] });
        out_bitc->input().connect(tp2->output());
        out_bitc->name(std::string(get_tensor(op.outputs(), 0).name()->string_view()) + "_b2s_outbitc");
        output_conn = &out_bitc->output();
    }

    link_input_tensor(input_conn, op.inputs()->Get(0));
    link_output_tensor(op.outputs()->Get(0), output_conn);
}
