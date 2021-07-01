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
    auto block_shape = load_axis<int32_t>(get_tensor(op.inputs(), 1));
    auto paddings = load_tensor<int32_t, 2>(get_tensor(op.inputs(), 2));
    auto in_shape = get_shape(input.shape());
    auto spatial_size = block_shape.size();
    auto remaining_shape_size = in_shape.size() - spatial_size - 1;

    xt::svector<padding> new_paddings;
    // batch
    new_paddings.push_back(padding::zero());
    // spatial
    for (size_t i = 0; i < spatial_size; i++)
        new_paddings.push_back(padding { paddings(i, 0), paddings(i, 1) });
    // remaining
    for (size_t i = 0; i < remaining_shape_size; i++)
        new_paddings.push_back(padding::zero());

    auto block_size_h = block_shape.data()[0];
    auto block_size_w = block_shape.data()[1];

    auto tp1 = graph_.emplace<transpose>(to_data_type(input.type()), get_shape(input.shape()), axis_t { 0, 3, 1, 2 });
    auto pad_value = input.type() != tflite::TensorType_FLOAT32 ? static_cast<int8_t>(input.quantization()->zero_point()->data()[0]) : 0.f;
    auto s2b = graph_.emplace<space_to_batch>(tp1->output().type(), tp1->output().shape(), block_size_h, block_size_w, new_paddings[1], new_paddings[2], pad_value);

    auto tp2 = graph_.emplace<transpose>(s2b->output().type(), s2b->output().shape(), axis_t { 0, 2, 3, 1 });
    s2b->input().connect(tp1->output());
    tp2->input().connect(s2b->output());

    auto input_conn = &tp1->input();
    auto output_conn = &tp2->output();

    link_input_tensor(input_conn, op.inputs()->Get(0));
    link_output_tensor(op.outputs()->Get(0), output_conn);
}

DEFINE_TFLITE_LOWER(BATCH_TO_SPACE_ND)
{
    auto &input = get_tensor(op.inputs(), 0);
    //    auto &output = get_tensor(op.outputs(), 0);
    auto block_shape = load_axis<int32_t>(get_tensor(op.inputs(), 1));
    auto crops = load_tensor<int32_t, 2>(get_tensor(op.inputs(), 2));
    auto in_shape = get_shape(input.shape());

    auto block_size_h = block_shape.data()[0];
    auto block_size_w = block_shape.data()[1];

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

    auto tp1 = graph_.emplace<transpose>(to_data_type(input.type()), get_shape(input.shape()), axis_t { 0, 3, 1, 2 });

    //    crops.data()
    std::array<int32_t, 2> crop_h { crops.data()[0], crops.data()[1] };
    std::array<int32_t, 2> crop_w { crops.data()[2], crops.data()[3] };

    auto b2s = graph_.emplace<batch_to_space>(tp1->output().type(), tp1->output().shape(), block_size_h, block_size_w, axis_t(crop_begs.size(), 1), crop_begs, crop_ends, crop_h, crop_w);
    auto tp2 = graph_.emplace<transpose>(b2s->output().type(), b2s->output().shape(), axis_t { 0, 2, 3, 1 });
    b2s->input().connect(tp1->output());
    tp2->input().connect(b2s->output());

    auto input_conn = &tp1->input();
    auto output_conn = &tp2->output();

    link_input_tensor(input_conn, op.inputs()->Get(0));
    link_output_tensor(op.outputs()->Get(0), output_conn);
}
