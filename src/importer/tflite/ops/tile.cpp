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
 * multiplesations under the License.
 */
#include "../tflite_importer.h"
#include <nncase/ir/ops/bitcast.h>
#include <nncase/ir/ops/concat.h>
#include <nncase/ir/ops/range.h>
#include <schema_generated.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_TFLITE_LOWER(TILE)
{
    auto &input = get_tensor(op.inputs(), 0);
    auto &multiples = get_tensor(op.inputs(), 1);

    auto &output = get_tensor(op.outputs(), 0);
    auto output_name = std::string(output.name()->string_view());

    auto input_shape = get_shape(input.shape());
    auto input_type = to_data_type(input.type());

    auto multiples_shape = get_shape(multiples.shape());
    auto multiples_type = to_data_type(multiples.type());

    // multiples
    std::vector<int64_t> repeats(input_shape.size(), 1);
    if (multiples_type == dt_int32 && is_constant_tensor<int32_t>(multiples))
    {
        auto arr = load_tensor<int32_t, 1>(multiples);
        assert(input_shape.size() == arr.size());
        repeats.assign(arr.begin(), arr.end());
    }
    else if (multiples_type == dt_int64 && is_constant_tensor<int64_t>(multiples))
    {
        auto arr = load_tensor<int64_t, 1>(multiples);
        assert(input_shape.size() == arr.size());
        repeats.assign(arr.begin(), arr.end());
    }
    else
    {
        throw std::runtime_error("We support static shape for TILE only.");
    }

    auto cur_shape = input_shape;
    concat *pcat = nullptr;
    std::vector<shape_t> concat_shapes;
    int32_t end = repeats.size() - 1;
    for (auto i = end; i >= 0; i--)
    {
        // skip the repeats[i] == 1
        if (repeats[i] == 1)
            continue;

        concat_shapes.clear();
        concat_shapes.assign(repeats[i], cur_shape);
        auto cat = graph_.emplace<concat>(input_type, concat_shapes, i);
        cat->name(output_name + "/concat(axis_" + std::to_string(i) + ")");
        if (pcat == nullptr)
        {
            for (auto j = 0; j < repeats[i]; j++)
                input_tensors_.emplace(&cat->input_at(j), op.inputs()->Get(0));
        }
        else
        {
            for (auto j = 0; j < repeats[i]; j++)
                cat->input_at(j).connect(pcat->output());
        }

        pcat = cat;
        cur_shape = pcat->output().shape();
    }

    if (pcat == nullptr)
    {
        // For all_of(repeats) == 1, just reshape the input tensor
        auto bc = graph_.emplace<bitcast>(input_type, input_shape, input_shape);
        bc->name(output_name + "/reshape");
        input_tensors_.emplace(&bc->input(), op.inputs()->Get(0));
        output_tensors_.emplace(op.outputs()->Get(0), &bc->output());
    }
    else
    {
        output_tensors_.emplace(op.outputs()->Get(0), &pcat->output());
    }
}
