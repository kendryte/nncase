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
#include <nncase/ir/ops/convert.h>
#include <nncase/ir/ops/gather.h>
#include <nncase/ir/ops/gather_nd.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

namespace
{
// indices_shape[:-1] + input_shape[indices_shape[-1] + batch_dims:]
shape_t get_gather_nd_shape(const shape_t &input_shape, const shape_t &indices_shape, int32_t batch_dims)
{
    shape_t out_shape(indices_shape.begin(), indices_shape.end() - 1);
    for (size_t i = 1 + batch_dims; i < input_shape.size(); ++i)
    {
        out_shape.push_back(input_shape[i]);
    }
    return out_shape;
}

axis_t get_init_perm(size_t rank)
{
    axis_t perm(rank);
    std::iota(perm.begin(), perm.end(), 0);
    return perm;
}

axis_t get_tr_perm(size_t data_rank, int32_t axis, int32_t batch_dims)
{
    auto perm = get_init_perm(data_rank);
    // perm = perm[:batch_dims] + perm[axis:axis+1] + perm[batch_dims:axis] + perm[axis+1:]
    perm[batch_dims] = axis;
    size_t perm_index = batch_dims + 1;
    for (auto i = batch_dims; i < axis; ++i, ++perm_index)
    {
        perm[perm_index] = i;
    }
    for (auto i = axis + 1; i < data_rank; ++i, ++perm_index)
    {
        perm[perm_index] = i;
    }
    return perm;
}

axis_t get_un_tr_perm(size_t data_rank, int32_t axis, int32_t batch_dims, const shape_t &indices_shape)
{
    const auto indices_rank = indices_shape.size();
    const auto shift_amt = axis - batch_dims;
    const auto result_rank = data_rank + indices_rank - 1 - batch_dims;
    auto un_perm = get_init_perm(result_rank);
    const auto j = indices_rank + shift_amt;
    // un_perm = un_perm[:batch_dims] + un_perm[indices_rank:j] + un_perm[batch_dims:indices_rank] + un_perm[j:]
    size_t perm_index = batch_dims + 1;
    perm_index = batch_dims;
    for (auto i = indices_rank; i < j; ++i, ++perm_index)
    {
        un_perm[perm_index] = i;
    }
    for (auto i = batch_dims; i < indices_rank; ++i, ++perm_index)
    {
        un_perm[perm_index] = i;
    }
    return un_perm;
}

shape_t get_after_bitcast_shape(shape_t shape)
{
    shape.push_back(1);
    return shape;
}
}

DEFINE_TFLITE_LOWER(GATHER)
{
    auto &input = get_tensor(op.inputs(), 0);
    auto &indices = get_tensor(op.inputs(), 1);
    auto &output = get_tensor(op.outputs(), 0);

    auto in_shape = get_shape(input.shape());
    auto indices_shape = get_shape(indices.shape());
    auto out_shape = get_shape(output.shape());

    auto &options = *op.builtin_options_as_GatherOptions();
    auto batch_dims = options.batch_dims();
    const auto in_type = to_data_type(input.type());
    const auto indices_type = to_data_type(indices.type());
    auto axis = options.axis();
    if (axis < 0)
    {
        axis = axis + static_cast<int32_t>(in_shape.size());
    }

    input_connector *g_input, *g_indices;
    output_connector *g_output;
    if (batch_dims == 0)
    {
        auto ga = graph_.emplace<gather>(in_type, in_shape, indices_shape, out_shape, axis);
        ga->name(get_tensor(op.outputs(), 0).name()->string_view());
        link_input_tensor(&ga->input(), op.inputs()->Get(0));
        if (indices_type != dt_int32)
        {
            auto ct = graph_.emplace<convert>(indices_type, indices_shape, dt_int32);
            ga->indices().connect(ct->output());
            link_input_tensor(&ct->input(), op.inputs()->Get(1));
        }
        else
        {
            link_input_tensor(&ga->indices(), op.inputs()->Get(1));
        }
        link_output_tensor(op.outputs()->Get(0), &ga->output());
        return;
    }

    auto after_bitcast_indices_shape = get_after_bitcast_shape(indices_shape);
    auto bc = graph_.emplace<bitcast>(dt_int32, indices_shape, after_bitcast_indices_shape);

    gather_nd *ga;
    if (indices_type != dt_int32)
    {
        auto ct = graph_.emplace<convert>(indices_type, indices_shape, dt_int32);
        g_indices = &ct->input();
        // indices -> convert -> bitcast
        bc->input().connect(ct->output());
    }
    else
    {
        // indices -> bitcast
        g_indices = &bc->input();
    }

    if (axis != batch_dims)
    {
        // before transpose
        const auto data_rank = in_shape.size();
        auto tr = graph_.emplace<transpose>(in_type, in_shape, get_tr_perm(data_rank, axis, batch_dims));
        // gather nd
        auto after_tr_shape = tr->output().shape();
        auto cast_out_shape = get_gather_nd_shape(after_tr_shape, after_bitcast_indices_shape, batch_dims);
        ga = graph_.emplace<gather_nd>(in_type, after_tr_shape, after_bitcast_indices_shape, cast_out_shape, batch_dims);
        // after transpose
        auto un_tr = graph_.emplace<transpose>(in_type, cast_out_shape, get_un_tr_perm(data_rank, axis, batch_dims, indices_shape));
        // connect node
        // input -> transpose -> gather_nd -> transpose -> output
        g_input = &tr->input();
        ga->input().connect(tr->output());
        un_tr->input().connect(ga->output());
        g_output = &un_tr->output();
    }
    else
    {
        auto bitcast_output_shape = get_gather_nd_shape(in_shape, after_bitcast_indices_shape, batch_dims);
        ga = graph_.emplace<gather_nd>(in_type, in_shape, after_bitcast_indices_shape, bitcast_output_shape, batch_dims);
        g_input = &ga->input();
        g_output = &ga->output();
    }

    ga->indices().connect(bc->output());
    link_input_tensor(g_input, op.inputs()->Get(0));
    link_input_tensor(g_indices, op.inputs()->Get(1));
    link_output_tensor(op.outputs()->Get(0), g_output);
    ga->name(get_tensor(op.outputs(), 0).name()->string_view());
}