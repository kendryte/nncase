/* Copyright 2019 Canaan Inc.
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
#include <ir/ops/pad.h>
#include <ir/ops/reshape.h>
#include <ir/ops/transpose.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_TFLITE_LOWER(SPACE_TO_BATCH_ND)
{
    auto &input = get_tensor(op.inputs(), 0);
    auto &options = *op.builtin_options_as_SpaceToBatchNDOptions();
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

    auto p = graph_.emplace<pad>(dt_float32, in_shape, new_paddings, 0.f);

    auto padded_shape = p->output().shape();
    shape_t reshapped_shape;
    // batch
    reshapped_shape.push_back(padded_shape[0]);
    // spatial
    for (size_t i = 0; i < spatial_size; i++)
    {
        reshapped_shape.push_back(padded_shape[i + 1] / block_shape[i]);
        reshapped_shape.push_back(block_shape[i]);
    }
    // remaining
    for (size_t i = 0; i < remaining_shape_size; i++)
        reshapped_shape.push_back(padded_shape[1 + spatial_size + i]);

    axis_t perm;
    // block shape
    for (size_t i = 0; i < spatial_size; i++)
        perm.push_back((int32_t)i * 2 + 2);
    // batch
    perm.push_back(0);
    // spatial
    for (size_t i = 0; i < spatial_size; i++)
        perm.push_back((int32_t)i * 2 + 1);
    // remaining
    for (size_t i = 0; i < remaining_shape_size; i++)
        perm.push_back((int32_t)i + spatial_size * 2 + 1);

    shape_t reshapped_shape2;
    // batch * block shape
    reshapped_shape2.push_back(padded_shape[0] * std::accumulate(block_shape.begin(), block_shape.end(), 1, std::multiplies<int32_t>()));
    // spatial
    for (size_t i = 0; i < spatial_size; i++)
        reshapped_shape2.push_back(padded_shape[i + 1] / block_shape[i]);
    // remaining
    for (size_t i = 0; i < remaining_shape_size; i++)
        reshapped_shape2.push_back(padded_shape[1 + spatial_size + i]);

    auto rshape = graph_.emplace<reshape>(dt_float32, p->output().shape(), reshapped_shape);
    auto tp = graph_.emplace<transpose>(dt_float32, rshape->output().shape(), perm);
    auto rshape2 = graph_.emplace<reshape>(dt_float32, tp->output().shape(), reshapped_shape2);
    rshape->input().connect(p->output());
    tp->input().connect(rshape->output());
    rshape2->input().connect(tp->output());

    input_tensors_.emplace(&p->input(), op.inputs()->Get(0));
    output_tensors_.emplace(op.outputs()->Get(0), &rshape2->output());
}
