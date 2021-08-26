// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "../ncnn_importer.h"
#include "nncase/importer/util.h"
#include "nncase/ir/ir_types.h"
#include <cassert>
#include <nncase/ir/graph.h>
#include <nncase/ir/op_utils.h>
#include <nncase/ir/ops/bitcast.h>
#include <nncase/ir/placeholders.h>
#include <stdexcept>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace ncnn;

void nncase::importer::ncnn_importer::convert_op_Input(const Layer &layer, const ParamDict &pd, const ModelBin & /*mb*/)
{
    const auto &op_name = layer.name;
    auto w = (size_t)pd.get(0, 0);
    auto h = (size_t)pd.get(1, 0);
    auto c = (size_t)pd.get(2, 0);
    shape_t in_shape { c, h, w };
    if (!w || !h || !c)
    {
        // take from shape hints
        in_shape = layer.bottom_shapes[0];
        if (in_shape.empty())
            throw std::runtime_error("Shape of " + layer.name + " must be set in ncnn param file");
    }

    shape_t new_in_shape { 1 };
    for (auto v : in_shape)
        new_in_shape.push_back(v);

    auto node = graph_.emplace<input_node>(dt_float32, new_in_shape);
    node->name(op_name + "(Input)");
    auto rshape = graph_.emplace<bitcast>(node->output().type(), node->output().shape(), in_shape);
    rshape->name(op_name + "(Reshape)");

    rshape->input().connect(node->output());
    output_tensors_.emplace(layer.tops[0], &rshape->output());
}
