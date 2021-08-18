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
#include <cassert>
#include <nncase/ir/graph.h>
#include <nncase/ir/op_utils.h>
#include <nncase/ir/ops/concat.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace ncnn;

void nncase::importer::ncnn_importer::convert_op_Concat(const Layer &layer, const ParamDict &pd, const ModelBin & /*mb*/)
{
    const int axis = pd.get(0, 0);

    const auto &op_name = layer.name;

    std::vector<shape_t> input_shapes;
    for (size_t i = 0; i < layer.bottoms.size(); i++)
        input_shapes.push_back(layer.bottom_shapes[i]);

    auto con = graph_.emplace<concat>(dt_float32, input_shapes, axis + 1);
    con->name(op_name + "(Concat)");

    for (size_t i = 0; i < layer.bottoms.size(); i++)
        input_tensors_.emplace(&con->input_at(i), layer.bottoms[i]);

    output_tensors_.emplace(layer.tops[0], &con->output());
}
