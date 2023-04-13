// Tencent is pleased to support the open source community by making pnnx available.
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

#include "../pnnx_importer.h"
#include "nncase/importer/util.h"
#include "nncase/ir/ir_types.h"
#include <cassert>
#include <nncase/ir/graph.h>
#include <nncase/ir/op_utils.h>
#include <nncase/ir/ops/binary.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/placeholders.h>
#include <stdexcept>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace pnnx;

void nncase::importer::pnnx_importer::convert_op_F_relu(const Operator &op)
{
    const auto &op_name = op.name;

    auto in_shape = op.inputs[0]->get_shape();

    auto zero = graph_.emplace<constant>(0.f);
    zero->name(op_name + ".zero(ReLU)");

    auto max = graph_.emplace<binary>(binary_max, in_shape, zero->output().shape(), value_range<float>::full());
    max->name(op_name + ".max(ReLU)");

    max->input_b().connect(zero->output());

    input_tensors_.emplace(&max->input_a(), op.inputs[0]->name);
    output_tensors_.emplace(op.outputs[0]->name, &max->output());
}

void nncase::importer::pnnx_importer::convert_op_nn_ReLU(const Operator &op)
{
    convert_op_F_relu(op);
}
