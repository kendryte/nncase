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
#include <nncase/ir/ir_types.h>
#include <nncase/ir/ops/binary.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/instancenorm.h>
#include <nncase/ir/ops/reduce.h>
#include <nncase/ir/ops/unary.h>
#include <nncase/ir/visitor.h>
#include <nncase/transforms/neutral/fold_instancenorm.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

bool fold_instancenorm_transform::on_try_match(node &node, transform_context &context)
{
    binary *add_bias = nullptr, *mul_scale = nullptr, *div = nullptr, *add_e = nullptr, *sub_mean = nullptr, *sub_mean_cmp = nullptr;
    unary *u_sqrt = nullptr, *u_square = nullptr;
    reduce *reduce_mean0 = nullptr, *reduce_mean1 = nullptr;
    constant *scale = nullptr, *bias = nullptr, *eps = nullptr;
    if (((add_bias = node_cast<binary>(node)) && (bias = try_get_direct_parent<constant>(*add_bias))) && add_bias->binary_op() == binary_add
        && (div = try_get_direct_parent<binary>(*add_bias)) && div->binary_op() == binary_div
        && (mul_scale = try_get_direct_parent<binary>(*div)) && (scale = try_get_direct_parent<constant>(*mul_scale)) && mul_scale->binary_op() == binary_mul
        && (u_sqrt = try_get_direct_parent<unary>(*div)) && u_sqrt->unary_op() == unary_sqrt
        && (add_e = try_get_direct_parent<binary>(*u_sqrt)) && (eps = try_get_direct_parent<constant>(*add_e)) && add_e->binary_op() == binary_add
        && (reduce_mean0 = try_get_direct_parent<reduce>(*add_e)) && reduce_mean0->reduce_op() == reduce_mean
        && (u_square = try_get_direct_parent<unary>(*reduce_mean0)) && u_square->unary_op() == unary_square
        && ((sub_mean = try_get_direct_parent<binary>(*u_square)) && (sub_mean_cmp = try_get_direct_parent<binary>(*mul_scale))
            && (sub_mean == sub_mean_cmp) && sub_mean->binary_op() == binary_sub)
        && (reduce_mean1 = try_get_direct_parent<reduce>(*sub_mean)) && reduce_mean1->reduce_op() == reduce_mean)
    {
        context.inputs.emplace_back(&reduce_mean1->input());
        context.outputs.emplace_back(&add_bias->output());
        context.matched_nodes.emplace_back(scale);
        context.matched_nodes.emplace_back(bias);
        context.matched_nodes.emplace_back(eps);
        return true;
    }

    return false;
}

void fold_instancenorm_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto scale = node_cast<constant>(*context.matched_nodes[0]);
    auto bias = node_cast<constant>(*context.matched_nodes[1]);
    auto eps = node_cast<constant>(*context.matched_nodes[2]);

    auto instancenorm_ = context.graph.emplace<instancenorm>(output.type(), output.shape(), *reinterpret_cast<const float *>(eps->data().data()));
    instancenorm_->name(scale->name());
    instancenorm_->input().connect(output);
    instancenorm_->scale().connect(scale->output());
    instancenorm_->bias().connect(bias->output());

    for (auto &in : dup(inputs))
        in->connect(instancenorm_->output());
}