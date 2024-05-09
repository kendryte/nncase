/* Copyright 2020 Canaan Inc.
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
#include <nncase/ir/ops/binary.h>
#include <nncase/ir/ops/bitcast.h>
#include <nncase/ir/ops/broadcast.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/layernorm.h>
#include <nncase/ir/ops/transpose.h>
#include <nncase/ir/ops/reduce.h>
#include <nncase/ir/ops/unary.h>
#include <nncase/ir/visitor.h>
#include <nncase/runtime/datatypes.h>
#include <nncase/transforms/neutral/fold_layernorm.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

bool fold_layernorm_pattern1_transform::on_try_match(node &node, transform_context &context)
{
    reduce *rd1 = nullptr, *rd2 = nullptr;
    binary *sub = nullptr, *pow = nullptr, *add_eps = nullptr, *div = nullptr, *mul = nullptr, *add_beta = nullptr;
    unary *sqrt = nullptr;
    bitcast *rshape1 = nullptr, *rshape2 = nullptr;

    if ((add_beta = node_cast<binary>(node)) and add_beta->binary_op() == binary_op_t::binary_add
        and (mul = try_get_direct_parent<binary>(*add_beta)) and mul->binary_op() == binary_op_t::binary_mul
        and (rshape2 = try_get_direct_parent<bitcast>(*mul))
        and (div = try_get_direct_parent<binary>(*rshape2)) and div->binary_op() == binary_op_t::binary_div
        and (sqrt = try_get_direct_parent<unary>(*div)) and sqrt->unary_op() == unary_op_t::unary_sqrt
        and (add_eps = try_get_direct_parent<binary>(*sqrt)) and add_eps->binary_op() == binary_op_t::binary_add
        and (rd2 = try_get_direct_parent<reduce>(*add_eps)) and rd2->reduce_op() == reduce_op_t::reduce_mean
        and (pow = try_get_direct_parent<binary>(*rd2)) and pow->binary_op() == binary_op_t::binary_pow
        and (sub = try_get_direct_parent<binary>(*pow)) and sub->binary_op() == binary_op_t::binary_sub
        and (rd1 = try_get_direct_parent<reduce>(*sub)) and rd1->reduce_op() == reduce_op_t::reduce_mean
        and (rshape1 = try_get_direct_parent<bitcast>(*rd1))
        and (sub->input_a().connection() == rd1->input().connection() or sub->input_b().connection() == rd1->input().connection())
        and try_get_direct_parent<binary>(*div) == sub)
    {
        context.inputs.emplace_back(&rshape1->input());
        context.outputs.emplace_back(&add_beta->output());
        
        if(auto esp_const = try_get_direct_parent<constant>(*add_eps))
            context.matched_nodes.emplace_back(esp_const);
        else
            return false;
        
        if(auto mul_const = try_get_direct_parent<constant>(*mul))
            context.matched_nodes.emplace_back(mul_const);
        else
            return false;
        
        if (auto add_beta_const = try_get_direct_parent<constant>(*add_beta))
            context.matched_nodes.emplace_back(add_beta_const);
        else
            return false;       
        

        return true;
    }

    return false;
}

void fold_layernorm_pattern1_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto eps = node_cast<constant>(*context.matched_nodes[0]);
    auto gamma = node_cast<constant>(*context.matched_nodes[1]);
    auto beta = node_cast<constant>(*context.matched_nodes[2]);

    auto axis = output.shape().size() - gamma->output().shape().size();
    auto ln = context.graph.emplace<layernorm>(output.type(), output.shape(), axis, *reinterpret_cast<const float *>(eps->data().data()));
    ln->name(output.name() + "/layernorm");

    ln->input().connect(output);
    ln->scale().connect(gamma->output());
    ln->bias().connect(beta->output());

    for (auto &in : dup(inputs))
        in->connect(ln->output());
}

bool fold_layernorm_pattern2_transform::on_try_match(node &node, transform_context &context)
{
    reduce *rd1 = nullptr, *rd2 = nullptr;
    binary *sub = nullptr, *pow = nullptr, *add_eps = nullptr, *div = nullptr, *mul = nullptr, *add_beta = nullptr;
    unary *sqrt = nullptr;

    if ((add_beta = node_cast<binary>(node)) and add_beta->binary_op() == binary_op_t::binary_add
        and (mul = try_get_direct_parent<binary>(*add_beta)) and mul->binary_op() == binary_op_t::binary_mul
        and (div = try_get_direct_parent<binary>(*mul)) and div->binary_op() == binary_op_t::binary_div
        and (sqrt = try_get_direct_parent<unary>(*div)) and sqrt->unary_op() == unary_op_t::unary_sqrt
        and (add_eps = try_get_direct_parent<binary>(*sqrt)) and add_eps->binary_op() == binary_op_t::binary_add
        and (rd2 = try_get_direct_parent<reduce>(*add_eps)) and rd2->reduce_op() == reduce_op_t::reduce_mean
        and (pow = try_get_direct_parent<binary>(*rd2)) and pow->binary_op() == binary_op_t::binary_pow
        and ((sub = try_get_direct_parent<binary>(*pow, 0)) or (sub = try_get_direct_parent<binary>(*pow, 1))) and sub->binary_op() == binary_op_t::binary_sub
        and (rd1 = try_get_direct_parent<reduce>(*sub)) and rd1->reduce_op() == reduce_op_t::reduce_mean
        and (sub->input_a().connection() == rd1->input().connection() or sub->input_b().connection() == rd1->input().connection())
        and try_get_direct_parent<binary>(*div) == sub)
    {
        context.inputs.emplace_back(&rd1->input());
        context.outputs.emplace_back(&add_beta->output());

        context.matched_nodes.emplace_back(rd1);
        
        if(auto esp_const = try_get_direct_parent<constant>(*add_eps))
            context.matched_nodes.emplace_back(esp_const);
        else
            return false;
        
        if(auto mul_const = try_get_direct_parent<constant>(*mul))
            context.matched_nodes.emplace_back(mul_const);
        else
            return false;
        
        if (auto add_beta_const = try_get_direct_parent<constant>(*add_beta))
            context.matched_nodes.emplace_back(add_beta_const);
        else
            return false;       
            
        return true;
    }

    return false;
}

void fold_layernorm_pattern2_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto input_shape = output.shape();
    int axis = node_cast<reduce>(*context.matched_nodes[0])->axis()[0]; // Here axis is a scalar

    auto eps = node_cast<constant>(*context.matched_nodes[1]);
    auto gamma = node_cast<constant>(*context.matched_nodes[2]);
    auto beta = node_cast<constant>(*context.matched_nodes[3]);

    auto ln = context.graph.emplace<layernorm>(output.type(), output.shape(), axis, *reinterpret_cast<const float *>(eps->data().data()), gamma->output().shape());
    ln->name(output.name() + "/layernorm");

    ln->input().connect(output);
    ln->scale().connect(gamma->output());
    ln->bias().connect(beta->output());

    for (auto &in : dup(inputs))
        in->connect(ln->output());
}

bool fold_layernorm_pattern3_transform::on_try_match(node &node, transform_context &context)
{
    reduce *rd_mu = nullptr, *rd_var = nullptr;
    binary *sub_mu = nullptr, *add_eps = nullptr, *mul_gamma = nullptr, *mul_x = nullptr, *mul_mu = nullptr, *sub_beta = nullptr, *add_all = nullptr;
    unary *rsqrt = nullptr, *square = nullptr;

    if ((add_all = node_cast<binary>(node)) and add_all->binary_op() == binary_op_t::binary_add
        and (mul_x = try_get_direct_parent<binary>(*add_all, 0)) and mul_x->binary_op() == binary_op_t::binary_mul
        and (sub_beta = try_get_direct_parent<binary>(*add_all, 1)) and sub_beta->binary_op() == binary_op_t::binary_sub
        and (mul_gamma = try_get_direct_parent<binary>(*mul_x, 1)) and mul_gamma->binary_op() == binary_op_t::binary_mul
        and (rsqrt = try_get_direct_parent<unary>(*mul_gamma, 0)) and rsqrt->unary_op() == unary_op_t::unary_rsqrt
        and (add_eps = try_get_direct_parent<binary>(*rsqrt)) and add_eps->binary_op() == binary_op_t::binary_add
        and (rd_var = try_get_direct_parent<reduce>(*add_eps, 0)) and rd_var->reduce_op() == reduce_op_t::reduce_mean
        and (square = try_get_direct_parent<unary>(*rd_var)) and square->unary_op() == unary_op_t::unary_square
        and (sub_mu = try_get_direct_parent<binary>(*square)) and sub_mu->binary_op() == binary_op_t::binary_sub
        and (rd_mu = try_get_direct_parent<reduce>(*sub_mu, 1)) and rd_mu->reduce_op() == reduce_op_t::reduce_mean
        and (mul_mu = try_get_direct_parent<binary>(*sub_beta, 1)) and mul_mu->binary_op() == binary_op_t::binary_mul
        and (mul_mu->input_a().connection() == sub_mu->input_b().connection())
        and (mul_mu->input_b().connection() == mul_x->input_b().connection())
        and (mul_x->input_a().connection() == sub_mu->input_a().connection())
        and (mul_x->input_a().connection() == rd_mu->input().connection()))
    {
        context.inputs.emplace_back(&rd_mu->input());
        context.outputs.emplace_back(&add_all->output());

        context.matched_nodes.emplace_back(rd_mu);
        context.matched_nodes.emplace_back(sub_mu);
        context.matched_nodes.emplace_back(square);
        context.matched_nodes.emplace_back(rd_var);
        context.matched_nodes.emplace_back(add_eps);
        context.matched_nodes.emplace_back(rsqrt);
        context.matched_nodes.emplace_back(mul_gamma);
        context.matched_nodes.emplace_back(mul_x);
        context.matched_nodes.emplace_back(mul_mu);
        context.matched_nodes.emplace_back(sub_beta);
        context.matched_nodes.emplace_back(add_all);

        return true;
    }

    return false;
}

void fold_layernorm_pattern3_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto eps = node_cast<constant>(context.matched_nodes[4]->input_at(1).connection()->owner());
    auto gamma = node_cast<constant>(context.matched_nodes[6]->input_at(1).connection()->owner());
    auto beta = node_cast<constant>(context.matched_nodes[9]->input_at(0).connection()->owner());

    auto axis = output.shape().size() - gamma->output().shape().size();
    auto ln = context.graph.emplace<layernorm>(output.type(), output.shape(), axis, *reinterpret_cast<const float *>(eps->data().data()));
    ln->name(output.name() + "/layernorm");

    ln->input().connect(output);
    ln->scale().connect(gamma->output());
    ln->bias().connect(beta->output());

    for (auto &in : dup(inputs))
        in->connect(ln->output());
}

bool convert_layernorm_to_channel_last::on_try_match(node &node, transform_context &context)
{
    if (auto ln = node_cast<layernorm>(node))
    {
        // if channel is last, skip.
        if(ln->axis() == ln->output().shape().size() - 1)
            return false;

        context.inputs.emplace_back(&ln->input());
        context.outputs.emplace_back(&ln->output());

        context.matched_nodes.emplace_back(ln);
        if (auto scale = try_get_direct_parent<constant>(*ln, 1))
            context.matched_nodes.emplace_back(scale);
        else
            return false;
            
        if (auto bias = try_get_direct_parent<constant>(*ln, 2))
            context.matched_nodes.emplace_back(bias);
        else
            return false;

        return true;
    }

    return false;
}

void convert_layernorm_to_channel_last::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto input_shape = output.shape();
    auto ln = node_cast<layernorm>(*context.matched_nodes[0]);
    auto gamma = node_cast<constant>(*context.matched_nodes[1]);
    auto beta = node_cast<constant>(*context.matched_nodes[2]);
    
    int axis = ln->axis();

    auto get_perm_with_axis = [&](std::vector<int> &perm, int axis, int shape_size) {
        for(int i = 0; i < shape_size; i++)
        {
            if(i != axis)
                perm.push_back(i);
        }
        perm.push_back(axis);
    };

    std::vector<int> in_perm, out_perm;

    get_perm_with_axis(in_perm, axis, input_shape.size());

    for(int i = 0; i < output.shape().size(); i++)
    {
        out_perm.emplace_back(in_perm[in_perm[i]]);
    }
    
    output_connector *tp_gamma = &gamma->output(), *tp_beta = &beta->output();
    if(gamma->output().shape().size() != 1)
    {
        std::vector<int> const_perm;
        int axis_gap = input_shape.size() - gamma->output().shape().size();
        get_perm_with_axis(const_perm, axis - axis_gap, gamma->output().shape().size());
        auto new_gamma = context.graph.emplace<transpose>(tp_gamma->type(), tp_gamma->shape(), axis_t { const_perm.begin(), const_perm.end() });
        auto new_beta = context.graph.emplace<transpose>(tp_beta->type(), tp_beta->shape(), axis_t { const_perm.begin(), const_perm.end() });
        
        new_gamma->input().connect(gamma->output());
        new_beta->input().connect(beta->output());
        
        tp_gamma = &new_gamma->output();
        tp_beta = &new_beta->output();
    }

    auto tp_in = context.graph.emplace<transpose>(output.type(), output.shape(), axis_t { in_perm.begin(), in_perm.end()});
    auto new_ln = context.graph.emplace<layernorm>(output.type(), tp_in->output().shape(), tp_in->output().shape().size()-1, ln->epsilon(), tp_gamma->shape());
    auto tp_out = context.graph.emplace<transpose>(output.type(), new_ln->output().shape(), axis_t { out_perm.begin(), out_perm.end()});
    new_ln->name(ln->name());

    tp_in->input().connect(output);
    tp_out->input().connect(new_ln->output());

    new_ln->input().connect(tp_in->output());
    new_ln->scale().connect(*tp_gamma);
    new_ln->bias().connect(*tp_beta);

    for (auto &in : dup(inputs))
        in->connect(tp_out->output());
}