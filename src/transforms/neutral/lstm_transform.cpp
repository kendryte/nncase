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
#include <nncase/ir/ir_types.h>
#include <nncase/ir/ops/binary.h>
#include <nncase/ir/ops/bitcast.h>
#include <nncase/ir/ops/concat.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/lstm.h>
#include <nncase/ir/ops/matmul.h>
#include <nncase/ir/ops/pad.h>
#include <nncase/ir/ops/slice.h>
#include <nncase/ir/ops/transpose.h>
#include <nncase/ir/ops/unary.h>
#include <nncase/ir/visitor.h>
#include <nncase/transforms/neutral/lstm_transform.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

output_connector *local_sigmoid(output_connector *x, transform_context &context)
{
    std::vector<float> one_data(xt::compute_size(x->shape()), 1.f);

    auto one = context.graph.emplace<constant>(dt_float32, x->shape(), one_data);
    auto neg_ = context.graph.emplace<unary>(unary_neg, x->shape());
    auto exp_ = context.graph.emplace<unary>(unary_exp, neg_->output().shape());
    auto add_ = context.graph.emplace<binary>(binary_add, exp_->output().shape(), one->output().shape(), value_range<float>::full());
    auto div_ = context.graph.emplace<binary>(binary_div, one->output().shape(), add_->output().shape(), value_range<float>::full());

    one->name(x->name() + "/one");
    neg_->name(x->name() + "/neg_");
    exp_->name(x->name() + "/exp_");
    add_->name(x->name() + "/add_");
    div_->name(x->name() + "/div_");

    neg_->input().connect(*x);
    exp_->input().connect(neg_->output());
    add_->input_a().connect(exp_->output());
    add_->input_b().connect(one->output());
    div_->input_a().connect(one->output());
    div_->input_b().connect(add_->output());

    return &div_->output();
}

output_connector *local_tanh(output_connector *x, transform_context &context)
{
    std::vector<float> one_data(xt::compute_size(x->shape()), 1.f);
    std::vector<float> two_data(xt::compute_size(x->shape()), 2.f);

    auto two_ = context.graph.emplace<constant>(dt_float32, x->shape(), two_data);
    auto one_ = context.graph.emplace<constant>(dt_float32, x->shape(), one_data);
    auto mul_1 = context.graph.emplace<binary>(binary_mul, x->shape(), two_->output().shape(), value_range<float>::full());
    auto sigm = local_sigmoid(&mul_1->output(), context);
    auto mul_2 = context.graph.emplace<binary>(binary_mul, sigm->shape(), two_->output().shape(), value_range<float>::full());
    auto sub_ = context.graph.emplace<binary>(binary_sub, mul_2->output().shape(), one_->output().shape(), value_range<float>::full());

    one_->name(x->name() + "/one_");
    two_->name(x->name() + "/two_");
    mul_1->name(x->name() + "/mul_1");
    mul_2->name(x->name() + "/mul_2");
    sub_->name(x->name() + "/sub_");

    mul_1->input_a().connect(*x);
    mul_1->input_b().connect(two_->output());
    mul_2->input_a().connect(*sigm);
    mul_2->input_b().connect(two_->output());
    sub_->input_a().connect(mul_2->output());
    sub_->input_b().connect(one_->output());

    return &sub_->output();
}

bool lstm_transform::on_try_match(node &node, transform_context &context)
{
    constant *w_xc, *b_xc, *w_hc;
    if (auto old_lstm = node_cast<lstm>(node))
    {
        if ((w_xc = try_get_direct_parent<constant>(*old_lstm, 1))
            && (b_xc = try_get_direct_parent<constant>(*old_lstm, 2))
            && (w_hc = try_get_direct_parent<constant>(*old_lstm, 3)))
        {
            context.inputs.emplace_back(&old_lstm->input());
            context.inputs.emplace_back(&old_lstm->w_xc());
            context.inputs.emplace_back(&old_lstm->b_xc());
            context.inputs.emplace_back(&old_lstm->w_hc());

            context.matched_nodes.emplace_back(old_lstm);
            context.matched_nodes.emplace_back(w_xc);
            context.matched_nodes.emplace_back(b_xc);
            context.matched_nodes.emplace_back(w_hc);
            if (old_lstm->has_static())
            {
                if (auto w_static = try_get_direct_parent<constant>(*old_lstm, 4))
                {
                    context.inputs.emplace_back(&old_lstm->input());
                    context.matched_nodes.emplace_back(w_static);
                }
                else
                {
                    return false;
                }
            }

            context.outputs.emplace_back(&old_lstm->output());

            return true;
        }
    }

    return false;
}

void lstm_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &old_lstm = static_cast<lstm &>(*context.matched_nodes[0]);
    auto &w_xc = static_cast<constant &>(*context.matched_nodes[1]);
    auto &b_xc = static_cast<constant &>(*context.matched_nodes[2]);
    auto &w_hc = static_cast<constant &>(*context.matched_nodes[3]);

    auto tp_wxc = context.graph.emplace<transpose>(dt_float32, w_xc.output().shape(), axis_t { 1, 0 });
    auto bitcast_wxc_pre = context.graph.emplace<bitcast>(dt_float32, old_lstm.input().shape(),
        shape_t { old_lstm.input().shape()[0] * old_lstm.input().shape()[1], old_lstm.input().shape()[2] });
    auto matmul_wxc = context.graph.emplace<matmul>(bitcast_wxc_pre->output().shape(), tp_wxc->output().shape(), value_range<float>::full());
    auto bitcast_wxc_post = context.graph.emplace<bitcast>(dt_float32, matmul_wxc->output().shape(),
        shape_t { old_lstm.input().shape()[0], old_lstm.input().shape()[1], matmul_wxc->output().shape()[1] });
    // auto matmul_bxc = context.graph.emplace<binary>(binary_add, matmul_wxc->output().shape(), b_xc.output().shape(), value_range<float>::full());
    matmul_wxc->name(old_lstm.name() + "/matmul_wxc");
    // matmul_bxc->name(old_lstm.name() + "/matmul_bxc");
    tp_wxc->input().connect(w_xc.output());
    bitcast_wxc_pre->input().connect(output);
    matmul_wxc->input_a().connect(bitcast_wxc_pre->output());
    matmul_wxc->input_b().connect(tp_wxc->output());
    matmul_wxc->bias().connect(b_xc.output());
    bitcast_wxc_post->input().connect(matmul_wxc->output());
    // matmul_bxc->input_a().connect(matmul_wxc->output());
    // matmul_bxc->input_b().connect(b_xc.output());

    //transpose w_hc
    auto tp_whc = context.graph.emplace<transpose>(dt_float32, w_hc.output().shape(), axis_t { 1, 0 });
    tp_whc->input().connect(w_hc.output());

    std::vector<float> constant_data((int)bitcast_wxc_post->output().shape()[1] * (int)bitcast_wxc_post->output().shape()[2] / 4, 0.f);
    auto c_0 = context.graph.emplace<constant>(dt_float32, shape_t { 1, bitcast_wxc_post->output().shape()[1], bitcast_wxc_post->output().shape()[2] / 4 }, constant_data);
    auto h_0 = context.graph.emplace<constant>(dt_float32, shape_t { 1, bitcast_wxc_post->output().shape()[1], bitcast_wxc_post->output().shape()[2] / 4 }, constant_data);
    auto c_ = &c_0->output();
    auto h_ = &h_0->output();

    std::vector<shape_t> lstm_h_s;
    for (size_t i = 0; i < (size_t)output.shape()[0]; i++)
    {
        lstm_h_s.push_back(h_0->output().shape());
    }
    //h_concat
    auto h_concat = context.graph.emplace<concat>(dt_float32, lstm_h_s, 0);

    // 0 constant for matmul
    std::vector<float> zero(tp_whc->output().shape()[1], 0.f);
    auto zero_constant = context.graph.emplace<constant>(dt_float32, shape_t { tp_whc->output().shape()[1] }, zero);

    for (size_t i = 0; i < (size_t)(bitcast_wxc_post->output().shape()[0]); i++)
    {
        std::vector<float> cont_data((int)bitcast_wxc_post->output().shape()[1], (i == 0) ? 0.f : 1.f);
        auto cont_ = context.graph.emplace<constant>(dt_float32, shape_t { 1, 1 }, cont_data);
        cont_->name(old_lstm.name() + "/cont_" + std::to_string(i));

        //slice
        auto scale_ = context.graph.emplace<binary>(binary_mul, h_->shape(), cont_->output().shape(), value_range<float>::full());
        scale_->name(old_lstm.name() + "/scale_" + std::to_string(i));
        scale_->input_a().connect(*h_);
        scale_->input_b().connect(cont_->output());

        //w_hc_h
        auto bitcast_whc_pre = context.graph.emplace<bitcast>(dt_float32, scale_->output().shape(),
            shape_t { scale_->output().shape()[0] * scale_->output().shape()[1], scale_->output().shape()[2] });
        auto w_hc_h = context.graph.emplace<matmul>(bitcast_whc_pre->output().shape(), tp_whc->output().shape(), value_range<float>::full());
        auto bitcast_whc_post = context.graph.emplace<bitcast>(dt_float32, w_hc_h->output().shape(),
            shape_t { scale_->output().shape()[0], scale_->output().shape()[1], w_hc_h->output().shape()[1] });
        w_hc_h->name(old_lstm.name() + "/w_hc_h_" + std::to_string(i));
        bitcast_whc_pre->input().connect(scale_->output());
        w_hc_h->input_a().connect(bitcast_whc_pre->output());
        w_hc_h->input_b().connect(tp_whc->output());
        w_hc_h->bias().connect(zero_constant->output());
        bitcast_whc_post->input().connect(w_hc_h->output());

        //slice w_xc_x
        auto w_xc_x = context.graph.emplace<slice>(bitcast_wxc_post->output().type(), bitcast_wxc_post->output().shape(),
            axis_t { (int32_t)i, 0, 0 }, axis_t { (int32_t)i + 1, (int32_t)bitcast_wxc_post->output().shape()[1], (int32_t)bitcast_wxc_post->output().shape()[2] });
        w_xc_x->name(old_lstm.name() + "/w_xc_x_" + std::to_string(i));
        w_xc_x->input().connect(bitcast_wxc_post->output());

        auto gate_input = context.graph.emplace<binary>(binary_add, w_xc_x->output().shape(), bitcast_whc_post->output().shape(), value_range<float>::full());
        gate_input->name(old_lstm.name() + "/gate_input_" + std::to_string(i));
        gate_input->input_a().connect(w_xc_x->output());
        gate_input->input_b().connect(bitcast_whc_post->output());

        // lstm_uint: need [c_, gate_input:[in_sigmoid:[i_t,f_t,o_t,g_t], in_tanh[g_t]]]
        // slice the in_sidmoid into i_t, f_t, o_t, g_t
        auto in_sigmoid = local_sigmoid(&gate_input->output(), context);
        auto in_tanh = local_tanh(&gate_input->output(), context);
        // i_t
        auto i_t = context.graph.emplace<slice>(in_sigmoid->type(), in_sigmoid->shape(),
            axis_t { 0, 0, 0 * (int32_t)c_->shape()[2] }, axis_t { (int32_t)in_sigmoid->shape()[0], (int32_t)in_sigmoid->shape()[1], (int32_t)c_->shape()[2] });
        i_t->name(old_lstm.name() + "/i_t_" + std::to_string(i));
        i_t->input().connect(*in_sigmoid);

        // f_t
        auto f_t = context.graph.emplace<slice>(in_sigmoid->type(), in_sigmoid->shape(),
            axis_t { 0, 0, 1 * (int32_t)c_->shape()[2] }, axis_t { (int32_t)in_sigmoid->shape()[0], (int32_t)in_sigmoid->shape()[1], 2 * (int32_t)c_->shape()[2] });
        f_t->name(old_lstm.name() + "/f_t_" + std::to_string(i));
        f_t->input().connect(*in_sigmoid);

        // o_t
        auto o_t = context.graph.emplace<slice>(in_sigmoid->type(), in_sigmoid->shape(),
            axis_t { 0, 0, 2 * (int32_t)c_->shape()[2] }, axis_t { (int32_t)in_sigmoid->shape()[0], (int32_t)in_sigmoid->shape()[1], 3 * (int32_t)c_->shape()[2] });
        o_t->name(old_lstm.name() + "/o_t_" + std::to_string(i));
        o_t->input().connect(*in_sigmoid);

        // g_t
        auto g_t = context.graph.emplace<slice>(in_tanh->type(), in_tanh->shape(),
            axis_t { 0, 0, 3 * (int32_t)c_->shape()[2] }, axis_t { (int32_t)in_sigmoid->shape()[0], (int32_t)in_sigmoid->shape()[1], 4 * (int32_t)c_->shape()[2] });
        g_t->name(old_lstm.name() + "/g_t_" + std::to_string(i));
        g_t->input().connect(*in_tanh);

        //c_t = cont_ * (f * c_) + (i * g)
        auto f_c_mul = context.graph.emplace<binary>(binary_mul, c_->shape(), f_t->output().shape(), value_range<float>::full());
        f_c_mul->name(old_lstm.name() + "/f_c_mul_" + std::to_string(i));
        f_c_mul->input_a().connect(*c_);
        f_c_mul->input_b().connect(f_t->output());

        auto c_f_c_mul = context.graph.emplace<binary>(binary_mul, cont_->output().shape(), f_c_mul->output().shape(), value_range<float>::full());
        c_f_c_mul->name(old_lstm.name() + "/c_f_c_mul_" + std::to_string(i));
        c_f_c_mul->input_a().connect(cont_->output());
        c_f_c_mul->input_b().connect(f_c_mul->output());

        auto i_g_mul = context.graph.emplace<binary>(binary_mul, i_t->output().shape(), g_t->output().shape(), value_range<float>::full());
        i_g_mul->name(old_lstm.name() + "/i_g_mul_" + std::to_string(i));
        i_g_mul->input_a().connect(i_t->output());
        i_g_mul->input_b().connect(g_t->output());

        auto c_t = context.graph.emplace<binary>(binary_add, c_f_c_mul->output().shape(), i_g_mul->output().shape(), value_range<float>::full());
        c_t->name(old_lstm.name() + "/c_t_" + std::to_string(i));
        c_t->input_a().connect(c_f_c_mul->output());
        c_t->input_b().connect(i_g_mul->output());

        //h_t = o_t * tanh(c_t)
        auto tanh_c_t = local_tanh(&c_t->output(), context);
        auto h_t = context.graph.emplace<binary>(binary_mul, o_t->output().shape(), tanh_c_t->shape(), value_range<float>::full());
        h_t->name(old_lstm.name() + "/h_t_" + std::to_string(i));
        h_t->input_a().connect(o_t->output());
        h_t->input_b().connect(*tanh_c_t);

        h_concat->input_at(i).connect(h_t->output());
        // update c_, h_
        c_ = &c_t->output();
        h_ = &h_t->output();
    }

    for (auto &in : dup(inputs))
        in->connect(h_concat->output());
}
