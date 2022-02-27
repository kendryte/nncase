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

output_connector *lstm_transform::local_sigmoid(output_connector *x, transform_context &context, std::string i)
{
    std::vector<float> one_data(1, 1.f);

    auto one = context.graph.emplace<constant>(dt_float32, shape_t { 1 }, one_data);
    auto neg_ = context.graph.emplace<unary>(unary_neg, x->shape());
    auto exp_ = context.graph.emplace<unary>(unary_exp, neg_->output().shape());
    auto add_ = context.graph.emplace<binary>(binary_add, x->type(), exp_->output().shape(), one->output().shape(), value_range<float>::full());
    auto div_ = context.graph.emplace<binary>(binary_div, x->type(), one->output().shape(), add_->output().shape(), value_range<float>::full());

    one->name(x->owner().name() + "/sig_one" + i);
    neg_->name(x->owner().name() + "/sig_neg_" + i);
    exp_->name(x->owner().name() + "/sig_exp_" + i);
    add_->name(x->owner().name() + "/sig_add_" + i);
    div_->name(x->owner().name() + "/sig_div_" + i);

    neg_->input().connect(*x);
    exp_->input().connect(neg_->output());
    add_->input_a().connect(exp_->output());
    add_->input_b().connect(one->output());
    div_->input_a().connect(one->output());
    div_->input_b().connect(add_->output());

    return &div_->output();
}

output_connector *lstm_transform::local_tanh(output_connector *x, transform_context &context, std::string i)
{
    std::vector<float> one_data(1, 1.f);
    std::vector<float> two_data(1, 2.f);

    auto two_ = context.graph.emplace<constant>(dt_float32, shape_t { 1 }, two_data);
    auto one_ = context.graph.emplace<constant>(dt_float32, shape_t { 1 }, one_data);
    auto mul_1 = context.graph.emplace<binary>(binary_mul, x->type(), x->shape(), two_->output().shape(), value_range<float>::full());
    one_->name(x->owner().name() + "/tanh_one_" + i);
    two_->name(x->owner().name() + "/tanh_two_" + i);
    mul_1->name(x->owner().name() + "/tanh_mul_" + i + "_1");

    auto sigm = local_sigmoid(&mul_1->output(), context, "_tanh_" + i + "_");
    auto mul_2 = context.graph.emplace<binary>(binary_mul, sigm->type(), sigm->shape(), two_->output().shape(), value_range<float>::full());
    auto sub_ = context.graph.emplace<binary>(binary_sub, mul_2->output().type(), mul_2->output().shape(), one_->output().shape(), value_range<float>::full());
    mul_2->name(x->owner().name() + "/tanh_mul_" + i + "_2");
    sub_->name(x->owner().name() + "/tanh_sub_" + i);

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
    constant *w, *r, *b, *init_h, *init_c;
    if (auto old_lstm = node_cast<lstm>(node))
    {
        if ((w = try_get_direct_parent<constant>(*old_lstm, 1))
            && (r = try_get_direct_parent<constant>(*old_lstm, 2))
            && (b = try_get_direct_parent<constant>(*old_lstm, 3)))
        {
            context.inputs.emplace_back(&old_lstm->input());
            context.inputs.emplace_back(&old_lstm->w());
            context.inputs.emplace_back(&old_lstm->r());
            context.inputs.emplace_back(&old_lstm->b());

            context.matched_nodes.emplace_back(old_lstm);
            context.matched_nodes.emplace_back(w);
            context.matched_nodes.emplace_back(r);
            context.matched_nodes.emplace_back(b);
            if ((init_h = try_get_direct_parent<constant>(*old_lstm, 4))
                && (init_c = try_get_direct_parent<constant>(*old_lstm, 5)))
            {
                context.matched_nodes.emplace_back(init_h);
                context.matched_nodes.emplace_back(init_c);
            }
            else
            {
                context.inputs.emplace_back(&old_lstm->initial_h());
                context.inputs.emplace_back(&old_lstm->initial_c());
            }
            if (old_lstm->has_static())
            {
                if (auto w_static = try_get_direct_parent<constant>(*old_lstm, 6))
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
            context.outputs.emplace_back(&old_lstm->output_h());
            context.outputs.emplace_back(&old_lstm->output_c());

            return true;
        }
    }

    return false;
}

void lstm_transform::process(transform_context &context)
{
    auto inputs = context.outputs[0]->connections();
    auto connect_h = context.outputs[1]->connections();
    auto connect_c = context.outputs[2]->connections();

    auto &old_lstm = static_cast<lstm &>(*context.matched_nodes[0]);
    if (old_lstm.direction() == kForward)
    {
        auto outs = forward(context);
        assert(outs.size() == 3);

        for (auto &in : dup(inputs))
            in->connect(*outs[0]);
        for (auto &in : dup(connect_h))
            in->connect(*outs[1]);
        for (auto &in : dup(connect_c))
            in->connect(*outs[2]);
    }
    else if (old_lstm.direction() == kReverse)
    {
        auto outs = reverse(context);
        assert(outs.size() == 3);

        for (auto &in : dup(inputs))
            in->connect(*outs[0]);
        for (auto &in : dup(connect_h))
            in->connect(*outs[1]);
        for (auto &in : dup(connect_c))
            in->connect(*outs[2]);
    }
    else
    {
        auto forward_outs = forward(context);
        auto reverse_outs = reverse(context);

        // concat forward and reverse at num_directions for y
        std::vector<shape_t> y_shapes { forward_outs[0]->shape(), reverse_outs[0]->shape() };
        auto cc_y = context.graph.emplace<concat>(dt_float32, y_shapes, 1);
        cc_y->name(old_lstm.name() + "/y_concat");
        cc_y->input_at(0).connect(*forward_outs[0]);
        cc_y->input_at(1).connect(*reverse_outs[0]);
        for (auto &in : dup(inputs))
            in->connect(cc_y->output());

        // concat forward and reverse at num_directions for y_h
        std::vector<shape_t> y_h_shapes { forward_outs[1]->shape(), reverse_outs[1]->shape() };
        auto cc_y_h = context.graph.emplace<concat>(dt_float32, y_h_shapes, 0);
        cc_y_h->name(old_lstm.name() + "/y_h_concat");
        cc_y_h->input_at(0).connect(*forward_outs[1]);
        cc_y_h->input_at(1).connect(*reverse_outs[1]);
        for (auto &in : dup(connect_h))
            in->connect(cc_y_h->output());

        // concat forward and reverse at num_directions for y_c
        std::vector<shape_t> y_c_shapes { forward_outs[2]->shape(), reverse_outs[2]->shape() };
        auto cc_y_c = context.graph.emplace<concat>(dt_float32, y_c_shapes, 0);
        cc_y_c->name(old_lstm.name() + "/y_h_concat");
        cc_y_c->input_at(0).connect(*forward_outs[2]);
        cc_y_c->input_at(1).connect(*reverse_outs[2]);
        for (auto &in : dup(connect_c))
            in->connect(cc_y_c->output());
    }
}

std::vector<output_connector *> lstm_transform::forward(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto &old_lstm = static_cast<lstm &>(*context.matched_nodes[0]);
    auto &w = static_cast<constant &>(*context.matched_nodes[1]);
    auto &r = static_cast<constant &>(*context.matched_nodes[2]);
    auto &b = static_cast<constant &>(*context.matched_nodes[3]);

    // reshape input
    auto input_shape = old_lstm.input().shape();
    auto input_type = old_lstm.input().type();
    auto input_bc = context.graph.emplace<bitcast>(input_type, input_shape, shape_t { input_shape[0] * input_shape[1], input_shape[2] });
    input_bc->name(old_lstm.name() + "/input_bc");
    input_bc->input().connect(output);

    // slice weight
    auto w_shape = w.output().shape();
    auto sl_w = context.graph.emplace<slice>(input_type, w_shape, axis_t { 0, 0, 0 },
        axis_t { 1, (int32_t)w_shape[1], (int32_t)w_shape[2] });
    sl_w->name(old_lstm.name() + "/sl_w");
    sl_w->input().connect(w.output());

    // reshape weight
    auto bc_w = context.graph.emplace<bitcast>(input_type, sl_w->output().shape(), shape_t { w_shape[1], w_shape[2] });
    bc_w->name(old_lstm.name() + "/bc_w");
    bc_w->input().connect(sl_w->output());

    // transpose weight
    auto tp_w = context.graph.emplace<transpose>(input_type, bc_w->output().shape(), axis_t { 1, 0 });
    tp_w->name(old_lstm.name() + "/tp_w");
    tp_w->input().connect(bc_w->output());

    // slice weight bias
    auto b_shape = b.output().shape();
    auto sl_w_b = context.graph.emplace<slice>(input_type, b_shape, axis_t { 0, 0 }, axis_t { 1, (int32_t)(b_shape[1] / 2) });
    sl_w_b->name(old_lstm.name() + "/sl_w_b");
    sl_w_b->input().connect(b.output());

    // reshape weight bias
    auto bc_w_b = context.graph.emplace<bitcast>(input_type, sl_w_b->output().shape(), shape_t { b_shape[1] / 2 });
    bc_w_b->name(old_lstm.name() + "/bc_w_b");
    bc_w_b->input().connect(sl_w_b->output());

    // input @ weight + bias
    auto mm_input_w = context.graph.emplace<matmul>(input_bc->output().shape(), tp_w->output().shape(), value_range<float>::full());
    mm_input_w->name(old_lstm.name() + "/mm_input_w");
    mm_input_w->input_a().connect(input_bc->output());
    mm_input_w->input_b().connect(tp_w->output());
    mm_input_w->bias().connect(bc_w_b->output());

    auto bc_mm_input_w = context.graph.emplace<bitcast>(dt_float32, mm_input_w->output().shape(),
        shape_t { input_shape[0], input_shape[1], mm_input_w->output().shape()[1] });
    bc_mm_input_w->name(old_lstm.name() + "/bc_mm_input_w");
    bc_mm_input_w->input().connect(mm_input_w->output());

    // slice r
    auto r_shape = r.output().shape();
    auto sl_r = context.graph.emplace<slice>(r.output().type(), r_shape, axis_t { 0, 0, 0 },
        axis_t { 1, (int32_t)r_shape[1], (int32_t)r_shape[2] });
    sl_r->name(old_lstm.name() + "/sl_r");
    sl_r->input().connect(r.output());

    // reshape r
    auto bc_r = context.graph.emplace<bitcast>(input_type, sl_r->output().shape(), shape_t { r_shape[1], r_shape[2] });
    bc_r->name(old_lstm.name() + "/bc_r");
    bc_r->input().connect(sl_r->output());

    // transpose r
    auto tp_r = context.graph.emplace<transpose>(input_type, bc_r->output().shape(), axis_t { 1, 0 });
    tp_r->name(old_lstm.name() + "/tp_r");
    tp_r->input().connect(bc_r->output());

    // slice r bias
    auto sl_r_b = context.graph.emplace<slice>(input_type, b_shape, axis_t { 0, (int32_t)(b_shape[1] / 2) }, axis_t { 1, (int32_t)b_shape[1] });
    sl_r_b->name(old_lstm.name() + "/sl_r_b");
    sl_r_b->input().connect(b.output());

    // reshape r bias
    auto bc_r_b = context.graph.emplace<bitcast>(input_type, sl_r_b->output().shape(), shape_t { b_shape[1] / 2 });
    bc_r_b->name(old_lstm.name() + "/bc_r_b");
    bc_r_b->input().connect(sl_r_b->output());

    output_connector *tmp_init_h = nullptr, *tmp_init_c = nullptr;
    if (context.matched_nodes.size() == 6)
    {
        auto init_h = &static_cast<constant &>(*context.matched_nodes[4]);
        auto init_c = &static_cast<constant &>(*context.matched_nodes[5]);
        tmp_init_h = &init_h->output();
        tmp_init_c = &init_c->output();
    }
    else
    {
        tmp_init_h = &*context.inputs[4]->connection();
        tmp_init_c = &*context.inputs[5]->connection();
    }

    // slice init_h
    auto h_shape = tmp_init_h->shape();
    auto sl_h = context.graph.emplace<slice>(tmp_init_h->type(), h_shape, axis_t { 0, 0, 0 }, axis_t { 1, (int32_t)h_shape[1], (int32_t)h_shape[2] });
    sl_h->name(old_lstm.name() + "/sl_h");
    sl_h->input().connect(*tmp_init_h);
    auto h_ = &sl_h->output();

    // slice init_c
    auto c_shape = tmp_init_c->shape();
    auto sl_c = context.graph.emplace<slice>(tmp_init_c->type(), c_shape, axis_t { 0, 0, 0 }, axis_t { 1, (int32_t)c_shape[1], (int32_t)c_shape[2] });
    sl_c->name(old_lstm.name() + "/sl_c");
    sl_c->input().connect(*tmp_init_c);
    auto c_ = &sl_c->output();

    std::vector<shape_t> lstm_h_s((size_t)output.shape()[0], h_->shape());

    // h_concat
    auto h_concat = context.graph.emplace<concat>(dt_float32, lstm_h_s, 0);
    h_concat->name(old_lstm.name() + "/h_concat");

    // reshape h_concat
    auto h_concat_shape = h_concat->output().shape();
    auto bc_h_concat = context.graph.emplace<bitcast>(input_type, h_concat_shape, shape_t { h_concat_shape[0], 1, h_concat_shape[1], h_concat_shape[2] });
    bc_h_concat->name(old_lstm.name() + "/bc_h_concat");
    bc_h_concat->input().connect(h_concat->output());

    for (size_t i = 0; i < (size_t)(bc_mm_input_w->output().shape()[0]); i++)
    {
        std::vector<float> cont_data((int)bc_mm_input_w->output().shape()[1], (i == 0) ? (old_lstm.framework() == "caffe" ? 0.f : 1.f) : 1.f);
        auto cont_ = context.graph.emplace<constant>(dt_float32, shape_t { 1, 1 }, cont_data);
        cont_->name(old_lstm.name() + "/cont_" + std::to_string(i));

        // scale
        auto scale_ = context.graph.emplace<binary>(binary_mul, h_->type(), h_->shape(), cont_->output().shape(), value_range<float>::full());
        scale_->name(old_lstm.name() + "/scale_" + std::to_string(i));
        scale_->input_a().connect(*h_);
        scale_->input_b().connect(cont_->output());

        // reshape scale
        auto scale_bc = context.graph.emplace<bitcast>(dt_float32, scale_->output().shape(),
            shape_t { scale_->output().shape()[0] * scale_->output().shape()[1], scale_->output().shape()[2] });
        scale_bc->name(old_lstm.name() + "/scale_bc" + std::to_string(i));
        scale_bc->input().connect(scale_->output());

        // scale @ r + bias
        auto mm_r_scale = context.graph.emplace<matmul>(scale_bc->output().shape(), tp_r->output().shape(), value_range<float>::full());
        mm_r_scale->name(old_lstm.name() + "/mm_r_scale_" + std::to_string(i));
        mm_r_scale->input_a().connect(scale_bc->output());
        mm_r_scale->input_b().connect(tp_r->output());
        mm_r_scale->bias().connect(bc_r_b->output());

        auto mm_r_scale_bc = context.graph.emplace<bitcast>(dt_float32, mm_r_scale->output().shape(),
            shape_t { scale_->output().shape()[0], scale_->output().shape()[1], mm_r_scale->output().shape()[1] });
        mm_r_scale_bc->name(old_lstm.name() + "/mm_r_scale_bc" + std::to_string(i));
        mm_r_scale_bc->input().connect(mm_r_scale->output());

        // slice sl_x_w
        auto sl_x_w = context.graph.emplace<slice>(bc_mm_input_w->output().type(), bc_mm_input_w->output().shape(),
            axis_t { (int32_t)i, 0, 0 }, axis_t { (int32_t)i + 1, (int32_t)bc_mm_input_w->output().shape()[1], (int32_t)bc_mm_input_w->output().shape()[2] });
        sl_x_w->name(old_lstm.name() + "/sl_x_w_" + std::to_string(i));
        sl_x_w->input().connect(bc_mm_input_w->output());

        auto gate_input = context.graph.emplace<binary>(binary_add, sl_x_w->output().type(), sl_x_w->output().shape(), mm_r_scale_bc->output().shape(), value_range<float>::full());
        gate_input->name(old_lstm.name() + "/gate_input_" + std::to_string(i));
        gate_input->input_a().connect(sl_x_w->output());
        gate_input->input_b().connect(mm_r_scale_bc->output());

        // lstm_uint: need [c_, gate_input:[in_sigmoid:[i_t,o_t,f_t,g_t], in_tanh[g_t]]] onnx(default)
        //            need [c_, gate_input:[in_sigmoid:[i_t,f_t,o_t,g_t], in_tanh[g_t]]] caffe
        // transpose the source of gate data
        auto gate_output_ptr = &gate_input->output();
        if (old_lstm.framework() == "caffe")
        {
            auto in_sigmoid = local_sigmoid(gate_output_ptr, context, std::to_string(i));
            auto in_tanh = local_tanh(gate_output_ptr, context, std::to_string(i));

            // slice the in_sidmoid into i_t, f_t, o_t, g_t
            // i_t
            auto i_t = context.graph.emplace<slice>(in_sigmoid->type(), in_sigmoid->shape(),
                axis_t { 0, 0, 0 * (int32_t)c_->shape()[2] }, axis_t { (int32_t)in_sigmoid->shape()[0], (int32_t)in_sigmoid->shape()[1], (int32_t)c_->shape()[2] });
            i_t->name(old_lstm.name() + "/i_t_" + std::to_string(i));
            i_t->input().connect(*in_sigmoid);

            // o_t
            auto o_t = context.graph.emplace<slice>(in_sigmoid->type(), in_sigmoid->shape(),
                axis_t { 0, 0, 2 * (int32_t)c_->shape()[2] }, axis_t { (int32_t)in_sigmoid->shape()[0], (int32_t)in_sigmoid->shape()[1], 3 * (int32_t)c_->shape()[2] });
            o_t->name(old_lstm.name() + "/o_t_" + std::to_string(i));
            o_t->input().connect(*in_sigmoid);

            // f_t
            auto f_t = context.graph.emplace<slice>(in_sigmoid->type(), in_sigmoid->shape(),
                axis_t { 0, 0, 1 * (int32_t)c_->shape()[2] }, axis_t { (int32_t)in_sigmoid->shape()[0], (int32_t)in_sigmoid->shape()[1], 2 * (int32_t)c_->shape()[2] });
            f_t->name(old_lstm.name() + "/f_t_" + std::to_string(i));
            f_t->input().connect(*in_sigmoid);

            // g_t
            auto g_t = context.graph.emplace<slice>(in_tanh->type(), in_tanh->shape(),
                axis_t { 0, 0, 3 * (int32_t)c_->shape()[2] }, axis_t { (int32_t)in_sigmoid->shape()[0], (int32_t)in_sigmoid->shape()[1], 4 * (int32_t)c_->shape()[2] });
            g_t->name(old_lstm.name() + "/g_t_" + std::to_string(i));
            g_t->input().connect(*in_tanh);

            //c_t = cont_ * (f * c_) + (i * g)
            auto f_c_mul = context.graph.emplace<binary>(binary_mul, c_->type(), c_->shape(), f_t->output().shape(), value_range<float>::full());
            f_c_mul->name(old_lstm.name() + "/f_c_mul_" + std::to_string(i));
            f_c_mul->input_a().connect(*c_);
            f_c_mul->input_b().connect(f_t->output());

            auto c_f_c_mul = context.graph.emplace<binary>(binary_mul, cont_->output().type(), cont_->output().shape(), f_c_mul->output().shape(), value_range<float>::full());
            c_f_c_mul->name(old_lstm.name() + "/c_f_c_mul_" + std::to_string(i));
            c_f_c_mul->input_a().connect(cont_->output());
            c_f_c_mul->input_b().connect(f_c_mul->output());

            auto i_g_mul = context.graph.emplace<binary>(binary_mul, i_t->output().type(), i_t->output().shape(), g_t->output().shape(), value_range<float>::full());
            i_g_mul->name(old_lstm.name() + "/i_g_mul_" + std::to_string(i));
            i_g_mul->input_a().connect(i_t->output());
            i_g_mul->input_b().connect(g_t->output());

            auto c_t = context.graph.emplace<binary>(binary_add, c_f_c_mul->output().type(), c_f_c_mul->output().shape(), i_g_mul->output().shape(), value_range<float>::full());
            c_t->name(old_lstm.name() + "/c_t_" + std::to_string(i));
            c_t->input_a().connect(c_f_c_mul->output());
            c_t->input_b().connect(i_g_mul->output());

            //h_t = o_t * tanh(c_t)
            auto tanh_c_t = local_tanh(&c_t->output(), context, std::to_string(i));
            auto h_t = context.graph.emplace<binary>(binary_mul, o_t->output().type(), o_t->output().shape(), tanh_c_t->shape(), value_range<float>::full());
            h_t->name(old_lstm.name() + "/h_t_" + std::to_string(i));
            h_t->input_a().connect(o_t->output());
            h_t->input_b().connect(*tanh_c_t);

            h_concat->input_at(i).connect(h_t->output());

            // update c_, h_
            c_ = &c_t->output();
            h_ = &h_t->output();
        }
        else
        {
            // get gate_output
            auto in_sigmoid = local_sigmoid(gate_output_ptr, context, std::to_string(i));
            auto in_tanh = local_tanh(gate_output_ptr, context, std::to_string(i));

            // slice the in_sidmoid into i_t, o_t, f_t, g_t
            // i_t
            auto i_t = context.graph.emplace<slice>(in_sigmoid->type(), in_sigmoid->shape(),
                axis_t { 0, 0, 0 * (int32_t)c_->shape()[2] }, axis_t { (int32_t)in_sigmoid->shape()[0], (int32_t)in_sigmoid->shape()[1], (int32_t)c_->shape()[2] });
            i_t->name(old_lstm.name() + "/i_t_" + std::to_string(i));
            i_t->input().connect(*in_sigmoid);

            // o_t
            auto o_t = context.graph.emplace<slice>(in_sigmoid->type(), in_sigmoid->shape(),
                axis_t { 0, 0, 1 * (int32_t)c_->shape()[2] }, axis_t { (int32_t)in_sigmoid->shape()[0], (int32_t)in_sigmoid->shape()[1], 2 * (int32_t)c_->shape()[2] });
            o_t->name(old_lstm.name() + "/o_t_" + std::to_string(i));
            o_t->input().connect(*in_sigmoid);

            // f_t
            auto f_t = context.graph.emplace<slice>(in_sigmoid->type(), in_sigmoid->shape(),
                axis_t { 0, 0, 2 * (int32_t)c_->shape()[2] }, axis_t { (int32_t)in_sigmoid->shape()[0], (int32_t)in_sigmoid->shape()[1], 3 * (int32_t)c_->shape()[2] });
            f_t->name(old_lstm.name() + "/f_t_" + std::to_string(i));
            f_t->input().connect(*in_sigmoid);

            // g_t
            auto g_t = context.graph.emplace<slice>(in_tanh->type(), in_tanh->shape(),
                axis_t { 0, 0, 3 * (int32_t)c_->shape()[2] }, axis_t { (int32_t)in_sigmoid->shape()[0], (int32_t)in_sigmoid->shape()[1], 4 * (int32_t)c_->shape()[2] });
            g_t->name(old_lstm.name() + "/g_t_" + std::to_string(i));
            g_t->input().connect(*in_tanh);

            //c_t = cont_ * (f * c_) + (i * g)
            auto f_c_mul = context.graph.emplace<binary>(binary_mul, c_->type(), c_->shape(), f_t->output().shape(), value_range<float>::full());
            f_c_mul->name(old_lstm.name() + "/f_c_mul_" + std::to_string(i));
            f_c_mul->input_a().connect(*c_);
            f_c_mul->input_b().connect(f_t->output());

            auto c_f_c_mul = context.graph.emplace<binary>(binary_mul, cont_->output().type(), cont_->output().shape(), f_c_mul->output().shape(), value_range<float>::full());
            c_f_c_mul->name(old_lstm.name() + "/c_f_c_mul_" + std::to_string(i));
            c_f_c_mul->input_a().connect(cont_->output());
            c_f_c_mul->input_b().connect(f_c_mul->output());

            auto i_g_mul = context.graph.emplace<binary>(binary_mul, i_t->output().type(), i_t->output().shape(), g_t->output().shape(), value_range<float>::full());
            i_g_mul->name(old_lstm.name() + "/i_g_mul_" + std::to_string(i));
            i_g_mul->input_a().connect(i_t->output());
            i_g_mul->input_b().connect(g_t->output());

            auto c_t = context.graph.emplace<binary>(binary_add, c_f_c_mul->output().type(), c_f_c_mul->output().shape(), i_g_mul->output().shape(), value_range<float>::full());
            c_t->name(old_lstm.name() + "/c_t_" + std::to_string(i));
            c_t->input_a().connect(c_f_c_mul->output());
            c_t->input_b().connect(i_g_mul->output());

            //h_t = o_t * tanh(c_t)
            auto tanh_c_t = local_tanh(&c_t->output(), context, std::to_string(i));
            auto h_t = context.graph.emplace<binary>(binary_mul, o_t->output().type(), o_t->output().shape(), tanh_c_t->shape(), value_range<float>::full());
            h_t->name(old_lstm.name() + "/h_t_" + std::to_string(i));
            h_t->input_a().connect(o_t->output());
            h_t->input_b().connect(*tanh_c_t);

            h_concat->input_at(i).connect(h_t->output());

            // update c_, h_
            c_ = &c_t->output();
            h_ = &h_t->output();
        }
    }

    std::vector<output_connector *> result { &bc_h_concat->output(), h_, c_ };
    return result;
}

std::vector<output_connector *> lstm_transform::reverse(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto &old_lstm = static_cast<lstm &>(*context.matched_nodes[0]);
    auto &w = static_cast<constant &>(*context.matched_nodes[1]);
    auto &r = static_cast<constant &>(*context.matched_nodes[2]);
    auto &b = static_cast<constant &>(*context.matched_nodes[3]);
    int num_directions = old_lstm.direction() == kBidirectional ? 2 : 1;

    // reshape input
    auto input_shape = old_lstm.input().shape();
    auto input_type = old_lstm.input().type();
    auto input_bc = context.graph.emplace<bitcast>(input_type, input_shape, shape_t { input_shape[0] * input_shape[1], input_shape[2] });
    input_bc->name(old_lstm.name() + "/input_bc");
    input_bc->input().connect(output);

    // slice weight
    auto w_shape = w.output().shape();
    auto sl_w = context.graph.emplace<slice>(input_type, w_shape, axis_t { num_directions - 1, 0, 0 },
        axis_t { num_directions, (int32_t)w_shape[1], (int32_t)w_shape[2] });
    sl_w->name(old_lstm.name() + "/sl_w");
    sl_w->input().connect(w.output());

    // reshape weight
    auto bc_w = context.graph.emplace<bitcast>(input_type, sl_w->output().shape(), shape_t { w_shape[1], w_shape[2] });
    bc_w->name(old_lstm.name() + "/bc_w");
    bc_w->input().connect(sl_w->output());

    // transpose weight
    auto tp_w = context.graph.emplace<transpose>(input_type, bc_w->output().shape(), axis_t { 1, 0 });
    tp_w->name(old_lstm.name() + "/tp_w");
    tp_w->input().connect(bc_w->output());

    // slice weight bias
    auto b_shape = b.output().shape();
    auto sl_w_b = context.graph.emplace<slice>(input_type, b_shape, axis_t { num_directions - 1, 0 }, axis_t { num_directions, (int32_t)(b_shape[1] / 2) });
    sl_w_b->name(old_lstm.name() + "/sl_w_b");
    sl_w_b->input().connect(b.output());

    // reshape weight bias
    auto bc_w_b = context.graph.emplace<bitcast>(input_type, sl_w_b->output().shape(), shape_t { b_shape[1] / 2 });
    bc_w_b->name(old_lstm.name() + "/bc_w_b");
    bc_w_b->input().connect(sl_w_b->output());

    // input @ weight + bias
    auto mm_input_w = context.graph.emplace<matmul>(input_bc->output().shape(), tp_w->output().shape(), value_range<float>::full());
    mm_input_w->name(old_lstm.name() + "/mm_input_w");
    mm_input_w->input_a().connect(input_bc->output());
    mm_input_w->input_b().connect(tp_w->output());
    mm_input_w->bias().connect(bc_w_b->output());

    auto bc_mm_input_w = context.graph.emplace<bitcast>(dt_float32, mm_input_w->output().shape(),
        shape_t { input_shape[0], input_shape[1], mm_input_w->output().shape()[1] });
    bc_mm_input_w->name(old_lstm.name() + "/bc_mm_input_w");
    bc_mm_input_w->input().connect(mm_input_w->output());

    // slice r
    auto r_shape = r.output().shape();
    auto sl_r = context.graph.emplace<slice>(r.output().type(), r_shape, axis_t { num_directions - 1, 0, 0 },
        axis_t { num_directions, (int32_t)r_shape[1], (int32_t)r_shape[2] });
    sl_r->name(old_lstm.name() + "/sl_r");
    sl_r->input().connect(r.output());

    // reshape r
    auto bc_r = context.graph.emplace<bitcast>(input_type, sl_r->output().shape(), shape_t { r_shape[1], r_shape[2] });
    bc_r->name(old_lstm.name() + "/bc_r");
    bc_r->input().connect(sl_r->output());

    // transpose r
    auto tp_r = context.graph.emplace<transpose>(input_type, bc_r->output().shape(), axis_t { 1, 0 });
    tp_r->name(old_lstm.name() + "/tp_r");
    tp_r->input().connect(bc_r->output());

    // slice r bias
    auto sl_r_b = context.graph.emplace<slice>(input_type, b_shape, axis_t { num_directions - 1, (int32_t)(b_shape[1] / 2) },
        axis_t { num_directions, (int32_t)b_shape[1] });
    sl_r_b->name(old_lstm.name() + "/sl_r_b");
    sl_r_b->input().connect(b.output());

    // reshape r bias
    auto bc_r_b = context.graph.emplace<bitcast>(input_type, sl_r_b->output().shape(), shape_t { b_shape[1] / 2 });
    bc_r_b->name(old_lstm.name() + "/bc_r_b");
    bc_r_b->input().connect(sl_r_b->output());

    output_connector *tmp_init_h = nullptr, *tmp_init_c = nullptr;
    if (context.matched_nodes.size() == 6)
    {
        auto init_h = &static_cast<constant &>(*context.matched_nodes[4]);
        auto init_c = &static_cast<constant &>(*context.matched_nodes[5]);
        tmp_init_h = &init_h->output();
        tmp_init_c = &init_c->output();
    }
    else
    {
        tmp_init_h = &*context.inputs[4]->connection();
        tmp_init_c = &*context.inputs[5]->connection();
    }

    // slice init_h
    auto h_shape = tmp_init_h->shape();
    auto sl_h = context.graph.emplace<slice>(tmp_init_h->type(), h_shape, axis_t { num_directions - 1, 0, 0 }, axis_t { num_directions, (int32_t)h_shape[1], (int32_t)h_shape[2] });
    sl_h->name(old_lstm.name() + "/sl_h");
    sl_h->input().connect(*tmp_init_h);
    auto h_ = &sl_h->output();

    // slice init_c
    auto c_shape = tmp_init_c->shape();
    auto sl_c = context.graph.emplace<slice>(tmp_init_c->type(), c_shape, axis_t { num_directions - 1, 0, 0 }, axis_t { num_directions, (int32_t)c_shape[1], (int32_t)c_shape[2] });
    sl_c->name(old_lstm.name() + "/sl_c");
    sl_c->input().connect(*tmp_init_c);
    auto c_ = &sl_c->output();

    std::vector<shape_t> lstm_h_s((size_t)output.shape()[0], h_->shape());

    // h_concat
    auto h_concat = context.graph.emplace<concat>(dt_float32, lstm_h_s, 0);
    h_concat->name(old_lstm.name() + "/h_concat");

    // reshape h_concat
    auto h_concat_shape = h_concat->output().shape();
    auto bc_h_concat = context.graph.emplace<bitcast>(input_type, h_concat_shape, shape_t { h_concat_shape[0], 1, h_concat_shape[1], h_concat_shape[2] });
    bc_h_concat->name(old_lstm.name() + "/bc_h_concat");
    bc_h_concat->input().connect(h_concat->output());

    for (int i = (int)(bc_mm_input_w->output().shape()[0]) - 1; i >= 0; i--)
    {
        std::vector<float> cont_data((int)bc_mm_input_w->output().shape()[1], (i == bc_mm_input_w->output().shape()[0] - 1) ? (old_lstm.framework() == "caffe" ? 0.f : 1.f) : 1.f);
        auto cont_ = context.graph.emplace<constant>(dt_float32, shape_t { 1, 1 }, cont_data);
        cont_->name(old_lstm.name() + "/cont_" + std::to_string(i));

        // scale
        auto scale_ = context.graph.emplace<binary>(binary_mul, h_->type(), h_->shape(), cont_->output().shape(), value_range<float>::full());
        scale_->name(old_lstm.name() + "/scale_" + std::to_string(i));
        scale_->input_a().connect(*h_);
        scale_->input_b().connect(cont_->output());

        // reshape scale
        auto scale_bc = context.graph.emplace<bitcast>(dt_float32, scale_->output().shape(),
            shape_t { scale_->output().shape()[0] * scale_->output().shape()[1], scale_->output().shape()[2] });
        scale_bc->name(old_lstm.name() + "/scale_bc" + std::to_string(i));
        scale_bc->input().connect(scale_->output());

        // scale @ r + bias
        auto mm_r_scale = context.graph.emplace<matmul>(scale_bc->output().shape(), tp_r->output().shape(), value_range<float>::full());
        mm_r_scale->name(old_lstm.name() + "/mm_r_scale_" + std::to_string(i));
        mm_r_scale->input_a().connect(scale_bc->output());
        mm_r_scale->input_b().connect(tp_r->output());
        mm_r_scale->bias().connect(bc_r_b->output());

        auto mm_r_scale_bc = context.graph.emplace<bitcast>(dt_float32, mm_r_scale->output().shape(),
            shape_t { scale_->output().shape()[0], scale_->output().shape()[1], mm_r_scale->output().shape()[1] });
        mm_r_scale_bc->name(old_lstm.name() + "/mm_r_scale_bc" + std::to_string(i));
        mm_r_scale_bc->input().connect(mm_r_scale->output());

        // slice sl_x_w
        auto sl_x_w = context.graph.emplace<slice>(bc_mm_input_w->output().type(), bc_mm_input_w->output().shape(),
            axis_t { (int32_t)i, 0, 0 }, axis_t { (int32_t)i + 1, (int32_t)bc_mm_input_w->output().shape()[1], (int32_t)bc_mm_input_w->output().shape()[2] });
        sl_x_w->name(old_lstm.name() + "/sl_x_w_" + std::to_string(i));
        sl_x_w->input().connect(bc_mm_input_w->output());

        auto gate_input = context.graph.emplace<binary>(binary_add, sl_x_w->output().type(), sl_x_w->output().shape(), mm_r_scale_bc->output().shape(), value_range<float>::full());
        gate_input->name(old_lstm.name() + "/gate_input_" + std::to_string(i));
        gate_input->input_a().connect(sl_x_w->output());
        gate_input->input_b().connect(mm_r_scale_bc->output());

        // lstm_uint: need [c_, gate_input:[in_sigmoid:[i_t,o_t,f_t,g_t], in_tanh[g_t]]] onnx(default)
        //            need [c_, gate_input:[in_sigmoid:[i_t,f_t,o_t,g_t], in_tanh[g_t]]] caffe
        // transpose the source of gate data
        auto gate_output_ptr = &gate_input->output();
        if (old_lstm.framework() == "caffe")
        {
            auto in_sigmoid = local_sigmoid(gate_output_ptr, context, std::to_string(i));
            auto in_tanh = local_tanh(gate_output_ptr, context, std::to_string(i));

            // slice the in_sidmoid into i_t, f_t, o_t, g_t
            // i_t
            auto i_t = context.graph.emplace<slice>(in_sigmoid->type(), in_sigmoid->shape(),
                axis_t { 0, 0, 0 * (int32_t)c_->shape()[2] }, axis_t { (int32_t)in_sigmoid->shape()[0], (int32_t)in_sigmoid->shape()[1], (int32_t)c_->shape()[2] });
            i_t->name(old_lstm.name() + "/i_t_" + std::to_string(i));
            i_t->input().connect(*in_sigmoid);

            // o_t
            auto o_t = context.graph.emplace<slice>(in_sigmoid->type(), in_sigmoid->shape(),
                axis_t { 0, 0, 2 * (int32_t)c_->shape()[2] }, axis_t { (int32_t)in_sigmoid->shape()[0], (int32_t)in_sigmoid->shape()[1], 3 * (int32_t)c_->shape()[2] });
            o_t->name(old_lstm.name() + "/o_t_" + std::to_string(i));
            o_t->input().connect(*in_sigmoid);

            // f_t
            auto f_t = context.graph.emplace<slice>(in_sigmoid->type(), in_sigmoid->shape(),
                axis_t { 0, 0, 1 * (int32_t)c_->shape()[2] }, axis_t { (int32_t)in_sigmoid->shape()[0], (int32_t)in_sigmoid->shape()[1], 2 * (int32_t)c_->shape()[2] });
            f_t->name(old_lstm.name() + "/f_t_" + std::to_string(i));
            f_t->input().connect(*in_sigmoid);

            // g_t
            auto g_t = context.graph.emplace<slice>(in_tanh->type(), in_tanh->shape(),
                axis_t { 0, 0, 3 * (int32_t)c_->shape()[2] }, axis_t { (int32_t)in_sigmoid->shape()[0], (int32_t)in_sigmoid->shape()[1], 4 * (int32_t)c_->shape()[2] });
            g_t->name(old_lstm.name() + "/g_t_" + std::to_string(i));
            g_t->input().connect(*in_tanh);

            //c_t = cont_ * (f * c_) + (i * g)
            auto f_c_mul = context.graph.emplace<binary>(binary_mul, c_->type(), c_->shape(), f_t->output().shape(), value_range<float>::full());
            f_c_mul->name(old_lstm.name() + "/f_c_mul_" + std::to_string(i));
            f_c_mul->input_a().connect(*c_);
            f_c_mul->input_b().connect(f_t->output());

            auto c_f_c_mul = context.graph.emplace<binary>(binary_mul, cont_->output().type(), cont_->output().shape(), f_c_mul->output().shape(), value_range<float>::full());
            c_f_c_mul->name(old_lstm.name() + "/c_f_c_mul_" + std::to_string(i));
            c_f_c_mul->input_a().connect(cont_->output());
            c_f_c_mul->input_b().connect(f_c_mul->output());

            auto i_g_mul = context.graph.emplace<binary>(binary_mul, i_t->output().type(), i_t->output().shape(), g_t->output().shape(), value_range<float>::full());
            i_g_mul->name(old_lstm.name() + "/i_g_mul_" + std::to_string(i));
            i_g_mul->input_a().connect(i_t->output());
            i_g_mul->input_b().connect(g_t->output());

            auto c_t = context.graph.emplace<binary>(binary_add, c_f_c_mul->output().type(), c_f_c_mul->output().shape(), i_g_mul->output().shape(), value_range<float>::full());
            c_t->name(old_lstm.name() + "/c_t_" + std::to_string(i));
            c_t->input_a().connect(c_f_c_mul->output());
            c_t->input_b().connect(i_g_mul->output());

            //h_t = o_t * tanh(c_t)
            auto tanh_c_t = local_tanh(&c_t->output(), context, std::to_string(i));
            auto h_t = context.graph.emplace<binary>(binary_mul, o_t->output().type(), o_t->output().shape(), tanh_c_t->shape(), value_range<float>::full());
            h_t->name(old_lstm.name() + "/h_t_" + std::to_string(i));
            h_t->input_a().connect(o_t->output());
            h_t->input_b().connect(*tanh_c_t);

            h_concat->input_at(i).connect(h_t->output());

            // update c_, h_
            c_ = &c_t->output();
            h_ = &h_t->output();
        }
        else
        {
            // get gate_output
            auto in_sigmoid = local_sigmoid(gate_output_ptr, context, std::to_string(i));
            auto in_tanh = local_tanh(gate_output_ptr, context, std::to_string(i));

            // slice the in_sidmoid into i_t, o_t, f_t, g_t
            // i_t
            auto i_t = context.graph.emplace<slice>(in_sigmoid->type(), in_sigmoid->shape(),
                axis_t { 0, 0, 0 * (int32_t)c_->shape()[2] }, axis_t { (int32_t)in_sigmoid->shape()[0], (int32_t)in_sigmoid->shape()[1], (int32_t)c_->shape()[2] });
            i_t->name(old_lstm.name() + "/i_t_" + std::to_string(i));
            i_t->input().connect(*in_sigmoid);

            // o_t
            auto o_t = context.graph.emplace<slice>(in_sigmoid->type(), in_sigmoid->shape(),
                axis_t { 0, 0, 1 * (int32_t)c_->shape()[2] }, axis_t { (int32_t)in_sigmoid->shape()[0], (int32_t)in_sigmoid->shape()[1], 2 * (int32_t)c_->shape()[2] });
            o_t->name(old_lstm.name() + "/o_t_" + std::to_string(i));
            o_t->input().connect(*in_sigmoid);

            // f_t
            auto f_t = context.graph.emplace<slice>(in_sigmoid->type(), in_sigmoid->shape(),
                axis_t { 0, 0, 2 * (int32_t)c_->shape()[2] }, axis_t { (int32_t)in_sigmoid->shape()[0], (int32_t)in_sigmoid->shape()[1], 3 * (int32_t)c_->shape()[2] });
            f_t->name(old_lstm.name() + "/f_t_" + std::to_string(i));
            f_t->input().connect(*in_sigmoid);

            // g_t
            auto g_t = context.graph.emplace<slice>(in_tanh->type(), in_tanh->shape(),
                axis_t { 0, 0, 3 * (int32_t)c_->shape()[2] }, axis_t { (int32_t)in_sigmoid->shape()[0], (int32_t)in_sigmoid->shape()[1], 4 * (int32_t)c_->shape()[2] });
            g_t->name(old_lstm.name() + "/g_t_" + std::to_string(i));
            g_t->input().connect(*in_tanh);

            //c_t = cont_ * (f * c_) + (i * g)
            auto f_c_mul = context.graph.emplace<binary>(binary_mul, c_->type(), c_->shape(), f_t->output().shape(), value_range<float>::full());
            f_c_mul->name(old_lstm.name() + "/f_c_mul_" + std::to_string(i));
            f_c_mul->input_a().connect(*c_);
            f_c_mul->input_b().connect(f_t->output());

            auto c_f_c_mul = context.graph.emplace<binary>(binary_mul, cont_->output().type(), cont_->output().shape(), f_c_mul->output().shape(), value_range<float>::full());
            c_f_c_mul->name(old_lstm.name() + "/c_f_c_mul_" + std::to_string(i));
            c_f_c_mul->input_a().connect(cont_->output());
            c_f_c_mul->input_b().connect(f_c_mul->output());

            auto i_g_mul = context.graph.emplace<binary>(binary_mul, i_t->output().type(), i_t->output().shape(), g_t->output().shape(), value_range<float>::full());
            i_g_mul->name(old_lstm.name() + "/i_g_mul_" + std::to_string(i));
            i_g_mul->input_a().connect(i_t->output());
            i_g_mul->input_b().connect(g_t->output());

            auto c_t = context.graph.emplace<binary>(binary_add, c_f_c_mul->output().type(), c_f_c_mul->output().shape(), i_g_mul->output().shape(), value_range<float>::full());
            c_t->name(old_lstm.name() + "/c_t_" + std::to_string(i));
            c_t->input_a().connect(c_f_c_mul->output());
            c_t->input_b().connect(i_g_mul->output());

            //h_t = o_t * tanh(c_t)
            auto tanh_c_t = local_tanh(&c_t->output(), context, std::to_string(i));
            auto h_t = context.graph.emplace<binary>(binary_mul, o_t->output().type(), o_t->output().shape(), tanh_c_t->shape(), value_range<float>::full());
            h_t->name(old_lstm.name() + "/h_t_" + std::to_string(i));
            h_t->input_a().connect(o_t->output());
            h_t->input_b().connect(*tanh_c_t);

            h_concat->input_at(i).connect(h_t->output());

            // update c_, h_
            c_ = &c_t->output();
            h_ = &h_t->output();
        }
    }

    std::vector<output_connector *> result { &bc_h_concat->output(), h_, c_ };
    return result;
}
