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
#include <nncase/ir/ops/binary.h>
#include <nncase/ir/ops/bitcast.h>
#include <nncase/ir/ops/concat.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/sigmoid.h>
#include <nncase/ir/ops/transpose.h>
#include <nncase/ir/visitor.h>
#include <nncase/transforms/neutral/squeeze_dims.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

shape_t squeeze_shape(shape_t old_shape)
{
    shape_t new_shape { 1, 1, 1, 1 };
    for (int i = old_shape.size() - 1, k = 3; i >= 0; i--)
    {
        new_shape[k] *= old_shape[i];
        if (k > 0)
            k--;
    }
    return new_shape;
}

auto squeeze_binary_shape(shape_t old_a_shape, shape_t old_b_shape)
{
    auto a_size = old_a_shape.size();
    auto b_size = old_b_shape.size();
    auto squeeze_times = std::max(a_size > 4 ? a_size - 4 : 0, b_size > 4 ? b_size - 4 : 0);
    if (squeeze_times <= 0)
        return std::tuple(false, old_a_shape, old_b_shape);
    shape_t new_a_shape, new_b_shape;

    if (a_size == b_size)
    {
        /*
        1.  a.shape == b.shape
        2.  a.shape : [s1, s2, s3, s4, s5]
            b.shape :  [1, 1, s3, 1, 1] ||[1, 1, 1, s4, 1]  ||...
        */
        // 1.   a.shape == b.shape
        if (old_a_shape == old_b_shape)
        {
            new_a_shape = squeeze_shape(old_a_shape);
            new_b_shape = squeeze_shape(old_b_shape);
        }
        // 2.   a.shape : [s1, s2, s3, s4, s5]
        //      b.shape :  [1, 1, s3, 1, 1] ||[1, 1, 1, s4, 1]  ||...
        else
        {
            new_a_shape = old_a_shape;
            new_b_shape = old_b_shape;

            // inquiry which dim can be fold
            std::vector<bool> can_fold_index_list(a_size, true);
            std::vector<std::tuple<size_t, size_t>> fold_index_couple;
            for (size_t i = 0; i < a_size; i++)
            {
                if (old_a_shape[i] == old_b_shape[i])
                    can_fold_index_list[i] = false;
            }
            for (size_t i = a_size - 1; i > 0; i--)
            {
                if (can_fold_index_list[i] && can_fold_index_list[i - 1])
                    fold_index_couple.emplace_back(std::make_tuple(i - 1, i));
            }

            while (squeeze_times && !fold_index_couple.empty())
            {
                auto it = fold_index_couple.back();
                auto front = std::get<0>(it);
                auto back = std::get<1>(it);
                new_a_shape[front] *= new_a_shape[back];
                new_b_shape[front] *= new_b_shape[back];
                new_a_shape.erase(std::begin(new_a_shape) + back);
                new_b_shape.erase(std::begin(new_b_shape) + back);
                fold_index_couple.pop_back();
                squeeze_times--;
            }

            if (new_a_shape.size() > 4)
            {
                // remove shape.front() == 1 || shape.back() == 1
                if (new_a_shape.front() == 1 && new_b_shape.front() == 1)
                {
                    new_a_shape.erase(std::begin(new_a_shape));
                    new_b_shape.erase(std::begin(new_b_shape));
                }
                else if (new_a_shape.back() == 1 && new_b_shape.back() == 1)
                {
                    new_a_shape.erase(std::end(new_a_shape) - 1);
                    new_b_shape.erase(std::end(new_b_shape) - 1);
                }
            }

            new_a_shape.shrink_to_fit();
            new_b_shape.shrink_to_fit();
            if (new_a_shape.size() > 4)
                return std::make_tuple(false, new_a_shape, new_b_shape);
        }
    }
    else
    {
        if (a_size != 1)
            new_a_shape = squeeze_shape(old_a_shape);
        else
            new_a_shape = old_a_shape;
        if (b_size != 1)
            new_b_shape = squeeze_shape(old_b_shape);
        else
            new_b_shape = old_b_shape;
    }
    return std::make_tuple(true, new_a_shape, new_b_shape);
}

auto squeeze_transpose_shape(shape_t old_shape, axis_t old_axis)
{
    if (old_shape.size() <= 4)
        return std::make_tuple(false, old_axis, old_shape);

    axis_t new_axis = old_axis;
    shape_t new_shape = old_shape;
    int squeeze_times = old_shape.size() - 4;

    std::vector<std::tuple<size_t, size_t>> fold_index_couple;
    for (size_t i = old_shape.size() - 1; i > 0; i--)
    {
        if (old_axis[i - 1] + 1 == old_axis[i])
            fold_index_couple.emplace_back(std::make_tuple(i - 1, i));
    }
    if (fold_index_couple.size() < squeeze_times)
        return std::make_tuple(false, new_axis, new_shape);

    while (squeeze_times && !fold_index_couple.empty())
    {
        auto it = fold_index_couple.back();
        auto front = std::get<0>(it);
        auto back = std::get<1>(it);
        new_shape[front] *= new_shape[back];
        new_shape.erase(std::begin(new_shape) + back);
        new_axis.erase(std::begin(new_axis) + back);
        fold_index_couple.pop_back();
        squeeze_times--;
    }

    // fix axis
    for (int i = 0, j = 0; j < 4; i++)
    {
        auto find_index = std::find(new_axis.begin(), new_axis.end(), i);
        if (find_index != new_axis.end())
        {
            *find_index = j;
            j++;
        }
    }

    return std::make_tuple(true, new_axis, new_shape);
}

auto squeeze_concat_shape(std::vector<shape_t> &old_shape, int concat_axis)
{
    int new_axis = 0;
    for (int index = 0; index < old_shape.size(); index++)
    {
        auto tmp_axis = concat_axis;
        auto squeeze_times = old_shape[index].size() - 4;
        shape_t new_shape { 1, 1, 1, 1 };
        for (int i = 0, j = 0; i < 4; i++, j++)
        {
            if (concat_axis > old_shape[index].size() - 4 - 1 && squeeze_times != 0)
            {
                new_shape[i] = old_shape[index][j] * old_shape[index][j + 1];
                squeeze_times--;
                j++;
                tmp_axis--;
            }
            else
            {
                new_shape[i] = old_shape[index][j];
            }
        }
        old_shape[index] = new_shape;
        new_axis = tmp_axis;
    }

    return new_axis;
}

bool check_op(node_opcode op)
{
    if (op == op_binary || op == op_sigmoid || op == op_transpose || op == op_concat)
        return true;
    return false;
}

bool squeeze_dims_transform::on_try_match(node &node, transform_context &context)
{
    if (check_op(node.runtime_opcode()))
    {
        bool need_squeeze = false;
        for (auto &it : node.inputs())
        {
            if (need_squeeze || it->shape().size() > 4)
            {
                need_squeeze = true;
                context.inputs.emplace_back(it);
            }
        }

        // double check all input emplaced if need squeeze
        if (need_squeeze)
        {
            for (auto &it : node.inputs())
            {
                if (std::find(context.inputs.begin(), context.inputs.end(), it) == context.inputs.end())
                    context.inputs.emplace_back(it);
            }
        }

        for (auto &it : node.outputs())
        {
            if (need_squeeze || it->shape().size() > 4)
            {
                need_squeeze = true;
                context.outputs.emplace_back(it);
            }
        }
        if (need_squeeze)
        {
            context.matched_nodes.emplace_back(&node);
            bool can_squeeze = true;
            NNCASE_UNUSED shape_t a_shape, b_shape;
            NNCASE_UNUSED axis_t new_axis;
            if (node.runtime_opcode() == op_binary)
                std::tie(can_squeeze, a_shape, b_shape) = squeeze_binary_shape(context.inputs[0]->shape(), context.inputs[1]->shape());
            else if (node.runtime_opcode() == op_transpose)
                std::tie(can_squeeze, new_axis, b_shape) = squeeze_transpose_shape(node_cast<transpose>(node)->input().shape(), node_cast<transpose>(node)->perm());

            return can_squeeze;
        }
    }

    return false;
}

void squeeze_dims_transform::process(transform_context &context)
{
    if (context.matched_nodes[0]->runtime_opcode() == op_binary)
    {
        auto &output_a = *context.inputs[0]->connection();
        auto &output_b = *context.inputs[1]->connection();
        auto inputs = context.outputs[0]->connections();
        auto &old_binary = static_cast<binary &>(*context.matched_nodes[0]);

        bitcast *in_a_bitc, *in_b_bitc, *out_bitc;
        auto [_, new_a_shape, new_b_shape] = squeeze_binary_shape(output_a.shape(), output_b.shape());
        if (output_a.shape().size() > 4)
            in_a_bitc = context.graph.emplace<bitcast>(output_a.type(), output_a.shape(), new_a_shape);
        else
            in_a_bitc = context.graph.emplace<bitcast>(output_a.type(), output_a.shape(), output_a.shape());

        if (output_b.shape().size() > 4)
            in_b_bitc = context.graph.emplace<bitcast>(output_b.type(), output_b.shape(), new_b_shape);
        else
            in_b_bitc = context.graph.emplace<bitcast>(output_b.type(), output_b.shape(), output_b.shape());

        auto new_binary = context.graph.emplace<binary>(old_binary.binary_op(), in_a_bitc->output().type(), in_a_bitc->output().shape(), in_b_bitc->output().shape(),
            old_binary.fused_activation());
        if (old_binary.output_at(0).shape().size() > 4)
            out_bitc = context.graph.emplace<bitcast>(new_binary->output().type(), new_binary->output().shape(), old_binary.output_at(0).shape());
        else
            out_bitc = context.graph.emplace<bitcast>(new_binary->output().type(), new_binary->output().shape(), new_binary->output().shape());

        in_a_bitc->name(old_binary.name() + "_in_a_bitc");
        in_b_bitc->name(old_binary.name() + "_in_b_bitc");
        new_binary->name(old_binary.name());
        out_bitc->name(old_binary.name() + "_out_bitc");

        new_binary->input_a().connect(in_a_bitc->output());
        new_binary->input_b().connect(in_b_bitc->output());
        out_bitc->input().connect(new_binary->output());

        in_a_bitc->input().connect(output_a);
        in_b_bitc->input().connect(output_b);
        for (auto &in : dup(inputs))
            in->connect(out_bitc->output());
    }
    else if (context.matched_nodes[0]->runtime_opcode() == op_sigmoid)
    {
        auto &output = *context.inputs[0]->connection();
        auto inputs = context.outputs[0]->connections();
        auto &old_sigmoid = static_cast<sigmoid &>(*context.matched_nodes[0]);

        bitcast *in_bitc, *out_bitc;
        if (output.shape().size() > 4)
            in_bitc = context.graph.emplace<bitcast>(output.type(), output.shape(), squeeze_shape(output.shape()));
        else
            in_bitc = context.graph.emplace<bitcast>(output.type(), output.shape(), output.shape());

        auto new_sigmoid = context.graph.emplace<sigmoid>(in_bitc->output().type(), in_bitc->output().shape());
        if (old_sigmoid.output_at(0).shape().size() > 4)
            out_bitc = context.graph.emplace<bitcast>(new_sigmoid->output().type(), new_sigmoid->output().shape(), old_sigmoid.output_at(0).shape());
        else
            out_bitc = context.graph.emplace<bitcast>(new_sigmoid->output().type(), new_sigmoid->output().shape(), new_sigmoid->output().shape());

        in_bitc->name(old_sigmoid.name() + "_in_bitc");
        new_sigmoid->name(old_sigmoid.name());
        out_bitc->name(old_sigmoid.name() + "_out_bitc");

        new_sigmoid->input().connect(in_bitc->output());
        out_bitc->input().connect(new_sigmoid->output());

        in_bitc->input().connect(output);
        for (auto &in : dup(inputs))
            in->connect(out_bitc->output());
    }
    else if (context.matched_nodes[0]->runtime_opcode() == op_transpose)
    {
        auto &output = *context.inputs[0]->connection();
        auto inputs = context.outputs[0]->connections();
        auto &old_transpose = static_cast<transpose &>(*context.matched_nodes[0]);

        auto [_, new_axis, new_shape] = squeeze_transpose_shape(output.shape(), old_transpose.perm());

        bitcast *in_bitc, *out_bitc;
        if (output.shape().size() > 4)
            in_bitc = context.graph.emplace<bitcast>(output.type(), output.shape(), new_shape);
        else
            in_bitc = context.graph.emplace<bitcast>(output.type(), output.shape(), output.shape());

        auto new_transpose = context.graph.emplace<transpose>(in_bitc->output().type(), in_bitc->output().shape(), new_axis);
        if (old_transpose.output_at(0).shape().size() > 4)
            out_bitc = context.graph.emplace<bitcast>(new_transpose->output().type(), new_transpose->output().shape(), old_transpose.output_at(0).shape());
        else
            out_bitc = context.graph.emplace<bitcast>(new_transpose->output().type(), new_transpose->output().shape(), new_transpose->output().shape());

        in_bitc->name(old_transpose.name() + "_in_bitc");
        new_transpose->name(old_transpose.name());
        out_bitc->name(old_transpose.name() + "_out_bitc");

        new_transpose->input().connect(in_bitc->output());
        out_bitc->input().connect(new_transpose->output());

        in_bitc->input().connect(output);
        for (auto &in : dup(inputs))
            in->connect(out_bitc->output());
    }
    else if (context.matched_nodes[0]->runtime_opcode() == op_concat)
    {
        auto inputs = context.outputs[0]->connections();
        auto &old_concat = static_cast<concat &>(*context.matched_nodes[0]);

        std::vector<shape_t> concat_shape;
        std::vector<output_connector *> concat_inputs;

        for (auto &it : context.inputs)
        {
            concat_shape.emplace_back(it->shape());
        }
        auto new_axis = squeeze_concat_shape(concat_shape, old_concat.axis());
        auto new_concat = context.graph.emplace<concat>(old_concat.output().type(), concat_shape, new_axis);
        new_concat->name(old_concat.name());

        for (size_t i = 0; i < context.inputs.size(); i++)
        {
            auto in_bitc = context.graph.emplace<bitcast>(context.inputs[i]->connection()->type(), context.inputs[i]->connection()->shape(), concat_shape[i]);

            in_bitc->input().connect(*context.inputs[i]->connection());
            in_bitc->name(old_concat.name() + "_in_bitc_" + std::to_string(i));
            new_concat->input_at(i).connect(in_bitc->output());
        }
        bitcast *out_bitc;
        if (old_concat.output_at(0).shape().size() > 4)
            out_bitc = context.graph.emplace<bitcast>(new_concat->output().type(), new_concat->output().shape(), old_concat.output_at(0).shape());
        else
            out_bitc = context.graph.emplace<bitcast>(new_concat->output().type(), new_concat->output().shape(), new_concat->output().shape());

        out_bitc->name(old_concat.name() + "_out_bitc");

        out_bitc->input().connect(new_concat->output());

        for (auto &in : dup(inputs))
            in->connect(out_bitc->output());
    }
}
