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
#include <ir/ops/binary.h>
#include <ir/ops/constant.h>
#include <ir/ops/k210/fake_kpu_conv2d.h>
#include <ir/ops/k210/fake_piecewise_linear.h>
#include <ir/visitor.h>
#include <transforms/k210/fake_piecewise_linear.h>
#include <unordered_set>
#include <vector>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::k210;
using namespace nncase::runtime::k210;
using namespace nncase::transforms;
using namespace nncase::transforms::k210;

namespace
{
static std::unordered_set<binary_op_t> allowd_ops_ { binary_add, binary_mul, binary_min, binary_max };
static std::unordered_set<binary_op_t> allowd_combine_ops_ { binary_min, binary_max };
static std::unordered_set<_kpu_pool_type> allowd_pools_ { kpu_pool_bypass, kpu_pool_left_top_2_s2, kpu_pool_left_top_4_s4, kpu_pool_right_top_2_s2 };

struct line
{
    float start_x;
    float end_x;
    float k;
    float b;
};

struct cross_point
{
    float x;
    float mul_a;
    float add_a;
    float mul_b;
    float add_b;
};

xt::svector<line> to_lines(const xt::svector<piecewise_linear_segment> &segs)
{
    xt::svector<line> lines;
    for (size_t i = 0; i < segs.size(); i++)
    {
        auto end = i == segs.size() - 1 ? std::numeric_limits<float>::max() : segs[i + 1].start;
        lines.push_back({ segs[i].start, end, segs[i].mul, segs[i].add });
    }

    return lines;
}

xt::svector<piecewise_linear_segment> combine_segments(const xt::svector<piecewise_linear_segment> &lhs, const xt::svector<piecewise_linear_segment> &rhs, binary_op_t op)
{
    auto lines_a = to_lines(lhs);
    auto lines_b = to_lines(rhs);

    xt::svector<cross_point> cross;
    for (auto &line_a : lines_a)
    {
        for (auto &line_b : lines_b)
        {
            float x;
            auto dk = line_a.k - line_b.k;
            if (dk)
                x = (line_b.b - line_a.b) / dk;
            else
                x = std::max(line_a.start_x, line_b.start_x);

            if (x >= line_a.end_x || x >= line_b.end_x
                || x < line_a.start_x || x < line_b.start_x)
                continue;
            else
                cross.push_back({ x, line_a.k, line_a.b, line_b.k, line_b.b });
        }
    }

    std::sort(cross.begin(), cross.end(), [](auto &a, auto &b) { return a.x < b.x; });
    xt::svector<piecewise_linear_segment> segs;
    if (op == binary_max)
    {
        if (cross[0].mul_a < cross[0].mul_b)
            segs.push_back({ std::numeric_limits<float>::lowest(), cross[0].mul_a, cross[0].add_a });
        else
            segs.push_back({ std::numeric_limits<float>::lowest(), cross[0].mul_b, cross[0].add_b });

        for (auto &cp : cross)
        {
            if (cp.mul_a > cp.mul_b)
                segs.push_back({ cp.x, cross[0].mul_a, cross[0].add_a });
            else
                segs.push_back({ cp.x, cross[0].mul_b, cross[0].add_b });
        }
    }
    else
    {
        throw std::runtime_error("Unsupported binary op");
    }

    return segs;
}

xt::svector<piecewise_linear_segment> seg_segments(const xt::svector<piecewise_linear_segment> &lhs, const xt::svector<piecewise_linear_segment> &rhs)
{
    auto lines_a = to_lines(lhs);
    auto lines_b = to_lines(rhs);

    xt::svector<piecewise_linear_segment> segs;
    for (auto &line_a : lines_a)
    {
        auto f_start = line_a.start_x * line_a.k + line_a.b;
        auto f_end = line_a.end_x * line_a.k + line_a.b;
        auto [f_min, f_max] = std::minmax(f_start, f_end);

        for (auto &line_b : lines_b)
        {
            if (line_b.start_x > f_max)
                break;
            if (line_b.end_x > f_min)
            {
                auto start = std::max(f_min, line_b.start_x);
                auto mul = line_a.k * line_b.k;
                auto add = line_a.b * line_b.k + line_b.b;
                segs.push_back({ start, mul, add });
            }
        }
    }

    return segs;
}
}

bool binary_to_fake_piecewise_linear_transform::on_try_match(node &node, transform_context &context)
{
    if (node.runtime_opcode() == op_k210_fake_kpu_conv2d)
    {
        auto &conv = static_cast<fake_kpu_conv2d &>(node);
        if (allowd_pools_.find(conv.pool_type()) == allowd_pools_.end())
            return false;

        if (auto bin = try_get_direct_child<binary>(node))
        {
            if (allowd_ops_.find(bin->binary_op()) == allowd_ops_.end())
                return false;

            constant *con;
            if (&bin->input_a().connection()->owner() == &node
                && bin->input_b().connection()->owner().runtime_opcode() == op_constant)
            {
                con = static_cast<constant *>(&bin->input_b().connection()->owner());
                context.inputs.emplace_back(&bin->input_a());
            }
            else if (&bin->input_b().connection()->owner() == &node
                && bin->input_a().connection()->owner().runtime_opcode() == op_constant)
            {
                con = static_cast<constant *>(&bin->input_a().connection()->owner());
                context.inputs.emplace_back(&bin->input_b());
            }
            else
            {
                return false;
            }

            if (con->output().shape() != shape_t { 1 })
                return false;

            context.outputs.emplace_back(&bin->output());

            context.matched_nodes.emplace_back(bin);
            context.matched_nodes.emplace_back(con);
            return true;
        }
    }

    return false;
}

void binary_to_fake_piecewise_linear_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();
    auto &old_bin = static_cast<binary &>(*context.matched_nodes[0]);
    auto param = *reinterpret_cast<const float *>(static_cast<constant &>(*context.matched_nodes[1]).data().data());

    xt::svector<piecewise_linear_segment> segs;
    switch (old_bin.binary_op())
    {
    case binary_add:
        segs.push_back({ std::numeric_limits<float>::lowest(), 1.f, param });
        break;
    case binary_mul:
        segs.push_back({ std::numeric_limits<float>::lowest(), param, 0.f });
        break;
    case binary_min:
        segs.push_back({ std::numeric_limits<float>::lowest(), 1.f, 0.f });
        segs.push_back({ param, 0.f, param });
        break;
    case binary_max:
        segs.push_back({ std::numeric_limits<float>::lowest(), 0.f, param });
        segs.push_back({ param, 1.f, 0.f });
        break;
    default:
        throw std::runtime_error("Invalid binary op");
    }

    auto piece = context.graph.emplace<fake_piecewise_linear>(output.shape(), segs);

    piece->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(piece->output());
}

bool fake_piecewise_linear_binary_transform::on_try_match(node &node, transform_context &context)
{
    if (node.runtime_opcode() == op_k210_fake_piecewise_linear)
    {
        auto &piece = static_cast<fake_piecewise_linear &>(node);
        if (auto bin = try_get_direct_child<binary>(node))
        {
            if (allowd_combine_ops_.find(bin->binary_op()) == allowd_combine_ops_.end())
                return false;

            if (&bin->input_a().connection()->owner() == &piece.input().connection()->owner())
                context.inputs.emplace_back(&bin->input_a());
            else if (&bin->input_b().connection()->owner() == &piece.input().connection()->owner())
                context.inputs.emplace_back(&bin->input_b());
            else
                return false;

            context.inputs.emplace_back(&piece.input());
            context.outputs.emplace_back(&bin->output());

            context.matched_nodes.emplace_back(&piece);
            context.matched_nodes.emplace_back(bin);
            return true;
        }
    }

    return false;
}

void fake_piecewise_linear_binary_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();
    auto &old_piece = static_cast<fake_piecewise_linear &>(*context.matched_nodes[0]);
    auto &old_bin = static_cast<binary &>(*context.matched_nodes[1]);

    xt::svector<piecewise_linear_segment> l_segs;
    l_segs.push_back({ std::numeric_limits<float>::lowest(), 1.f, 0.f });

    auto r_segs = old_piece.segments();
    auto segs = combine_segments(l_segs, r_segs, old_bin.binary_op());

    auto piece = context.graph.emplace<fake_piecewise_linear>(output.shape(), segs);

    piece->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(piece->output());
}

bool fuse_fake_kpu_conv2d_piecewise_linear_transform::on_try_match(node &node, transform_context &context)
{
    if (node.runtime_opcode() == op_k210_fake_kpu_conv2d)
    {
        auto &conv = static_cast<fake_kpu_conv2d &>(node);
        if (allowd_pools_.find(conv.pool_type()) != allowd_pools_.end())
        {
            if (auto piece = try_get_direct_child<fake_piecewise_linear>(node))
            {
                context.inputs.emplace_back(&conv.input());
                context.outputs.emplace_back(&piece->output());

                context.matched_nodes.emplace_back(&conv);
                context.matched_nodes.emplace_back(piece);
                return true;
            }
        }
    }

    return false;
}

void fuse_fake_kpu_conv2d_piecewise_linear_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();
    auto &old_conv = static_cast<fake_kpu_conv2d &>(*context.matched_nodes[0]);
    auto &old_piece = static_cast<fake_piecewise_linear &>(*context.matched_nodes[1]);

    auto l_segs = old_conv.fused_activation();
    auto r_segs = old_piece.segments();
    auto segs = seg_segments(l_segs, r_segs);

    auto conv = context.graph.emplace<fake_kpu_conv2d>(old_conv.input().shape(), old_conv.is_depthwise(), old_conv.filter_type(), old_conv.pool_type(),
        old_conv.weights(), old_conv.bias(), segs);

    conv->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(conv->output());
}

namespace
{
bool is_max_seg(const piecewise_linear_segment &seg0, const piecewise_linear_segment &seg1)
{
    return seg0.start == std::numeric_limits<float>::lowest() && seg0.mul == 0 && seg0.add == 0
        && seg1.start == 0 && seg1.mul == 1 && seg1.add == 0;
}

bool is_min_seg(const piecewise_linear_segment &seg0, const piecewise_linear_segment &seg1)
{
    return seg0.start == std::numeric_limits<float>::lowest() && seg0.mul == 1 && seg0.add == 0
        && seg1.start == 0 && seg1.mul == 0 && seg1.add == 0;
}
}

bool revert_piecewise_linear_transform::on_try_match(node &node, transform_context &context)
{
    if (node.runtime_opcode() == op_k210_fake_piecewise_linear)
    {
        auto &piece = static_cast<fake_piecewise_linear &>(node);
        if (piece.segments().size() == 2 && (is_max_seg(piece.segments()[0], piece.segments()[1]) || is_min_seg(piece.segments()[0], piece.segments()[1])))
        {
            context.inputs.emplace_back(&piece.input());
            context.outputs.emplace_back(&piece.output());

            context.matched_nodes.emplace_back(&piece);
            return true;
        }
    }

    return false;
}

void revert_piecewise_linear_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();
    auto &old_piece = static_cast<fake_piecewise_linear &>(*context.matched_nodes[0]);

    output_connector *bin;
    if (is_max_seg(old_piece.segments()[0], old_piece.segments()[1]))
    {
        auto zero = context.graph.emplace<constant>(0.f);
        auto max = context.graph.emplace<binary>(binary_max, output.shape(), zero->output().shape(), value_range<float>::full());
        max->input_a().connect(output);
        max->input_b().connect(zero->output());
        bin = &max->output();
    }
    else if (is_min_seg(old_piece.segments()[0], old_piece.segments()[1]))
    {
        auto zero = context.graph.emplace<constant>(0.f);
        auto min = context.graph.emplace<binary>(binary_min, output.shape(), zero->output().shape(), value_range<float>::full());
        min->input_a().connect(output);
        min->input_b().connect(zero->output());
        bin = &min->output();
    }
    else
    {
        throw std::runtime_error("Unsupported piecewise linear revert");
    }

    for (auto &in : dup(inputs))
        in->connect(*bin);
}