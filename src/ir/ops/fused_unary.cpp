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
#include <magic_enum.hpp>
#include <nncase/ir/op_utils.h>
#include <nncase/ir/ops/fused_unary.h>
#include <xtensor/xarray.hpp>

using namespace nncase;
using namespace nncase::ir;

namespace
{
void calc_op_used_count(const std::vector<fused_unary_op> subgraph, size_t root, std::vector<size_t> &sequence, std::vector<size_t> &access_counts)
{
    auto access = access_counts[root]++;
    if (access == 0)
    {
        auto &op = subgraph[root];
        switch (op.opcode)
        {
        case fu_constant:
        case fu_ldx:
            break;
        case fu_identity:
            calc_op_used_count(subgraph, op.identity.input.op_id, sequence, access_counts);
            break;
        case fu_unary:
            calc_op_used_count(subgraph, op.unary.input.op_id, sequence, access_counts);
            break;
        case fu_binary:
            calc_op_used_count(subgraph, op.binary.input_a.op_id, sequence, access_counts);
            calc_op_used_count(subgraph, op.binary.input_b.op_id, sequence, access_counts);
            break;
        case fu_clamp:
            calc_op_used_count(subgraph, op.clamp.input.op_id, sequence, access_counts);
            calc_op_used_count(subgraph, op.clamp.low.op_id, sequence, access_counts);
            calc_op_used_count(subgraph, op.clamp.high.op_id, sequence, access_counts);
            break;
        default:
            throw std::invalid_argument("Invalid fused unary op");
        }

        sequence.emplace_back(root);
    }
}
}

void fused_unary::compile_graph(const std::vector<fused_unary_op> &subgraph, codegen::nnil_builder &builder)
{
    std::vector<size_t> sequence;
    std::vector<size_t> access_counts(subgraph.size());
    calc_op_used_count(subgraph, subgraph.size() - 1, sequence, access_counts);

    for (auto &id : sequence)
    {
        auto &op = subgraph[id];
        auto access = access_counts[id];

        if (!access)
            continue;

        switch (op.opcode)
        {
        case fu_constant:
            builder.emit_ldc_r4(op.constant.value);
            break;
        case fu_identity:
            builder.emit_dup();
            break;
        case fu_ldx:
            builder.emit_lda_0();
            break;
        case fu_unary:
        {
            switch (op.unary.unary_op)
            {
            case unary_abs:
                builder.emit_abs();
                break;
            case unary_acos:
                builder.emit_acos();
                break;
            case unary_asin:
                builder.emit_asin();
                break;
            case unary_ceil:
                builder.emit_ceil();
                break;
            case unary_cos:
                builder.emit_cos();
                break;
            case unary_exp:
                builder.emit_exp();
                break;
            case unary_floor:
                builder.emit_floor();
                break;
            case unary_log:
                builder.emit_log();
                break;
            case unary_neg:
                builder.emit_neg();
                break;
            case unary_round:
                builder.emit_round();
                break;
            case unary_rsqrt:
                builder.emit_rsqrt();
                break;
            case unary_sign:
                builder.emit_sign();
                break;
            case unary_sin:
                builder.emit_sin();
                break;
            case unary_sqrt:
                builder.emit_sqrt();
                break;
            case unary_square:
                builder.emit_square();
                break;
            case unary_tanh:
                builder.emit_tanh();
                break;
            case unary_bitwise_not:
                builder.emit_bitwise_not();
                break;
            case unary_logical_not:
                builder.emit_logical_not();
                break;
            case unary_erf:
                builder.emit_erf();
                break;
            default:
                throw std::invalid_argument("Unsupported unary op for nnil: " + (std::string)magic_enum::enum_name(op.unary.unary_op));
            }
            break;
        }
        case fu_binary:
        {
            switch (op.binary.binary_op)
            {
            case binary_add:
                builder.emit_add();
                break;
            case binary_sub:
                builder.emit_sub();
                break;
            case binary_mul:
                builder.emit_mul();
                break;
            case binary_div:
                builder.emit_div();
                break;
            case binary_min:
                builder.emit_min();
                break;
            case binary_max:
                builder.emit_max();
                break;
            case binary_pow:
                builder.emit_pow();
                break;
            default:
                throw std::invalid_argument("Unsupported binary op: " + (std::string)magic_enum::enum_name(op.binary.binary_op));
            }
            break;
        }
        case fu_clamp:
            builder.emit_clamp();
            break;
        default:
            throw std::invalid_argument("Invalid fused unary op");
        }

        for (size_t i = 0; i < access - 1; i++)
            builder.emit_dup();
    }

    builder.emit_ret();
}

fused_unary::fused_unary(std::vector<fused_unary_op> subgraph, datatype_t in_type, shape_t in_shape)
    : subgraph_(std::move(subgraph))
{
    add_input("input", in_type, in_shape);
    add_output("output", in_type, in_shape);
}

std::vector<fused_unary_op> ir::concat_subgraph(const std::vector<fused_unary_op> &src1, const std::vector<fused_unary_op> &src2)
{
    std::vector<fused_unary_op> result = src1;

    // Turn ldx to identity
    auto src1_out_arg = result.size() - 1;
    auto arg_inc = result.size();
    auto first_ldx = true;
    for (auto &op : src2)
    {
        auto n_op = op;
        switch (n_op.opcode)
        {
        case fu_ldx:
            if (!first_ldx)
            {
                result.emplace_back(fused_unary_op::make_identity({ src1_out_arg }));
            }
            else
            {
                first_ldx = false;
                arg_inc--;
            }
            continue;
        case fu_constant:
            break;
        case fu_unary:
            n_op.unary.input.op_id += arg_inc;
            break;
        case fu_binary:
            n_op.binary.input_a.op_id += arg_inc;
            n_op.binary.input_b.op_id += arg_inc;
            break;
        case fu_clamp:
            n_op.clamp.input.op_id += arg_inc;
            n_op.clamp.low.op_id += arg_inc;
            n_op.clamp.high.op_id += arg_inc;
            break;
        default:
            throw std::runtime_error("Invalid fused unary op");
        }

        result.emplace_back(n_op);
    }

    return result;
}

bool fused_unary::properties_equal(node &other) const
{
    [[maybe_unused]] auto &r = static_cast<fused_unary &>(other);
    // TODO: compare subgraph
    return false;
}
