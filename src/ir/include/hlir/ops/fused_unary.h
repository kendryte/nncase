/* Copyright 2019-2020 Canaan Inc.
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
#pragma once
#include "../graph.h"
#include "../node.h"
#include <runtime/nnil.h>
#include <xtensor/xtensor.hpp>

namespace nncase
{
namespace hlir
{
    enum fused_unary_opcode
    {
        fu_constant,
        fu_identity,
        fu_ldx,
        fu_unary,
        fu_binary,
        fu_clamp
    };

    struct fused_unary_arg
    {
        size_t op_id;
    };

    struct fused_unary_constant
    {
        float value;
    };

    struct fused_unary_identity
    {
        fused_unary_arg input;
    };

    struct fused_unary_ldx
    {
    };

    struct fused_unary_unary
    {
        unary_op_t unary_op;
        fused_unary_arg input;
    };

    struct fused_unary_binary
    {
        binary_op_t binary_op;
        fused_unary_arg input_a;
        fused_unary_arg input_b;
    };

    struct fused_unary_clamp
    {
        fused_unary_arg input;
        fused_unary_arg low;
        fused_unary_arg high;
    };

    struct fused_unary_op
    {
        fused_unary_opcode opcode;

        union {
            fused_unary_constant constant;
            fused_unary_identity identity;
            fused_unary_ldx ldx;
            fused_unary_unary unary;
            fused_unary_binary binary;
            fused_unary_clamp clamp;
        };

        static fused_unary_op make_ldx() noexcept
        {
            fused_unary_op op { fu_ldx };
            return op;
        }

        static fused_unary_op make_constant(float value) noexcept
        {
            fused_unary_op op { fu_constant };
            op.constant.value = value;
            return op;
        }

        static fused_unary_op make_unary(unary_op_t unary_op, fused_unary_arg input) noexcept
        {
            fused_unary_op op { fu_unary };
            op.unary = { unary_op, input };
            return op;
        }

        static fused_unary_op make_binary(binary_op_t binary_op, fused_unary_arg input_a, fused_unary_arg input_b) noexcept
        {
            fused_unary_op op { fu_binary };
            op.binary = { binary_op, input_a, input_b };
            return op;
        }

        static fused_unary_op make_clamp(fused_unary_arg input, fused_unary_arg low, fused_unary_arg high) noexcept
        {
            fused_unary_op op { fu_clamp };
            op.clamp = { input, low, high };
            return op;
        }

        static fused_unary_op make_identity(fused_unary_arg input) noexcept
        {
            fused_unary_op op { fu_identity };
            op.identity = { input };
            return op;
        }
    };

    std::vector<fused_unary_op> concat_subgraph(const std::vector<fused_unary_op> &src1, const std::vector<fused_unary_op> &src2);

    class fused_unary : public node
    {
    public:
        static void compile_graph(const std::vector<fused_unary_op> &subgraph, runtime::nnil_builder &builder);

        DEFINE_NODE_OPCODE(op_fused_unary);

        input_connector &input() { return input_at(0); }
        output_connector &output() { return output_at(0); }
        std::vector<fused_unary_op> &subgraph() noexcept { return subgraph_; }

        fused_unary(std::vector<fused_unary_op> subgraph, shape_t in_shape);

        void compile(hlir_compile_context &context) override;

    private:
        std::vector<fused_unary_op> subgraph_;
    };
}
}
