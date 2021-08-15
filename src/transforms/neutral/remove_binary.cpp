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
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/convert.h>
#include <nncase/ir/visitor.h>
#include <nncase/transforms/neutral/remove_binary.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

bool remove_nonsense_binary::on_try_match(node &node, transform_context &context)
{
    if (auto b = node_cast<binary>(node))
    {
        constant *ct_a = try_get_direct_parent<constant>(*b, 0);
        constant *ct_b = try_get_direct_parent<constant>(*b, 1);
        constant *ct = ct_a ? ct_a : ct_b;
        if (ct)
        {
            auto b_op = b->binary_op();
            // clang-format off
            if ((b_op == binary_add && constant_equal_to(ct, 0.f)) || \
            //  NOTE only a-0 can be remove
             (b_op == binary_sub && constant_equal_to(ct, 0.f) && ct_b) || \
             (b_op == binary_mul && constant_equal_to(ct, 1.f)) || \
            //  NOTE only a/1 can be remove
             (b_op == binary_div && constant_equal_to(ct, 1.f) && ct_b))
            {
                // clang-format on
                context.matched_nodes.emplace_back(b);
                if (ct_a) // ensure the nonconst input at inputs_0
                    context.inputs.emplace_back(&b->input_b());
                else if (ct_b)
                    context.inputs.emplace_back(&b->input_a());

                context.outputs.emplace_back(&b->output());
                return true;
            }
        }
    }

    return false;
}

bool remove_nonsense_binary::constant_equal_to(constant *node, float value) noexcept
{

#define CONSTEQ_CASE(dtype)                                         \
    if (node->data_type() == dtype)                                 \
    {                                                               \
        return this->constant_equal_to<dtype>(node->data(), value); \
    }
    // clang-format off
  CONSTEQ_CASE(dt_float32) 
  else CONSTEQ_CASE(dt_bfloat16)
  else CONSTEQ_CASE(dt_float16)
  else CONSTEQ_CASE(dt_uint64)
  else CONSTEQ_CASE(dt_uint32)
  else CONSTEQ_CASE(dt_uint8)
// clang-format on
#undef CONSTEQ_CASE
        return false;
}

void remove_nonsense_binary::process(transform_context &context)
{
    NNCASE_UNUSED auto old_b = static_cast<constant *>(context.matched_nodes[0]);
    auto &output_v = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();
    for (auto &in : dup(inputs))
        in->connect(output_v);
}
