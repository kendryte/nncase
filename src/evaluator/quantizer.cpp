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
#include <chrono>
#include <ir/ops/constant.h>
#include <ir/ops/fake_quantize.h>
#include <ir/quantizer.h>
#include <ir/visitor.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::scheduler;

#define COMBINE_METHOD 1

namespace
{
value_range<float> combine(const value_range<float> &lhs, const value_range<float> &rhs)
{
#if COMBINE_METHOD == 0
    return { std::min(lhs.min, rhs.min), std::max(lhs.max, rhs.max) };
#elif COMBINE_METHOD == 1
    const float alpha = 0.01f;
    return { (1 - alpha) * lhs.min + alpha * rhs.min, (1 - alpha) * lhs.max + alpha * rhs.max };
#else
#error "Invalid combine method"
#endif
}
}

void quantizer::record(ir::output_connector &connector, value_range<float> range)
{
    auto it = quant_ranges_.find(&connector);
    if (it == quant_ranges_.end())
        quant_ranges_.emplace(&connector, range);
    else
        it->second = combine(it->second, range);
}

void quantizer::record(output_connector &connector, xtl::span<const float> data)
{
    record(connector, get_range(data.begin(), data.end()));
}

quant_param_t quantizer::get_quant_param(value_range<float> range, int32_t bits) const
{
    range = fixup_range(range);
    auto r = range.max - range.min;
    auto scale = ((1LL << bits) - 1) / r;
    auto bias = std::round(-range.min * scale);
    assert(bias >= 0);
    return { static_cast<int32_t>(bias), scale };
}

value_range<float> quantizer::get(ir::output_connector &connector) const
{
    return quant_ranges_.at(&connector);
}

fixed_mul quantizer::get_fixed_mul(float value, int32_t max_bits, uint8_t max_shift, bool is_signed) const
{
    assert(!is_signed || value >= 0);

    auto bits = is_signed ? max_bits - 1 : max_bits;
    int32_t shift = 0;
    float mul = 0;

    if (std::abs(value) > 1)
    {
        int mul_shift;
        mul = std::frexp(value, &mul_shift);
        shift = std::min((int32_t)max_shift, bits - mul_shift);
        mul = mul * std::pow(2, shift + mul_shift);
    }
    else if (value == 0)
    {
        mul = shift = 0;
    }
    else
    {
        int mul_shift;
        mul = std::frexp(value, &mul_shift);
        shift = std::min(max_shift + mul_shift, bits);
        mul = mul * std::pow(2, shift);
        shift -= mul_shift;
    }

    assert(std::abs(mul) < std::pow(2, bits));
    assert(shift >= 0 && shift <= max_shift);
    assert(std::abs(value - mul * std::pow(2, -shift)) <= std::numeric_limits<float>::epsilon());
    return { mul, static_cast<int8_t>(shift) };
}

void quantizer::broadcast_output(ir::graph &graph, const std::unordered_set<node_opcode> &ops)
{
    auto visitor = make_relay_ir_visitor([&](node &node) {
        if (node.inputs().size() == 1)
        {
            auto it = quant_ranges_.find(node.input_at(0).connection());
            if (it != quant_ranges_.end())
                broadcast_output(node, it->second, ops);
        }
    });
    visitor.visit(graph);
}

void quantizer::broadcast_output(ir::node &node, const value_range<float> &range, const std::unordered_set<node_opcode> &ops)
{
    if (ops.find(node.runtime_opcode()) != ops.end())
    {
        for (auto &out : node.outputs())
        {
            auto it = quant_ranges_.find(&out);
            if (it != quant_ranges_.end())
                it->second = range;

            for (auto &con : out.connections())
                broadcast_output(con->owner(), range, ops);
        }
    }
}
