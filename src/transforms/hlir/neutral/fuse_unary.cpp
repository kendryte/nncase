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
#include <hlir/ops/binary.h>
#include <hlir/ops/constant.h>
#include <hlir/ops/unary.h>
#include <hlir/transforms/neutral/fuse_unary.h>
#include <hlir/visitor.h>

using namespace nncase;
using namespace nncase::hlir;
using namespace nncase::hlir::transforms;

namespace
{
value_range<float> combine(value_range<float> act, constant &c_low, constant &c_high)
{
    auto low = *reinterpret_cast<const float *>(c_low.data().data());
    auto high = *reinterpret_cast<const float *>(c_high.data().data());
    act.min = std::max(act.min, low);
    act.max = std::max(act.max, high);
    return act;
}
}

bool fuse_unary_transform::on_try_match(node &node, transform_context &context)
{
    return false;
}

void fuse_unary_transform::process(transform_context &context)
{
}
