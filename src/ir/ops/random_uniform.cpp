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
#include <nncase/ir/op_utils.h>
#include <nncase/ir/ops/random_uniform.h>
#include <xtensor/xarray.hpp>

using namespace nncase;
using namespace nncase::ir;

random_uniform::random_uniform(datatype_t output_type, shape_t output_shape, float low, float high, float seed)
    : low_(low), high_(high), seed_(seed)
{
    add_output("output", output_type, output_shape);
}

bool random_uniform::properties_equal(node &other) const
{
    auto &r = static_cast<random_uniform &>(other);
    return (low() == r.low()) && (high() == r.high()) && (seed() == r.seed());
}
