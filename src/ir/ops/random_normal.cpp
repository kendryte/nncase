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
#include <nncase/ir/ops/random_normal.h>
#include <xtensor/xarray.hpp>

using namespace nncase;
using namespace nncase::ir;

random_normal::random_normal(datatype_t output_type, shape_t output_shape, float mean, float std, float seed)
    : mean_(mean), std_(std), seed_(seed)
{
    add_output("output", output_type, output_shape);
}

bool random_normal::properties_equal(node &other) const
{
    auto &r = static_cast<random_normal &>(other);
    return (mean() == r.mean()) && (std() == r.std()) && (seed() == r.seed());
}
