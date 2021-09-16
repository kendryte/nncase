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
#include <nncase/ir/type_infer.h>
#include <range/v3/algorithm/all_of.hpp>
#include <range/v3/view/transform.hpp>

using namespace nncase;
using namespace nncase::ir;

type ir::broadcast_type(std::initializer_list<tensor_type> inputs) {
    assert(inputs.size() >= 2);
    auto dtype = (*inputs.begin())->dtype();
    CHECK_TYPE(
        ranges::all_of(
            inputs, [=](const tensor_type &t) { return t->dtype() == dtype; }),
        "inputs must have same dtype");

    // If any input is not fixed, result is unranked
    if (ranges::any_of(inputs, [=](const tensor_type &t) {
            return !t->shape().is_fixed();
        }))
        return tensor_type(dtype, unranked_shape);

    shape_t out_shape(scalar_shape);
    const auto dest_rank =
        ranges::max(inputs | ranges::views::transform([](const tensor_type &t) {
                        return *t->shape().rank();
                    }));

    for (size_t dim_idx = 0; dim_idx < dest_rank; dim_idx++) {
        itlib::small_vector<size_t> in_dims;
        in_dims.reserve(inputs.size());
        for (auto it = inputs.begin(); it != inputs.end(); ++it) {
            const auto &in_shape = (*it)->shape();
            const auto in_extend = dest_rank - *in_shape.rank();
            auto in_dim = dim_value_t(dim_idx - in_extend);
            in_dim = in_dim < 0 ? 1 : in_shape[in_dim].fixed_value();
            assert(in_dim != 0);
            in_dims.emplace_back(in_dim);
        }

        // 1. Sort descending
        std::sort(in_dims.begin(), in_dims.end(), std::greater<>());
        assert(in_dims.front() > 0);
        // 2. Find first 1
        auto first_one = std::find(in_dims.begin(), in_dims.end(), 1);
        auto expected_dim = in_dims.front();
        // 3. Dims before 1 are all same or 1 is not found, it's ok to broadcast
        if (first_one == in_dims.end() ||
            std::all_of(in_dims.begin(), first_one,
                        [=](size_t dim) { return dim == expected_dim; })) {
            out_shape.emplace_back(expected_dim);
        } else {
            return invalid_type("inputs are not compatible to broadcast");
        }
    }

    return tensor_type(dtype, out_shape);
}
