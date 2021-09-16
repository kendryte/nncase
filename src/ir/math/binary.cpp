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
#include <nncase/ir/math/binary.h>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/transform.hpp>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::math;

binary_node::binary_node(binary_op_t binary_op) : binary_op_(binary_op) {
    add_parameter("lhs");
    add_parameter("rhs");
}

binary::binary(binary_op_t binary_op) : object_t(std::in_place, binary_op) {}

struct to_array_tview {
    template <ranges::sized_range rng_t>
    auto operator()(rng_t &&input_view) const {
        constexpr size_t size = ranges::size(input_view);
        return impl(std::forward<rng_t>(input_view),
                    std::make_index_sequence<size>{});
    }

  private:
    template <typename rng_t, std::size_t... idx>
    static auto impl(rng_t &&input_view, std::index_sequence<idx...> const &)
        -> std::array<ranges::range_value_t<std::decay_t<rng_t>>,
                      sizeof...(idx)> {
        return {input_view[idx]...};
    }
};

template <ranges::sized_range rng_t>
auto operator|(rng_t &&input_view, to_array_tview const &to_array_tview) {
    return to_array_tview(std::forward<rng_t>(input_view));
}

to_array_tview const to_array;

type binary_node::infer_invoke_result_type(type_infer_context &context) {
    CHECK_ARGUMENT_AS_TENSOR(lhs);
    CHECK_ARGUMENT_AS_TENSOR(rhs);
    return broadcast_type(lhs_t, rhs_t);
}
