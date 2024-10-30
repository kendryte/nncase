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
#pragma once
#include "../apply.h"
#include "../utility.h"

namespace nncase::ntt {

namespace detail {

std::vector<std::vector<size_t>>
continuous_dims_groups(const std::vector<size_t> &input) {
    std::vector<std::vector<size_t>> result;
    if (input.empty())
        return result;

    std::vector<size_t> currentSequence = {input[0]};

    for (size_t i = 1; i < input.size(); ++i) {
        if (input[i] != input[i - 1] + 1) {
            result.push_back(currentSequence);
            currentSequence = {input[i]};
        } else {
            currentSequence.push_back(input[i]);
        }
    }

    result.push_back(currentSequence);

    return result;
}
} // namespace detail

template <size_t Axis, typename TA, typename TB, typename TC>
void gather(const TA &input, const TB &indices, TC &&output) noexcept {
    constexpr auto rank = TA::shape_type::rank();
    using element_type = element_or_scalar_t<TA>;
    constexpr auto element_size = sizeof(element_type);

    std::vector<size_t> input_v(indices.elements().begin(),
                                indices.elements().end());
    auto result = detail::continuous_dims_groups(input_v);

    auto domain_before_axis = slice_fixed_dims<Axis>(input.shape());
    auto domain_after_axis =
        slice_fixed_dims<rank - Axis - 1, Axis + 1>(input.shape());

    auto addr_output =
        reinterpret_cast<unsigned char *>(output.buffer().data());

    auto input_conti_dims = contiguous_dims(input.shape(), input.strides());

    constexpr auto indices_rank = TB::shape_type::rank();
    constexpr auto out_shape = std::decay_t<TC>::shape();
    ranked_shape<rank> in_index;
    ranked_shape<indices_rank> indices_index;
    ranked_shape<rank> src_index;
    for (size_t i = 0; i < rank; i++) {
        src_index[i] = 0;
    }

    if (input_conti_dims == rank) {
        apply(domain_before_axis, [&](auto index) {
            for (const auto &seq : result) {
                for (size_t i = 0; i < Axis; i++) {
                    src_index[i] = index[i];
                }
                src_index[Axis] = seq[0];
                auto len =
                    seq.size() * domain_after_axis.length() * element_size;
                std::memcpy(addr_output, &(input(src_index)), len);
                addr_output += len;
            }
        });
    } else {
        apply(out_shape, [&](auto out_index) {
            // in_index[:axis] = out_index[:axis]
            loop<Axis>([&](auto i) { in_index[i] = out_index[i]; });

            // in_index[axis] = indices(indices_index)
            loop<indices_rank>(
                [&](auto i) { indices_index[i] = out_index[i + Axis]; });
            in_index[Axis] = indices(indices_index);

            // in_index[axis:] = out_index[axis:]
            loop<rank - (Axis + 1)>([&](auto i) {
                in_index[Axis + 1 + i] = out_index[Axis + indices_rank + i];
            });
            output(out_index) = input(in_index);
        });
    }
}
} // namespace nncase::ntt
