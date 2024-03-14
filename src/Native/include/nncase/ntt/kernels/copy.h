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

namespace copy_detail {

template <typename, typename> struct impl;

// ranked shape version
template <IsRankedTensor TA, IsRankedTensor TB> struct impl<TA, TB>;

// fixed shape version
template <IsFixedTensor TA, IsFixedTensor TB> struct impl<TA, TB> {
  public:
    void operator()(const TA &input, TB &output) {
        constexpr auto a_shape = typename TA::shape_type{};
        constexpr auto a_strides = typename TA::shape_type{};
        constexpr auto b_shape = typename TB::shape_type{};
        constexpr auto b_strides = typename TB::shape_type{};
        constexpr auto conti_dims =
            std::min(contiguous_dims(a_shape, a_strides),
                     contiguous_dims(b_shape, b_strides));
        static_assert(conti_dims == 0,
                      "currently only support all contiguous ");
        std::copy(input.buffer().begin(), input.buffer().end(),
                  output.buffer().begin());
    }
};

} // namespace copy_detail

template <class TA, class TB>
void tensor_copy(const TA &input, TB &output) noexcept {
    copy_detail::impl<TA, TB> impl_;
    impl_(input, output);
}
} // namespace nncase::ntt
