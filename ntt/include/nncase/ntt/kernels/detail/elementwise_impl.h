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
#include "../../loop.h"
#include "../../tensor_traits.h"
#include "../../utility.h"
#include <tuple>
#include <utility>

namespace nncase::ntt::detail {
template <class TDerived, Tensor TOutput, Tensor... TInputs>
class elementwise_impl {
  public:
    template <class... TArgs>
    constexpr void operator()(const TInputs &...inputs, TOutput &output,
                              TArgs &&...args) {
        apply_broadcasted(
            std::make_tuple(inputs.broadcast_to(output.shape())...), output,
            std::make_index_sequence<sizeof...(TInputs)>(),
            std::forward<TArgs>(args)...);
    }

  private:
    template <Tensor... TBroadcastedInputs, size_t... I, class... TArgs>
    constexpr void
    apply_broadcasted(const std::tuple<TBroadcastedInputs...> &inputs,
                      TOutput &output, std::index_sequence<I...>,
                      TArgs &&...args) {
        derived().apply(std::get<I>(inputs)..., output,
                        std::forward<TArgs>(args)...);
    }

  protected:
    TDerived &derived() noexcept { return static_cast<TDerived &>(*this); }
};
} // namespace nncase::ntt::detail
