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
#include "../ukernels/u_rope.h"

namespace nncase::ntt {
template <Tensor TInput, Tensor TCos, Tensor TSin, class TOut>
void rope(const TInput &input, const TCos &cos, const TSin &sin,
          TOut &&output) {
    using rope_layout = ukernels::rope_layout;
    // [head, dim, seq]
    const auto half_dim = input.shape()[rope_layout::dim_axis] / 2_dim;
    const auto num_heads = input.shape()[rope_layout::head_axis];
    const auto seq_len = input.shape()[rope_layout::seq_axis];

    using TElem = typename TInput::element_type;
    const TElem *NTT_RESTRICT input_p = input.elements().data();
    const TElem *NTT_RESTRICT cos_p = cos.elements().data();
    const TElem *NTT_RESTRICT sin_p = sin.elements().data();
    TElem *NTT_RESTRICT output_p = output.elements().data();

    ntt::u_rope<num_heads, half_dim>(input_p, cos_p, sin_p, output_p, seq_len,
                                     input.strides(), cos.strides(),
                                     sin.strides(), output.strides());
}
} // namespace nncase::ntt
