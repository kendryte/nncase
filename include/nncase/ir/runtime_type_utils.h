/* Copyright 2020 Canaan Inc.
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
#include "ir_types.h"

namespace nncase::ir
{
template <class T>
runtime_shape_t to(const xt::dynamic_shape<T> &in_shape, T default_val = 1)
{
    assert(in_shape.size() <= 4);

    runtime_shape_t r_in_shape {};
    const auto in_ext = 4 - (int32_t)in_shape.size();

    for (int32_t i = 0; i < in_ext; i++)
        r_in_shape[i] = int32_t(default_val);
    for (size_t i = in_ext; i < 4; i++)
        r_in_shape[i] = int32_t(in_shape[i - in_ext]);
    return r_in_shape;
}

inline runtime_paddings_t to(const xt::svector<padding> &paddings)
{
    assert(paddings.size() <= 4);

    runtime_paddings_t r_paddings;
    const auto in_ext = 4 - (int32_t)paddings.size();

    for (int32_t i = 0; i < in_ext; i++)
        r_paddings[i] = padding::zero();
    for (size_t i = in_ext; i < 4; i++)
        r_paddings[i] = paddings[i - in_ext];
    return r_paddings;
}

inline void extend_transpose_shape(const shape_t &in_shape, const axis_t &perm, runtime_shape_t &r_in_shape, runtime_shape_t &r_perm)
{
    assert(perm.size() <= 4);

    const auto in_ext = 4 - in_shape.size();
    const auto perm_ext = 4 - perm.size();
    r_in_shape = to(in_shape);

    for (size_t i = 0; i < perm_ext; i++)
        r_perm[i] = int32_t(i);
    for (size_t i = 0; i < perm.size(); i++)
        r_perm[i + perm_ext] = int32_t(perm[i] + in_ext);
}
}
