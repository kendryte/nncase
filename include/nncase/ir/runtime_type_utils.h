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
#include "ir_types.h"

namespace nncase::ir
{
template <class T = size_t, class U>
itlib::small_vector<T, 4> to(const xt::dynamic_shape<U> &in_shape, [[maybe_unused]] T default_val = 1)
{
    itlib::small_vector<T, 4> r_in_shape(in_shape.size());
    for (size_t i = 0; i < in_shape.size(); i++)
        r_in_shape[i] = (T)in_shape[i];
    return r_in_shape;
}

inline itlib::small_vector<padding, 4> to(const xt::svector<padding> &paddings)
{
    itlib::small_vector<padding, 4> result(paddings.size());
    for (size_t i = 0; i < paddings.size(); i++)
        result[i] = paddings[i];
    return result;
}

inline void extend_transpose_shape(const shape_t &in_shape, const axis_t &perm, runtime_shape_t &r_in_shape, runtime_shape_t &r_perm)
{
    r_in_shape = to(in_shape);
    r_perm.resize(perm.size());

    for (size_t i = 0; i < perm.size(); i++)
    {
        auto value = perm[i];
        if (value < 0)
            r_perm[i] = (size_t)((int32_t)perm.size() + value);
        else
            r_perm[i] = (size_t)value;
    }
}
}
