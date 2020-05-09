/* Copyright 2019-2020 Canaan Inc.
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
#include "../transform.h"
#include <runtime/k210/k210_runtime_op_utility.h>
#include <xtensor/xstorage.hpp>

namespace nncase::hlir::k210
{
inline bool is_supported_in_shape(const shape_t &in_shape)
{
    return in_shape[1] <= 1024 && in_shape[2] >= 4 && in_shape[2] <= 256 && in_shape[3] >= 4 && in_shape[3] <= 512;
}

inline bool is_supported_out_shape(const shape_t &in_shape)
{
    return in_shape[1] <= 1024;
}

inline bool is_bad_shape(const shape_t &in_shape, const shape_t &out_shape)
{
    return false;
}

inline bool is_supported_filter(int32_t filter_h, int32_t filter_w)
{
    return (filter_h == filter_w) && (filter_h == 3 || filter_h == 1);
}

template <bool Pre>
padding get_padding(const padding &padding)
{
    if (Pre)
        return { padding.before > 0 ? padding.before : 0, padding.after > 0 ? padding.after : 0 };
    else
        return { padding.before < 0 ? padding.before : 0, padding.after < 0 ? padding.after : 0 };
}

inline runtime::k210::kpu_filter_type_t get_filter_type(int32_t filter)
{
    using namespace runtime::k210;

    return filter == 1 ? kpu_filter_1x1 : kpu_filter_3x3;
}

inline runtime::k210::kpu_pool_type_t get_filter_type(reduce_op_t op, int32_t filter, int32_t stride)
{
    using namespace runtime::k210;

    if (op == reduce_max)
    {
        if (filter == 2)
        {
            if (stride == 2)
                return kpu_pool_max_2_s2;
            else if (stride == 1)
                return kpu_pool_max_2_s1;
        }
        else if (filter == 4)
            return kpu_pool_max_4_s4;
    }
    else if (op == reduce_mean)
    {
        if (filter == 2)
        {
            if (stride == 2)
                return kpu_pool_mean_2_s2;
            else if (stride == 1)
                return kpu_pool_mean_2_s1;
        }
        else if (filter == 4)
            return kpu_pool_mean_4_s4;
    }

    throw std::invalid_argument("Unsupported reduce window");
}

inline bool is_supported_filter(reduce_op_t op, int32_t filter_h, int32_t filter_w, int32_t stride_h, int32_t stride_w)
{
    if (filter_h != filter_w || stride_h != stride_w)
        return false;

    try
    {
        get_filter_type(op, filter_h, stride_h);
        return true;
    }
    catch (...)
    {
        return false;
    }
}
}
