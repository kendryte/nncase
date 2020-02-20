/* Copyright 2019 Canaan Inc.
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

namespace nncase::hlir::transforms::k210
{
inline xt::svector<nncase::runtime::k210::piecewise_linear_segment> clamp_to_piecewise(value_range<float> clamp)
{
    using namespace nncase::runtime::k210;

    xt::svector<piecewise_linear_segment> segs;
    if (clamp.min != std::numeric_limits<float>::lowest())
        segs.push_back({ std::numeric_limits<float>::lowest(), 0.f, clamp.min });
    segs.push_back({ clamp.min, 1.f, 0.f });
    if (clamp.max != std::numeric_limits<float>::max())
        segs.push_back({ clamp.max, 0.f, clamp.max });
    return segs;
}
}
