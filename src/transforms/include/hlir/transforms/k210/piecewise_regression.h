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
#include <hlir/quantizer.h>

namespace nncase::hlir::transforms::k210
{
struct point
{
    float x;
    float y;
};

struct segment
{
    float start;
    float stop;
    float slop;
    float intercept;

    constexpr float y(float x) const noexcept
    {
        assert(x >= start && x <= stop);
        return (x - start) * slop + intercept;
    }
};

class piecewise_regression
{
public:
    piecewise_regression(size_t segments_count);

    std::vector<segment> fit(std::vector<point> &points) const;

private:
    size_t desired_segments_count_;
};
}
