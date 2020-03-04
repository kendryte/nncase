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
#include <hlir/transforms/k210/piecewise_regression.h>

using namespace nncase::hlir::transforms::k210;

piecewise_regression::piecewise_regression(size_t segments_count)
    : desired_segments_count_(segments_count)
{
}

std::vector<segment> piecewise_regression::fit(std::vector<point> &points) const
{
    if (points.size() <= desired_segments_count_)
        throw std::invalid_argument("Insufficient points");

    std::sort(points.begin(), points.end(), [](const point &a, const point &b) {
        return a.x < b.x;
    });

    // 1. initialize segments
    std::vector<segment> segments(points.size() - 1);
    for (size_t i = 0; i < points.size() - 1; i++)
    {
        const auto &p0 = points[i];
        const auto &p1 = points[i + 1];
        segments[i] = { p0.x, p1.x, (p1.y - p0.y) / (p1.x - p0.x), p0.y };
    }

    // 2. combine sibling segments
    while (segments.size() != desired_segments_count_)
    {
        // 2.1 find min slope difference
        float min_diff = std::numeric_limits<float>::max();
        size_t min_idx = -1;
        for (size_t i = 0; i < segments.size() - 1; i++)
        {
            const auto &s0 = segments[i];
            const auto &s1 = segments[i + 1];
            auto diff = std::abs(s0.slop - s1.slop);
            if (diff < min_diff)
            {
                min_diff = diff;
                min_idx = i;
            }
        }

        // 2.2 combine
        auto &s0 = segments[min_idx];
        auto &s1 = segments[min_idx + 1];
        auto y0 = s0.y(s0.start);
        auto y1 = s1.y(s1.stop);
        auto slope = (y1 - y0) / (s1.stop - s0.start);
        s0.slop = slope;
        s0.stop = s1.stop;
        segments.erase(segments.begin() + min_idx + 1);
    }

    return segments;
}
