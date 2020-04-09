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
#include <cassert>
#include <hlir/graph.h>
#include <scheduler/scheduler.h>
#include <unordered_map>
#include <unordered_set>

namespace nncase
{
namespace hlir
{
    enum class quantize_stage
    {
        collect_range,
        collect_distribution,
        finish
    };

    enum class calibrate_method
    {
		no_clip,
		l2
    };

    class quantizer
    {
        class histogram
        {
        public:
            histogram(value_range<float> range, size_t src_bins, size_t dest_bins);

            void record(xtl::span<const float> data);
            void finish();
            value_range<float> optimal_range() const noexcept { return optimal_range_; }

        private:
            std::vector<float> src_bins_;
            std::vector<float> dest_bins_;
            value_range<float> range_;
            float src_bin_interval_;
            value_range<float> optimal_range_;
        };

    public:
        quantizer(calibrate_method cali_method, size_t bins);

        template <class TIt>
        static value_range<float> get_range(TIt begin, TIt end)
        {
            float min = std::numeric_limits<float>::max();
            float max = std::numeric_limits<float>::min();
            while (begin != end)
            {
                auto value = *begin++;
                auto fc = std::fpclassify(value);
                if (fc == FP_NORMAL || fc == FP_SUBNORMAL || fc == FP_ZERO)
                {
                    min = std::min(min, value);
                    max = std::max(max, value);
                }
            }

            return { min, max };
        }

        static value_range<float> fixup_range(value_range<float> range)
        {
            if (range.min < -1e3)
                range.min = -1e3;
            if (range.max > 1e3)
                range.max = 1e3;
            auto r = range.max - range.min;
            if (r == 0)
                r = 0.1f;
            else if (r < 0.01f)
                r = 0.01f;
            range.max = range.min + r;

            if (range.max < 0)
                range.max = 0;
            if (range.min > 0)
                range.min = 0;
            return range;
        }

        static quant_param_t get_quant_param(value_range<float> range, int32_t bits);
        static fixed_mul get_fixed_mul(float value, int32_t max_bits, uint8_t max_shift, bool is_signed);

        void record(hlir::output_connector &connector, value_range<float> range);
        void set(hlir::output_connector &connector, value_range<float> range);
        void record(hlir::output_connector &connector, xtl::span<const float> data);
        value_range<float> get(hlir::output_connector &connector) const;
        void broadcast_output(hlir::graph &graph, const std::unordered_set<node_opcode> &ops);
        void broadcast_output(hlir::node &node, const value_range<float> &range, const std::unordered_set<node_opcode> &ops);
        void begin_collect_distribution();
        void end_collect_distribution(std::function<void(size_t)> progress);
        size_t histograms_count() const noexcept { return histograms_.size(); }

    private:
        calibrate_method cali_method_;
        quantize_stage stage_ = quantize_stage::collect_range;
        const size_t bins_;
        std::unordered_map<hlir::output_connector *, value_range<float>> quant_ranges_;
        std::unordered_map<hlir::output_connector *, histogram> histograms_;
    };
}
}
