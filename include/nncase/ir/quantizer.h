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
#include <cassert>
#include <nncase/ir/graph.h>
#include <unordered_map>
#include <unordered_set>

namespace nncase::ir
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
    l2,
    kld_m0,
    kld_m1,
    kld_m2,
    cdf,
    auto_select
};

class NNCASE_API quantizer
{
    class histogram
    {
    public:
        histogram(value_range<float> range, size_t src_bins, size_t dest_bins, calibrate_method cali_method);

        void record(std::span<const float> data);
        void record(std::span<const bfloat16> data);
        void record(std::span<const half> data);
        void finish();
        value_range<float> optimal_range() const noexcept { return optimal_range_; }

    private:
        std::vector<float> src_bins_;
        std::vector<float> dest_bins_;
        value_range<float> range_;
        float src_bin_interval_;
        value_range<float> optimal_range_;
        calibrate_method cali_method_;
    };

public:
    quantizer(calibrate_method cali_method, size_t bins);

    template <class TIt>
    static value_range<float> get_range(TIt begin, TIt end)
    {
        using value_t = std::decay_t<decltype(*begin)>;
        auto min = std::numeric_limits<value_t>::max();
        auto max = std::numeric_limits<value_t>::lowest();
        while (begin != end)
        {
            auto value = *begin++;
            auto fc = std::fpclassify((float)value);
            if (fc == FP_NORMAL || fc == FP_SUBNORMAL || fc == FP_ZERO)
            {
                min = std::min(min, value);
                max = std::max(max, value);
            }
        }

        return { min, max };
    }

    static value_range<float> fixup_range(value_range<float> range, bool symmetric = false)
    {
        if (symmetric)
        {
            auto r = std::max({ std::abs(range.min), std::abs(range.max), 0.01f });
            return { -r, r };
        }
        else
        {
            if (range.max < 0)
                range.max = 0;
            if (range.min > 0)
                range.min = 0;

            auto r = range.max - range.min;
            if (r == 0)
                r = 0.1f;
            // else if (r < 0.01f)
            //     r = 0.01f;
            range.max = range.min + r;
        }

        return range;
    }

    enum class quant_mode
    {
        unsigned_mode,
        signed_symmetric_mode,
        signed_asymmetric_mode
    };

    static quant_param_t get_quant_param(value_range<float> range, int32_t bits, quant_mode qm);
    static fixed_mul get_fixed_mul(float value, int32_t max_bits, uint8_t max_shift, bool is_signed);

    void record(ir::output_connector &connector, value_range<float> range);
    void set(ir::output_connector &connector, value_range<float> range);
    bool has_record(ir::output_connector &connector) const;
    void record(ir::output_connector &connector, std::span<const float> data);
    void record(ir::output_connector &connector, std::span<const bfloat16> data);
    void record(ir::output_connector &connector, std::span<const half> data);
    void record_buffers(ir::output_connector &connector, std::span<const float> data);
    void record_buffers(ir::output_connector &connector, std::span<const bfloat16> data);
    void record_buffers(ir::output_connector &connector, std::span<const half> data);
    void record_quant_buffers(ir::output_connector &connector, std::span<const float> data);
    void record_quant_buffers(ir::output_connector &connector, std::span<const bfloat16> data);
    void record_quant_buffers(ir::output_connector &connector, std::span<const half> data);
    value_range<float> get(ir::output_connector &connector) const;
    void broadcast_output(ir::graph &graph, const std::unordered_set<node_opcode> &ops);
    void broadcast_output(ir::node &node, const value_range<float> &range, const std::unordered_set<node_opcode> &ops);
    void begin_collect_distribution();
    void end_collect_distribution(std::function<void(size_t cnt, size_t total)> progress);
    size_t histograms_count() const noexcept { return histograms_.size(); }
    void end_sample() { has_record_.clear(); }
    std::unordered_map<ir::output_connector *, std::vector<float>> output_buffers() const noexcept { return output_buffers_; }
    std::vector<ir::output_connector *> quant_buffers_insert_order() const noexcept { return quant_buffers_insert_order_; }
    std::unordered_map<ir::output_connector *, value_range<float>> ranges() const noexcept { return quant_ranges_; }
    std::vector<ir::output_connector *> ranges_insert_order() const noexcept { return ranges_insert_order_; }
    void set_model_output_range(ir::graph &graph);
    value_range<float> get_model_output_range() const noexcept { return model_output_range_; }

private:
    calibrate_method cali_method_;
    quantize_stage stage_ = quantize_stage::collect_range;
    const size_t bins_;
    std::unordered_map<ir::output_connector *, value_range<float>> quant_ranges_;
    std::unordered_map<ir::output_connector *, histogram> histograms_;
    std::unordered_map<ir::output_connector *, bool> has_record_;
    std::unordered_map<ir::output_connector *, std::vector<float>> output_buffers_;
    std::vector<ir::output_connector *> quant_buffers_insert_order_;
    std::vector<ir::output_connector *> ranges_insert_order_;
    value_range<float> model_output_range_;
};
}
